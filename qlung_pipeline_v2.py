"""Leakage-safe QLung-AST V2 for ICBHI 2017.

V2 keeps qlung_pipeline.py as the frozen V1 baseline and changes only the
parts for which V1 diverged from the paper or lacked a validation protocol:

* 798-frame AST inputs for an 8-second/10-ms-hop signal;
* bicubic interpolation of the pretrained AST positional patch grid;
* train-only calibration of AQS to the paper's mid-quality range;
* patient-disjoint five-fold tuning of epoch and a single Normal-logit bias;
* full-official-train final models using only OOF-selected hyperparameters;
* one no-TTA official-test evaluation after all final models are ready.

V2 is self-contained apart from this repository's dataset.py and metrics.py.
Do not overwrite the V1 script, checkpoints, caches, or evaluation output.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

from dataset import CLASS_NAMES, ICBHIDataset, safe_torch_load
from metrics import calc_icbhi_score, calc_legacy_macro_ovr_score


AUDIO_DIR = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
CSV_PATH = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
MODEL_DIR = "pretrained/ast_audioset"
INPUT_LENGTH = 798
CACHE_DIR = "precomputed/qlung_ast_v2"
TRAIN_CACHE = f"{CACHE_DIR}/ast_train_len798.pth"
TEST_CACHE = f"{CACHE_DIR}/ast_test_len798.pth"
TUNE_DIR = "runs/qlung_ast_v2_tune"
TUNING_SUMMARY = f"{TUNE_DIR}/tuning_summary.json"
FINAL_DIR = "runs/qlung_ast_v2"

CACHE_VERSION = 2
CHECKPOINT_VERSION = 2
AQS_TARGET_MEAN = 0.45
AQS_TARGET_STD = 0.08

# QLung paper settings.  They live in V2 instead of being imported from V1 so
# this file works with both versions of qlung_pipeline.py that have circulated
# during the experiments.
LAMBDA_DFAM = 0.4
GAMMA = 0.5
TARGET_MARGIN = 0.2
ANGULAR_SCALE = 37.0
DFAM_SCALE = 15.0
QUALITY_SCALE = 0.5


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scaler_for(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def spec_augment(
    features: torch.Tensor,
    frequency_width: int = 48,
    time_width: int = 192,
) -> torch.Tensor:
    augmented = features.clone()
    _, time_steps, frequency_bins = augmented.shape
    for item in augmented:
        width = int(
            torch.randint(
                0,
                min(frequency_width, frequency_bins) + 1,
                (1,),
            ).item()
        )
        if width:
            start = int(
                torch.randint(
                    0,
                    frequency_bins - width + 1,
                    (1,),
                ).item()
            )
            item[:, start : start + width] = 0
        width = int(
            torch.randint(
                0,
                min(time_width, time_steps) + 1,
                (1,),
            ).item()
        )
        if width:
            start = int(
                torch.randint(
                    0,
                    time_steps - width + 1,
                    (1,),
                ).item()
            )
            item[start : start + width] = 0
    return augmented


class AngularClassifier(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_classes: int = 4,
        scale: float = ANGULAR_SCALE,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dimension))
        self.scale = float(scale)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor):
        normalized_features = F.normalize(features, dim=1)
        normalized_weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_features, normalized_weights).clamp(-1, 1)
        return self.scale * cosine, cosine


class QLungLoss(nn.Module):
    def __init__(self, counts: torch.Tensor) -> None:
        super().__init__()
        counts = counts.detach().float()
        if counts.ndim != 1 or len(counts) != 4 or bool((counts <= 0).any()):
            raise ValueError(f"四类训练样本数量必须为正数，得到 {counts.tolist()}")
        frequency = counts / counts.sum()
        class_scale = TARGET_MARGIN / math.log(len(counts))
        self.register_buffer("class_margin", class_scale * (-frequency.log()))

    def forward(self, logits, cosine, labels, quality):
        classification_loss = F.cross_entropy(logits, labels)
        quality_margin = QUALITY_SCALE * quality.float().clamp(0, 1)
        class_margin = self.class_margin[labels]
        margin = GAMMA * quality_margin + (1 - GAMMA) * class_margin

        stable_cosine = cosine.float().clamp(-1 + 1e-6, 1 - 1e-6)
        target_cosine = stable_cosine.gather(1, labels[:, None]).squeeze(1)
        target_sine = (1 - target_cosine.square()).clamp_min(1e-7).sqrt()
        margin_target = (
            target_cosine * margin.cos() - target_sine * margin.sin()
        )
        dfam_logits = DFAM_SCALE * stable_cosine
        dfam_logits = dfam_logits.scatter(
            1,
            labels[:, None],
            (DFAM_SCALE * margin_target)[:, None],
        )
        dfam_loss = F.cross_entropy(dfam_logits, labels)
        total = classification_loss + LAMBDA_DFAM * dfam_loss
        return total, classification_loss, dfam_loss, margin.mean()


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sum = matrix.sum(1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sum != 0,
    )
    annotation = np.empty_like(matrix, dtype=object)
    for row in range(4):
        for column in range(4):
            annotation[row, column] = (
                f"{normalized[row, column]:.1%}\n({matrix[row, column]})"
            )
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        normalized,
        annot=annotation,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(
        title="QLung AST V2 Official Test (No TTA)",
        xlabel="Predicted",
        ylabel="True",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_grid(config, input_length: int) -> tuple[int, int]:
    patch_size = int(config.patch_size)
    frequency_patches = (
        int(config.num_mel_bins) - patch_size
    ) // int(config.frequency_stride) + 1
    time_patches = (
        int(input_length) - patch_size
    ) // int(config.time_stride) + 1
    if frequency_patches <= 0 or time_patches <= 0:
        raise ValueError("AST patch 网格无效。")
    return frequency_patches, time_patches


def resize_pretrained_position_embeddings(model: ASTModel, input_length: int) -> None:
    """Interpolate AST patch positions from the pretrained grid to V2 grid."""

    embeddings = model.embeddings
    position = embeddings.position_embeddings.detach()
    old_length = int(model.config.max_length)
    old_frequency, old_time = patch_grid(model.config, old_length)
    new_frequency, new_time = patch_grid(model.config, input_length)
    expected_tokens = 2 + old_frequency * old_time
    if position.ndim != 3 or position.size(1) != expected_tokens:
        raise RuntimeError(
            f"预训练位置编码形状异常: {tuple(position.shape)}, "
            f"expected tokens={expected_tokens}"
        )

    special_tokens = position[:, :2]
    patch_tokens = position[:, 2:].reshape(
        1,
        old_frequency,
        old_time,
        position.size(-1),
    )
    patch_tokens = patch_tokens.permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens.float(),
        size=(new_frequency, new_time),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=position.dtype)
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(
        1,
        new_frequency * new_time,
        position.size(-1),
    )
    embeddings.position_embeddings = nn.Parameter(
        torch.cat([special_tokens, patch_tokens], dim=1)
    )
    model.config.max_length = int(input_length)


class QLungASTV2(nn.Module):
    def __init__(
        self,
        model_dir: Union[str, os.PathLike],
        input_length: int = INPUT_LENGTH,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        model_dir = str(Path(model_dir).expanduser().resolve())
        if pretrained:
            self.ast = ASTModel.from_pretrained(
                model_dir,
                local_files_only=True,
                use_safetensors=True,
            )
            resize_pretrained_position_embeddings(self.ast, input_length)
        else:
            config = ASTConfig.from_pretrained(model_dir, local_files_only=True)
            config.max_length = int(input_length)
            self.ast = ASTModel(config)
        self.head = AngularClassifier(int(self.ast.config.hidden_size))
        frequency_patches, time_patches = patch_grid(self.ast.config, input_length)
        actual_tokens = int(self.ast.embeddings.position_embeddings.size(1))
        expected_tokens = 2 + frequency_patches * time_patches
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"V2 position tokens={actual_tokens}, expected={expected_tokens}"
            )

    def forward(self, input_values: torch.Tensor):
        output = self.ast(input_values=input_values, return_dict=True)
        logits, cosine = self.head(output.pooler_output)
        return logits, cosine


@torch.inference_mode()
def raw_audio_quality(waveforms: torch.Tensor) -> torch.Tensor:
    """Compute a rank-preserving raw AQS before train-only calibration.

    Entropy is calculated over the complete time-frequency power distribution,
    unlike V1's mean frame entropy. It is normalized by log(F*T), then combined
    with full-scale RMS using the QLung formula.
    """

    values = waveforms.squeeze(1).float()
    window = torch.hann_window(400, dtype=values.dtype, device=values.device)
    spectrum = torch.stft(
        values,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spectrum.abs().square().clamp_min(1e-12)
    probability = power / power.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(dim=(1, 2))
    entropy = entropy / math.log(power.size(1) * power.size(2))
    rms = values.square().mean(dim=1).sqrt().clamp(0.0, 1.0)
    return (1.0 - 0.7 * entropy.clamp(0.0, 1.0) + 0.3 * rms).clamp(0.0, 1.0)


def calibrate_quality(
    raw_quality: torch.Tensor,
    fit_indices: Union[Sequence[int], torch.Tensor, np.ndarray],
    target_mean: float,
    target_std: float,
) -> tuple[torch.Tensor, dict]:
    """Fit an affine AQS calibration using training indices only."""

    indices = torch.as_tensor(fit_indices, dtype=torch.long)
    fitted = raw_quality[indices].float()
    source_mean = float(fitted.mean())
    source_std = float(fitted.std(unbiased=False))
    if not math.isfinite(source_std) or source_std < 1e-6:
        raise RuntimeError(f"raw AQS 方差过小: {source_std}")
    calibrated = target_mean + target_std * (
        raw_quality.float() - source_mean
    ) / source_std
    calibrated = calibrated.clamp(0.0, 1.0)
    fitted_calibrated = calibrated[indices]
    statistics = {
        "source_mean": source_mean,
        "source_std": source_std,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "fitted_mean_after_clip": float(fitted_calibrated.mean()),
        "fitted_std_after_clip": float(fitted_calibrated.std(unbiased=False)),
        "fitted_min": float(fitted_calibrated.min()),
        "fitted_max": float(fitted_calibrated.max()),
    }
    return calibrated, statistics


def prepare_split(args, split: str, extractor: ASTFeatureExtractor) -> None:
    destination = Path(args.cache_dir) / f"ast_{split}_len{args.input_length}.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not args.force:
        print(f"V2 缓存已存在，跳过: {destination}")
        return

    source = ICBHIDataset(
        data_path=args.audio_dir,
        split=split,
        metadatafile=args.csv_path,
        duration=8.0,
        samplerate=16000,
        cache=True,
    )
    shape = (len(source), args.input_length, int(extractor.num_mel_bins))
    features = torch.empty(shape, dtype=torch.float16)
    raw_quality = torch.empty(len(source), dtype=torch.float32)
    iterator = range(0, len(source), args.feature_batch_size)
    for start in tqdm(iterator, desc=f"V2 prepare [{split}]"):
        stop = min(start + args.feature_batch_size, len(source))
        waveforms = source.data[start:stop].float()
        raw_audio = [waveform.squeeze(0).numpy() for waveform in waveforms]
        encoded = extractor(
            raw_audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values
        expected = (stop - start, args.input_length, int(extractor.num_mel_bins))
        if tuple(encoded.shape) != expected:
            raise RuntimeError(f"V2 AST 特征 {tuple(encoded.shape)} != {expected}")
        features[start:stop].copy_(encoded.half())
        raw_quality[start:stop].copy_(raw_audio_quality(waveforms))

    torch.save(
        {
            "cache_version": CACHE_VERSION,
            "split": split,
            "input_length": int(args.input_length),
            "features": features,
            "labels": source.labels.long().clone(),
            "raw_quality": raw_quality,
            "patient_ids": list(source.patient_ids),
            "sample_ids": list(source.sample_ids),
            "quality_definition": "global TF entropy + RMS; calibrated on train only",
        },
        destination,
    )
    print(
        f"保存: {destination} | shape={shape} | raw AQS "
        f"min/mean/max={raw_quality.min():.4f}/{raw_quality.mean():.4f}/"
        f"{raw_quality.max():.4f}"
    )


def run_prepare(args) -> None:
    extractor = ASTFeatureExtractor.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    extractor.max_length = int(args.input_length)
    print(
        f"V2 extractor | length={extractor.max_length} | "
        f"mel={extractor.num_mel_bins} | mean={extractor.mean} | std={extractor.std}"
    )
    for split in args.splits:
        prepare_split(args, split, extractor)


class FeatureCache(Dataset):
    def __init__(
        self,
        filename: Union[str, os.PathLike],
        expected_split: str,
    ) -> None:
        self.path = Path(filename).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 V2 缓存: {self.path}，请先运行 prepare。")
        data = safe_torch_load(self.path)
        if int(data.get("cache_version", -1)) != CACHE_VERSION:
            raise ValueError("V2 缓存版本不匹配。")
        if data.get("split") != expected_split:
            raise ValueError(f"缓存 split={data.get('split')}, expected={expected_split}")
        self.input_length = int(data["input_length"])
        self.features = data["features"].contiguous()
        self.labels = data["labels"].long().contiguous()
        self.raw_quality = data["raw_quality"].float().contiguous()
        self.patient_ids = [str(value) for value in data["patient_ids"]]
        self.sample_ids = [str(value) for value in data["sample_ids"]]
        expected_shape = (len(self.labels), self.input_length, 128)
        if tuple(self.features.shape) != expected_shape:
            raise ValueError(f"V2 特征 {tuple(self.features.shape)} != {expected_shape}")
        if not (
            len(self.labels)
            == len(self.raw_quality)
            == len(self.patient_ids)
            == len(self.sample_ids)
        ):
            raise ValueError("V2 缓存字段长度不一致。")
        print(
            f"加载 V2 {expected_split}: samples={len(self)} | "
            f"shape={tuple(self.features.shape)} | "
            f"classes={torch.bincount(self.labels, minlength=4).tolist()}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.raw_quality[index]


class FeatureView(Dataset):
    def __init__(
        self,
        base: FeatureCache,
        indices: Union[Sequence[int], np.ndarray],
        calibrated_quality: torch.Tensor,
    ) -> None:
        self.base = base
        self.indices = [int(value) for value in indices]
        self.quality = calibrated_quality.float()
        self.labels = base.labels[self.indices]
        self.patient_ids = [base.patient_ids[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        return (
            self.base.features[source_index],
            self.base.labels[source_index],
            self.quality[source_index],
        )


def search_patient_folds(
    dataset: FeatureCache,
    num_folds: int,
    seed: int,
    max_candidates: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, float]:
    indices = np.arange(len(dataset), dtype=np.int64)
    labels = dataset.labels.numpy().astype(np.int64)
    patients = np.asarray(dataset.patient_ids, dtype=object)
    expected_classes = set(range(4))
    all_patients = set(patients.tolist())
    target_distribution = np.bincount(labels, minlength=4).astype(np.float64)
    target_distribution /= target_distribution.sum()
    target_ratio = 1.0 / num_folds
    best = None
    valid_count = 0

    for offset in range(max_candidates):
        split_seed = seed + offset
        splitter = StratifiedGroupKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=split_seed,
        )
        folds = []
        seen_validation_patients = set()
        objective = 0.0
        valid = True
        for train_indices, validation_indices in splitter.split(
            indices,
            labels,
            patients,
        ):
            train_indices = np.asarray(train_indices, dtype=np.int64)
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            train_patients = set(patients[train_indices].tolist())
            validation_patients = set(patients[validation_indices].tolist())
            if train_patients.intersection(validation_patients):
                valid = False
                break
            if seen_validation_patients.intersection(validation_patients):
                valid = False
                break
            if set(labels[train_indices].tolist()) != expected_classes:
                valid = False
                break
            if set(labels[validation_indices].tolist()) != expected_classes:
                valid = False
                break
            seen_validation_patients.update(validation_patients)
            distribution = np.bincount(
                labels[validation_indices], minlength=4
            ).astype(np.float64)
            distribution /= distribution.sum()
            objective += float(np.abs(distribution - target_distribution).sum())
            objective += 2.0 * abs(len(validation_indices) / len(indices) - target_ratio)
            folds.append((train_indices, validation_indices))
        if not valid or seen_validation_patients != all_patients:
            continue
        if len(folds) != num_folds:
            continue
        valid_count += 1
        candidate = (objective, split_seed, folds)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if valid_count >= 25:
            break

    if best is None:
        raise RuntimeError("无法生成每折四类齐全、患者互斥的 V2 划分。")
    objective, split_seed, folds = best
    return folds, int(split_seed), float(objective)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    device,
    amp_enabled: bool,
    description: str,
) -> dict:
    model.train()
    sums = np.zeros(4, dtype=np.float64)
    sample_count = 0
    skipped = 0
    progress = tqdm(loader, desc=description, leave=False)
    for features, labels, quality in progress:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        if device.type == "cpu":
            features = features.float()
        features = spec_augment(features)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, cosine = model(features)
            loss, ce_loss, dfam_loss, margin = criterion(
                logits,
                cosine,
                labels,
                quality,
            )
        if not torch.isfinite(loss):
            skipped += 1
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        current = int(labels.size(0))
        sample_count += current
        sums += np.asarray(
            [
                float(loss.detach()),
                float(ce_loss.detach()),
                float(dfam_loss.detach()),
                float(margin.detach()),
            ]
        ) * current
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}", skipped=skipped)
    if sample_count == 0:
        raise RuntimeError("V2 epoch 没有成功更新任何 batch。")
    means = sums / sample_count
    return {
        "loss": float(means[0]),
        "ce": float(means[1]),
        "dfam": float(means[2]),
        "margin": float(means[3]),
        "skipped": int(skipped),
    }


@torch.inference_mode()
def validation_logits(model, loader, device, amp_enabled: bool):
    model.eval()
    logits_all = []
    labels_all = []
    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        if device.type == "cpu":
            features = features.float()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, _ = model(features)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.long().cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def predictions_with_normal_bias(logits: torch.Tensor, bias: float) -> list[int]:
    adjusted = logits.float().clone()
    adjusted[:, 0] += float(bias)
    return adjusted.argmax(dim=1).tolist()


def tune_normal_bias(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bias_min: float,
    bias_max: float,
    bias_steps: int,
) -> tuple[float, object]:
    if bias_steps < 2 or bias_max <= bias_min:
        raise ValueError("Normal bias 搜索范围无效。")
    labels_list = labels.long().tolist()
    best_bias = 0.0
    best_result = None
    for bias in np.linspace(bias_min, bias_max, bias_steps):
        prediction = predictions_with_normal_bias(logits, float(bias))
        result = calc_icbhi_score(labels_list, prediction)
        if best_result is None:
            best_bias, best_result = float(bias), result
            continue
        if result.score > best_result.score + 1e-12:
            best_bias, best_result = float(bias), result
        elif math.isclose(result.score, best_result.score, abs_tol=1e-12):
            if abs(float(bias)) < abs(best_bias):
                best_bias, best_result = float(bias), result
    return best_bias, best_result


def clone_cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def run_tune(args) -> None:
    if not 0 <= args.fold < args.num_folds:
        raise ValueError("fold 必须位于 [0, num_folds)。")
    fold_seed = args.seed + args.fold * 1009
    seed_all(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output = Path(args.output_dir) / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)

    cache = FeatureCache(args.train_cache, "train")
    if cache.input_length != args.input_length:
        raise ValueError(
            f"V2 train cache input_length={cache.input_length}，"
            f"但命令要求 {args.input_length}。请重新 prepare 或修正参数。"
        )
    folds, split_seed, objective = search_patient_folds(
        cache,
        args.num_folds,
        args.seed,
        args.split_candidates,
    )
    train_indices, validation_indices = folds[args.fold]
    calibrated_quality, calibration = calibrate_quality(
        cache.raw_quality,
        train_indices,
        args.aqs_target_mean,
        args.aqs_target_std,
    )
    train_set = FeatureView(cache, train_indices, calibrated_quality)
    validation_set = FeatureView(cache, validation_indices, calibrated_quality)
    train_patients = set(train_set.patient_ids)
    validation_patients = set(validation_set.patient_ids)
    if train_patients.intersection(validation_patients):
        raise RuntimeError("V2 tune 训练与验证患者重叠。")

    print("=" * 76)
    print(
        f"V2 tune fold={args.fold}/{args.num_folds - 1} | seed={fold_seed} | "
        f"device={device} | AMP={amp_enabled}"
    )
    print(f"split_seed={split_seed} | objective={objective:.4f}")
    print(
        f"train={torch.bincount(train_set.labels, minlength=4).tolist()} | "
        f"validation={torch.bincount(validation_set.labels, minlength=4).tolist()}"
    )
    print(f"AQS calibration={calibration}")
    print("official test 不参与 V2 调参、epoch 选择或 Normal bias 选择。")
    print("=" * 76)

    train_loader = make_loader(
        train_set,
        args.batch_size,
        args.num_workers,
        True,
        device,
        fold_seed + 1,
    )
    validation_loader = make_loader(
        validation_set,
        args.batch_size,
        args.num_workers,
        False,
        device,
        fold_seed + 2,
    )
    model = QLungASTV2(args.model_dir, cache.input_length, pretrained=True).to(device)
    class_counts = torch.bincount(train_set.labels, minlength=4)
    criterion = QLungLoss(class_counts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = scaler_for(amp_enabled)

    history = []
    best_score = -1.0
    best_epoch = 0
    best_bias = 0.0
    best_state = None
    best_logits = None
    best_labels = None
    best_result = None

    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            f"V2 fold {args.fold} epoch {epoch}/{args.epochs}",
        )
        logits, labels = validation_logits(
            model,
            validation_loader,
            device,
            amp_enabled,
        )
        unbiased = calc_icbhi_score(labels.tolist(), logits.argmax(dim=1).tolist())
        normal_bias, calibrated_result = tune_normal_bias(
            logits,
            labels,
            args.bias_min,
            args.bias_max,
            args.bias_steps,
        )
        improved = calibrated_result.score > best_score + 1e-12
        if improved:
            best_score = float(calibrated_result.score)
            best_epoch = epoch
            best_bias = float(normal_bias)
            best_state = clone_cpu_state(model)
            best_logits = logits.clone()
            best_labels = labels.clone()
            best_result = copy.deepcopy(calibrated_result)

        record = {
            "epoch": epoch,
            "training": training,
            "validation_without_bias": unbiased.to_dict(),
            "normal_bias": float(normal_bias),
            "validation_with_bias": calibrated_result.to_dict(),
            "best_epoch": best_epoch,
            "best_score": best_score,
        }
        history.append(record)
        write_json(output / "history.json", history)
        print(
            f"fold={args.fold} epoch={epoch:02d} loss={training['loss']:.4f} "
            f"margin={training['margin']:.4f} | raw={unbiased.score:.4f} "
            f"cal={calibrated_result.score:.4f} bias={normal_bias:+.3f} | "
            f"best={best_score:.4f}@{best_epoch}"
        )

    if best_state is None or best_logits is None or best_labels is None:
        raise RuntimeError("V2 tune 没有得到最佳权重。")
    checkpoint = output / "best_checkpoint.pth"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "method": "QLung-AST-V2-tune",
            "model_state_dict": best_state,
            "fold": int(args.fold),
            "num_folds": int(args.num_folds),
            "split_seed": int(split_seed),
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
            "best_epoch": int(best_epoch),
            "best_normal_bias": float(best_bias),
            "best_validation_metrics": best_result.to_dict(),
            "best_validation_logits": best_logits,
            "best_validation_labels": best_labels,
            "aqs_calibration": calibration,
            "input_length": int(cache.input_length),
        },
        checkpoint,
    )
    summary = {
        "fold": int(args.fold),
        "split_seed": int(split_seed),
        "best_epoch": int(best_epoch),
        "best_normal_bias": float(best_bias),
        "best_validation_metrics": best_result.to_dict(),
        "checkpoint": str(checkpoint.resolve()),
    }
    write_json(output / "summary.json", summary)
    print("=" * 76)
    print(
        f"V2 fold {args.fold} 完成 | best={best_score:.4f}@{best_epoch} | "
        f"bias={best_bias:+.3f}"
    )
    print(f"checkpoint={checkpoint}")
    print("=" * 76)


def run_summarize(args) -> None:
    tune_dir = Path(args.tune_dir).expanduser().resolve()
    epochs = []
    fold_biases = []
    logits_all = []
    labels_all = []
    validation_indices_all = []
    fold_results = []
    split_seed = None

    for fold in range(args.num_folds):
        path = tune_dir / f"fold_{fold}" / "best_checkpoint.pth"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 V2 tune checkpoint: {path}")
        checkpoint = safe_torch_load(path)
        if (
            int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION
            or checkpoint.get("method") != "QLung-AST-V2-tune"
        ):
            raise ValueError(f"不是 V2 tune checkpoint: {path}")
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"V2 tune checkpoint 的 fold 错误: {path}")
        if int(checkpoint.get("num_folds", -1)) != args.num_folds:
            raise ValueError(f"V2 tune checkpoint 的 num_folds 错误: {path}")
        current_seed = int(checkpoint["split_seed"])
        if split_seed is None:
            split_seed = current_seed
        elif current_seed != split_seed:
            raise ValueError("V2 tune 各折使用了不同的 split_seed。")
        epochs.append(int(checkpoint["best_epoch"]))
        fold_biases.append(float(checkpoint["best_normal_bias"]))
        logits_all.append(checkpoint["best_validation_logits"].float())
        labels_all.append(checkpoint["best_validation_labels"].long())
        validation_indices_all.extend(
            int(value) for value in checkpoint["validation_indices"]
        )
        fold_results.append(
            {
                "fold": fold,
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_normal_bias": float(checkpoint["best_normal_bias"]),
                "metrics": checkpoint["best_validation_metrics"],
            }
        )

    if len(validation_indices_all) != len(set(validation_indices_all)):
        raise RuntimeError("V2 OOF validation indices 存在重复。")
    sorted_indices = sorted(validation_indices_all)
    if sorted_indices != list(range(len(sorted_indices))):
        raise RuntimeError("V2 OOF validation indices 没有完整覆盖训练集。")
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    if len(logits) != len(validation_indices_all):
        raise RuntimeError("V2 OOF logits 数量与 validation indices 不一致。")
    raw = calc_icbhi_score(labels.tolist(), logits.argmax(dim=1).tolist())
    global_bias, calibrated = tune_normal_bias(
        logits,
        labels,
        args.bias_min,
        args.bias_max,
        args.bias_steps,
    )
    selected_epoch = max(1, int(round(float(np.median(epochs)))))
    summary = {
        "method": "QLung-AST-V2 OOF tuning",
        "split_seed": int(split_seed),
        "num_folds": int(args.num_folds),
        "best_epochs": epochs,
        "selected_final_epoch": selected_epoch,
        "fold_normal_biases": fold_biases,
        "selected_normal_bias": float(global_bias),
        "oof_without_bias": raw.to_dict(),
        "oof_with_bias": calibrated.to_dict(),
        "folds": fold_results,
        "official_test_used": False,
    }
    destination = tune_dir / "tuning_summary.json"
    write_json(destination, summary)
    print("=" * 76)
    print(f"best epochs={epochs} -> final epoch={selected_epoch}")
    print(f"fold biases={[round(value, 3) for value in fold_biases]}")
    print(
        f"OOF raw Score={raw.score:.4f} | SE={raw.sensitivity:.4f} | "
        f"SP={raw.specificity:.4f}"
    )
    print(
        f"OOF calibrated Score={calibrated.score:.4f} | "
        f"SE={calibrated.sensitivity:.4f} | SP={calibrated.specificity:.4f} | "
        f"normal_bias={global_bias:+.3f}"
    )
    print(f"保存: {destination}")
    print("=" * 76)


def load_tuning_summary(path: Union[str, os.PathLike]) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 V2 tuning summary: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("method") != "QLung-AST-V2 OOF tuning":
        raise ValueError("不是 QLung-AST-V2 tuning summary。")
    if data.get("official_test_used") is not False:
        raise ValueError("V2 tuning summary 必须明确 official_test_used=False。")
    if int(data.get("selected_final_epoch", 0)) <= 0:
        raise ValueError("V2 tuning summary 的 selected_final_epoch 无效。")
    if not math.isfinite(float(data.get("selected_normal_bias", math.nan))):
        raise ValueError("V2 tuning summary 的 selected_normal_bias 无效。")
    return data


def run_train(args) -> None:
    tuning = load_tuning_summary(args.tuning_summary)
    selected_epoch = int(tuning["selected_final_epoch"])
    epochs = selected_epoch if args.epochs is None else int(args.epochs)
    if args.epochs is not None and epochs != selected_epoch:
        print(
            f"警告: 手动 epochs={epochs} 与 OOF 选择的 {selected_epoch} 不同。"
        )
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output = Path(args.output_dir) / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    cache = FeatureCache(args.train_cache, "train")
    if cache.input_length != args.input_length:
        raise ValueError(
            f"V2 train cache input_length={cache.input_length}，"
            f"但命令要求 {args.input_length}。请重新 prepare 或修正参数。"
        )
    all_indices = np.arange(len(cache), dtype=np.int64)
    calibrated_quality, calibration = calibrate_quality(
        cache.raw_quality,
        all_indices,
        args.aqs_target_mean,
        args.aqs_target_std,
    )
    dataset = FeatureView(cache, all_indices, calibrated_quality)
    loader = make_loader(
        dataset,
        args.batch_size,
        args.num_workers,
        True,
        device,
        args.seed + 1,
    )
    model = QLungASTV2(args.model_dir, cache.input_length, pretrained=True).to(device)
    class_counts = torch.bincount(dataset.labels, minlength=4)
    criterion = QLungLoss(class_counts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = scaler_for(amp_enabled)
    history = []

    print("=" * 76)
    print(
        f"V2 final train | seed={args.seed} | epochs={epochs} | "
        f"device={device} | AMP={amp_enabled}"
    )
    print(f"AQS calibration={calibration}")
    print(
        f"OOF-selected normal_bias={float(tuning['selected_normal_bias']):+.3f} "
        "(训练时不使用，仅最终推理使用)"
    )
    print("完整 official train；official test 不参与训练或选择。")
    print("=" * 76)

    for epoch in range(1, epochs + 1):
        training = train_epoch(
            model,
            criterion,
            loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            f"V2 seed {args.seed} epoch {epoch}/{epochs}",
        )
        training["epoch"] = epoch
        history.append(training)
        write_json(output / "history.json", history)
        print(
            f"seed={args.seed} epoch={epoch:02d} loss={training['loss']:.4f} "
            f"ce={training['ce']:.4f} dfam={training['dfam']:.4f} "
            f"margin={training['margin']:.4f} skipped={training['skipped']}"
        )

    checkpoint = output / "final_checkpoint.pth"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "method": "QLung-AST-V2-final",
            "model_state_dict": clone_cpu_state(model),
            "seed": int(args.seed),
            "epoch": int(epochs),
            "input_length": int(cache.input_length),
            "aqs_calibration": calibration,
            "selected_normal_bias": float(tuning["selected_normal_bias"]),
            "tuning_summary": str(Path(args.tuning_summary).expanduser().resolve()),
            "history": history,
            "tta": False,
        },
        checkpoint,
    )
    print(f"V2 final 训练完成: {checkpoint}")


def load_final_model(path: Path, args, device):
    checkpoint = safe_torch_load(path)
    if (
        int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION
        or checkpoint.get("method") != "QLung-AST-V2-final"
    ):
        raise ValueError(f"不是 V2 final checkpoint: {path}")
    input_length = int(checkpoint["input_length"])
    model = QLungASTV2(args.model_dir, input_length, pretrained=False).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"V2 checkpoint 缺少 model_state_dict: {path}")
    model.load_state_dict(
        {str(name).removeprefix("module."): value for name, value in state.items()},
        strict=True,
    )
    return (
        model.eval(),
        int(checkpoint["seed"]),
        input_length,
        float(checkpoint["selected_normal_bias"]),
    )


@torch.inference_mode()
def run_evaluate(args) -> None:
    tuning = load_tuning_summary(args.tuning_summary)
    normal_bias = float(tuning["selected_normal_bias"])
    paths = [Path(value).expanduser().resolve() for value in args.checkpoint]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"V2 final checkpoint 不完整: {paths}")
    if len(paths) != len(set(paths)):
        raise ValueError("V2 evaluate 收到了重复 checkpoint。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    cache = FeatureCache(args.test_cache, "test")
    loader = make_loader(
        cache,
        args.batch_size,
        args.num_workers,
        False,
        device,
        0,
    )
    loaded = [load_final_model(path, args, device) for path in paths]
    if any(
        input_length != cache.input_length
        for _, _, input_length, _ in loaded
    ):
        raise ValueError("V2 checkpoint 与 test cache 的 input_length 不一致。")
    if any(
        not math.isclose(checkpoint_bias, normal_bias, abs_tol=1e-9)
        for _, _, _, checkpoint_bias in loaded
    ):
        raise ValueError("V2 checkpoint 与 tuning summary 的 Normal bias 不一致。")
    models = [item[0] for item in loaded]
    seeds = [item[1] for item in loaded]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"V2 evaluate 收到了重复 seed: {seeds}")
    labels_all = []
    ensemble_predictions = []
    individual_predictions = [[] for _ in models]

    for features, labels, _ in tqdm(loader, desc="V2 official test (no TTA)"):
        features = features.to(device, non_blocking=True)
        if device.type == "cpu":
            features = features.float()
        probability_sum = None
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            for index, model in enumerate(models):
                logits, _ = model(features)
                logits = logits.float()
                logits[:, 0] += normal_bias
                probability = logits.softmax(dim=1)
                individual_predictions[index].extend(
                    probability.argmax(dim=1).cpu().tolist()
                )
                probability_sum = (
                    probability
                    if probability_sum is None
                    else probability_sum + probability
                )
        ensemble_predictions.extend(probability_sum.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.tolist())

    individuals = []
    for seed, prediction in zip(seeds, individual_predictions):
        result = calc_icbhi_score(labels_all, prediction)
        individuals.append({"seed": seed, "official_icbhi": result.to_dict()})
        print(
            f"V2 seed={seed} score={result.score:.4f} "
            f"SE={result.sensitivity:.4f} SP={result.specificity:.4f}"
        )
    result = calc_icbhi_score(labels_all, ensemble_predictions)
    legacy = calc_legacy_macro_ovr_score(labels_all, ensemble_predictions)
    report = classification_report(
        labels_all,
        ensemble_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print("=" * 76)
    print(f"V2 models={len(models)} seeds={seeds} TTA=False")
    print(f"OOF-selected normal_bias={normal_bias:+.3f}")
    print(
        f"Official ICBHI Score={result.score:.4f} | "
        f"SE={result.sensitivity:.4f} | SP={result.specificity:.4f} | "
        f"Legacy={legacy['score']:.4f}"
    )
    print(report)
    print(result.confusion_matrix)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_matrix(result.confusion_matrix, output / "confusion_matrix.png")
    write_json(
        output / "metrics.json",
        {
            "method": "QLung-AST-V2 probability ensemble",
            "checkpoints": [str(path) for path in paths],
            "seeds": seeds,
            "input_length": int(cache.input_length),
            "normal_bias": normal_bias,
            "normal_bias_selected_on": "patient-disjoint OOF train validation",
            "tta": False,
            "official_icbhi": result.to_dict(),
            "legacy_macro_ovr": legacy,
            "individual_models": individuals,
        },
    )
    print(f"V2 结果保存到: {output}")


def run_check(args) -> None:
    model = QLungASTV2(args.model_dir, args.input_length, pretrained=True)
    frequency_patches, time_patches = patch_grid(model.ast.config, args.input_length)
    tokens = int(model.ast.embeddings.position_embeddings.size(1))
    raw = torch.linspace(0.2, 0.8, 100)
    calibrated, statistics = calibrate_quality(
        raw,
        np.arange(80),
        args.aqs_target_mean,
        args.aqs_target_std,
    )
    assert tokens == 2 + frequency_patches * time_patches
    assert calibrated.min() >= 0 and calibrated.max() <= 1
    print("=" * 76)
    print("V2 self-check 通过")
    print(
        f"AST grid={frequency_patches}x{time_patches} | "
        f"tokens={tokens} | input_length={args.input_length}"
    )
    print(f"AQS check={statistics}")
    print("=" * 76)


def common_model_arguments(parser) -> None:
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--input-length", type=int, default=INPUT_LENGTH)
    parser.add_argument("--aqs-target-mean", type=float, default=AQS_TARGET_MEAN)
    parser.add_argument("--aqs-target-std", type=float, default=AQS_TARGET_STD)


def common_training_arguments(parser) -> None:
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--no-amp", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(description="Leakage-safe QLung-AST V2")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Check AST resize and AQS calibration")
    common_model_arguments(check)

    prepare = commands.add_parser("prepare", help="Prepare 798-frame V2 caches")
    common_model_arguments(prepare)
    prepare.add_argument("--audio-dir", default=AUDIO_DIR)
    prepare.add_argument("--csv-path", default=CSV_PATH)
    prepare.add_argument("--cache-dir", default=CACHE_DIR)
    # Deliberately prepare train only by default.  The test cache is generated
    # after all OOF choices and final training are finished.
    prepare.add_argument("--splits", nargs="+", default=["train"])
    prepare.add_argument("--feature-batch-size", type=int, default=16)
    prepare.add_argument("--force", action="store_true")

    tune = commands.add_parser("tune", help="Tune one patient-disjoint fold")
    common_model_arguments(tune)
    common_training_arguments(tune)
    tune.add_argument("--train-cache", default=TRAIN_CACHE)
    tune.add_argument("--output-dir", default=TUNE_DIR)
    tune.add_argument("--seed", type=int, default=24923)
    tune.add_argument("--num-folds", type=int, default=5)
    tune.add_argument("--fold", type=int, required=True)
    tune.add_argument("--split-candidates", type=int, default=500)
    tune.add_argument("--epochs", type=int, default=50)
    tune.add_argument("--bias-min", type=float, default=-4.0)
    tune.add_argument("--bias-max", type=float, default=8.0)
    tune.add_argument("--bias-steps", type=int, default=241)

    summarize = commands.add_parser("summarize", help="Aggregate five OOF folds")
    summarize.add_argument("--tune-dir", default=TUNE_DIR)
    summarize.add_argument("--num-folds", type=int, default=5)
    summarize.add_argument("--bias-min", type=float, default=-4.0)
    summarize.add_argument("--bias-max", type=float, default=8.0)
    summarize.add_argument("--bias-steps", type=int, default=241)

    train = commands.add_parser("train", help="Train one full-data final seed")
    common_model_arguments(train)
    common_training_arguments(train)
    train.add_argument("--train-cache", default=TRAIN_CACHE)
    train.add_argument("--tuning-summary", default=TUNING_SUMMARY)
    train.add_argument("--output-dir", default=FINAL_DIR)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int)

    evaluate = commands.add_parser("evaluate", help="Final no-TTA evaluation")
    evaluate.add_argument("--checkpoint", nargs="+", required=True)
    evaluate.add_argument("--model-dir", default=MODEL_DIR)
    evaluate.add_argument("--test-cache", default=TEST_CACHE)
    evaluate.add_argument("--tuning-summary", default=TUNING_SUMMARY)
    evaluate.add_argument("--output-dir", default="evaluation/qlung_ast_v2_final")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--num-workers", type=int, default=2)
    evaluate.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if hasattr(args, "input_length") and args.input_length <= 16:
        raise ValueError("input-length 必须大于 AST patch size。")
    if hasattr(args, "aqs_target_mean"):
        if not 0 < args.aqs_target_mean < 1:
            raise ValueError("aqs-target-mean 必须位于 (0, 1)。")
        if not 0 < args.aqs_target_std < 0.5:
            raise ValueError("aqs-target-std 必须位于 (0, 0.5)。")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch-size 必须为正整数。")
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("num-workers 不能为负数。")
    if hasattr(args, "feature_batch_size") and args.feature_batch_size <= 0:
        raise ValueError("feature-batch-size 必须为正整数。")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise ValueError("learning-rate 必须为正数。")
    if hasattr(args, "num_folds") and args.num_folds < 2:
        raise ValueError("num-folds 必须至少为 2。")
    if hasattr(args, "split_candidates") and args.split_candidates <= 0:
        raise ValueError("split-candidates 必须为正整数。")
    if hasattr(args, "epochs") and args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs 必须为正整数。")
    if hasattr(args, "seed") and args.seed < 0:
        raise ValueError("seed 不能为负数。")
    if hasattr(args, "bias_steps"):
        if args.bias_steps < 2 or args.bias_max <= args.bias_min:
            raise ValueError("Normal bias 搜索范围无效。")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.command == "check":
        run_check(args)
    elif args.command == "prepare":
        if not set(args.splits).issubset({"train", "test"}):
            raise ValueError("splits 只能包含 train/test。")
        run_prepare(args)
    elif args.command == "tune":
        run_tune(args)
    elif args.command == "summarize":
        run_summarize(args)
    elif args.command == "train":
        run_train(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()