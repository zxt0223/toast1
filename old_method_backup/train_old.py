from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from tqdm import tqdm

from dataset import CLASS_NAMES, ICBHIDataset, ICBHIView
from model import HierarchicalRespiratoryNet


@dataclass
class Config:
    audio_dir: str = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
    csv_path: str = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
    output_dir: str = "runs/icbhi_hierarchical_v2"

    seed: int = 24923
    split_seed: int = 24923

    epochs: int = 90
    batch_size: int = 32
    num_workers: int = 8
    patience: int = 16

    sample_rate: int = 16000
    duration: float = 8.0

    # 双分辨率前端。
    transient_n_fft: int = 512
    transient_hop: int = 160
    context_n_fft: int = 1024
    context_hop: int = 320
    n_mels: int = 96
    f_min: float = 50.0
    f_max: float = 4000.0
    top_db: float = 80.0
    target_frames: int = 384

    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup_epochs: int = 5
    weight_decay: float = 1e-2
    grad_clip: float = 2.0

    val_ratio: float = 0.20

    binary_loss_weight: float = 1.0
    subtype_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.20
    label_smoothing: float = 0.05

    use_ema: bool = True
    ema_decay: float = 0.997
    use_amp: bool = True

    freq_mask_param: int = 8
    time_mask_param: int = 24
    noise_probability: float = 0.25
    min_noise_snr_db: float = 20.0
    max_noise_snr_db: float = 32.0
    max_time_shift_seconds: float = 0.35

    threshold_min: float = 0.25
    threshold_max: float = 0.75
    threshold_step: float = 0.01

    strict_official_test_patient_disjoint: bool = False
    skip_official_test: bool = False


class DualResolutionFrontend(nn.Module):
    """瞬态视图 + 长时上下文视图，输出两个输入通道。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.target_mels = cfg.n_mels
        self.target_frames = cfg.target_frames

        self.transient_mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.transient_n_fft,
            win_length=cfg.transient_n_fft,
            hop_length=cfg.transient_hop,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.context_mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.context_n_fft,
            win_length=cfg.context_n_fft,
            hop_length=cfg.context_hop,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=cfg.freq_mask_param,
            iid_masks=True,
        )
        self.time_mask = T.TimeMasking(
            time_mask_param=cfg.time_mask_param,
            iid_masks=True,
        )

    @staticmethod
    def _normalize(view: torch.Tensor) -> torch.Tensor:
        mean = view.mean(dim=(-2, -1), keepdim=True)
        std = view.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (view - mean) / std

    def _finish(self, linear: torch.Tensor) -> torch.Tensor:
        db = self.to_db(linear.clamp_min(1e-10))
        db = self._normalize(db)
        return F.interpolate(
            db,
            size=(self.target_mels, self.target_frames),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, waveforms: torch.Tensor, augment: bool = False) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            waveforms = waveforms.float()
            transient = self._finish(self.transient_mel(waveforms))
            context = self._finish(self.context_mel(waveforms))
            features = torch.cat([transient, context], dim=1)
            if augment:
                features = self.freq_mask(features)
                features = self.time_mask(features)
            return features


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        warmup_decay = (1.0 + self.updates) / (10.0 + self.updates)
        decay = min(self.decay, warmup_decay)

        source_parameters = dict(model.named_parameters())
        for name, ema_parameter in self.module.named_parameters():
            source = source_parameters[name].detach()
            ema_parameter.mul_(decay).add_(source, alpha=1.0 - decay)

        source_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.module.named_buffers():
            ema_buffer.copy_(source_buffers[name].detach())


def parse_args():
    parser = argparse.ArgumentParser(
        description="HATS-Net：ICBHI 层次化异常检测与异常类型分类"
    )
    parser.add_argument("--seed", type=int, default=24923)
    parser.add_argument("--split-seed", type=int, default=24923)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--strict-patient-disjoint", action="store_true")
    return parser.parse_args()


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def unpack_batch(batch):
    if len(batch) == 4:
        waveforms, labels, metadata, patient_ids = batch
        return waveforms, labels, metadata, patient_ids
    if len(batch) == 3:
        waveforms, labels, metadata = batch
        return waveforms, labels, metadata, None
    raise ValueError(f"Dataset batch 长度应为 3 或 4，实际为 {len(batch)}")


def calc_icbhi_score(y_true: Sequence[int], y_pred: Sequence[int]):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    normal_total = int(cm[0].sum())
    abnormal_total = int(cm[1:].sum())

    sp = float(cm[0, 0] / normal_total) if normal_total else 0.0
    abnormal_correct = int(cm[1, 1] + cm[2, 2] + cm[3, 3])
    se = float(abnormal_correct / abnormal_total) if abnormal_total else 0.0
    score = 0.5 * (se + sp)

    recalls = []
    for index in range(4):
        total = int(cm[index].sum())
        recalls.append(float(cm[index, index] / total) if total else 0.0)
    return score, se, sp, cm, recalls


def print_metrics(prefix: str, metrics, threshold: float) -> None:
    score, se, sp, cm, recalls = metrics
    recall_text = ", ".join(
        f"{name}={value:.4f}" for name, value in zip(CLASS_NAMES, recalls)
    )
    print(
        f"{prefix} Score={score:.4f} | SE={se:.4f} | SP={sp:.4f} | "
        f"Threshold={threshold:.2f} | Recall[{recall_text}]"
    )
    print(cm)


def metrics_to_dict(metrics, threshold: float) -> Dict:
    score, se, sp, cm, recalls = metrics
    return {
        "score": float(score),
        "sensitivity": float(se),
        "specificity": float(sp),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, recalls)
        },
    }


def add_waveform_augmentation(waveforms: torch.Tensor, cfg: Config) -> torch.Tensor:
    batch_size = waveforms.size(0)

    gain_db = torch.empty(
        batch_size, 1, 1, device=waveforms.device
    ).uniform_(-3.0, 3.0)
    output = waveforms * torch.pow(10.0, gain_db / 20.0)

    if torch.rand((), device=waveforms.device).item() < cfg.noise_probability:
        snr_db = torch.empty(
            batch_size, 1, 1, device=waveforms.device
        ).uniform_(cfg.min_noise_snr_db, cfg.max_noise_snr_db)
        signal_rms = output.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        noise_rms = signal_rms / torch.pow(10.0, snr_db / 20.0)
        output = output + torch.randn_like(output) * noise_rms

    max_shift = int(cfg.max_time_shift_seconds * cfg.sample_rate)
    if max_shift > 0:
        shift = int(
            torch.randint(
                low=-max_shift,
                high=max_shift + 1,
                size=(1,),
                device=waveforms.device,
            ).item()
        )
        if shift != 0:
            shifted = torch.zeros_like(output)
            if shift > 0:
                shifted[..., shift:] = output[..., :-shift]
            else:
                shifted[..., :shift] = output[..., -shift:]
            output = shifted

    return output.clamp(-1.0, 1.0)


def balanced_binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    normal_mask = targets < 0.5
    abnormal_mask = ~normal_mask

    components = []
    if normal_mask.any():
        components.append(losses[normal_mask].mean())
    if abnormal_mask.any():
        components.append(losses[abnormal_mask].mean())
    return torch.stack(components).mean()


def make_patient_split(
    dataset: ICBHIDataset,
    test_dataset: ICBHIDataset,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(dataset))
    candidate_indices = all_indices

    train_patients = set(dataset.patient_ids)
    test_patients = set(test_dataset.patient_ids)
    overlap = sorted(train_patients.intersection(test_patients))
    if overlap:
        print(f"⚠️ metadata 中 train/test 患者重叠: {overlap}")

    if cfg.strict_official_test_patient_disjoint and overlap:
        overlap_set = set(overlap)
        candidate_indices = np.array(
            [
                index
                for index in all_indices
                if dataset.patient_ids[index] not in overlap_set
            ],
            dtype=np.int64,
        )
        print(
            "🔒 严格患者独立模式：从开发集移除 "
            f"{len(all_indices) - len(candidate_indices)} 个周期"
        )

    labels = dataset.labels.numpy()[candidate_indices]
    groups = np.asarray(dataset.patient_ids, dtype=object)[candidate_indices]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.val_ratio,
        random_state=cfg.split_seed,
    )
    train_pos, val_pos = next(
        splitter.split(candidate_indices, labels, groups)
    )
    train_indices = candidate_indices[train_pos]
    val_indices = candidate_indices[val_pos]

    train_groups = set(np.asarray(dataset.patient_ids)[train_indices])
    val_groups = set(np.asarray(dataset.patient_ids)[val_indices])
    if train_groups.intersection(val_groups):
        raise RuntimeError("内部 train/validation 患者发生重叠。")

    print(f"🔒 固定患者划分种子: split_seed={cfg.split_seed}")
    print(
        "📊 内部 train 类别数: "
        f"{torch.bincount(dataset.labels[train_indices], minlength=4).tolist()}"
    )
    print(
        "📊 内部 val 类别数: "
        f"{torch.bincount(dataset.labels[val_indices], minlength=4).tolist()}"
    )
    return train_indices, val_indices


def make_loaders(cfg: Config):
    base_train = ICBHIDataset(
        data_path=cfg.audio_dir,
        split="train",
        metadatafile=cfg.csv_path,
        duration=cfg.duration,
        samplerate=cfg.sample_rate,
        training=False,
        cache=True,
    )
    test_dataset = ICBHIDataset(
        data_path=cfg.audio_dir,
        split="test",
        metadatafile=cfg.csv_path,
        duration=cfg.duration,
        samplerate=cfg.sample_rate,
        training=False,
        cache=True,
    )

    train_indices, val_indices = make_patient_split(base_train, test_dataset, cfg)
    train_dataset = ICBHIView(base_train, train_indices, training=True)
    val_dataset = ICBHIView(base_train, val_indices, training=False)

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed + 2),
        **loader_kwargs,
    )

    print(
        "✅ 患者级内部划分: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"official_test={len(test_dataset)}"
    )
    print(
        "👥 患者数: "
        f"train={len(set(train_dataset.patient_ids))}, "
        f"val={len(set(val_dataset.patient_ids))}, "
        f"test={len(set(test_dataset.patient_ids))}"
    )
    return train_loader, val_loader, test_loader


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.inference_mode()
def collect_outputs(
    model: nn.Module,
    frontend: DualResolutionFrontend,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    frontend.eval()

    labels_all = []
    abnormal_probabilities = []
    subtype_probabilities = []
    auxiliary_probabilities = []

    for batch in loader:
        waveforms, labels, _, _ = unpack_batch(batch)
        waveforms = waveforms.to(device, non_blocking=True)
        features = frontend(waveforms, augment=False)
        abnormal_logit, subtype_logits, auxiliary_logits = model(features)

        labels_all.append(labels.cpu())
        abnormal_probabilities.append(torch.sigmoid(abnormal_logit.float()).cpu())
        subtype_probabilities.append(torch.softmax(subtype_logits.float(), dim=1).cpu())
        auxiliary_probabilities.append(torch.softmax(auxiliary_logits.float(), dim=1).cpu())

    return (
        torch.cat(labels_all).numpy(),
        torch.cat(abnormal_probabilities).numpy(),
        torch.cat(subtype_probabilities).numpy(),
        torch.cat(auxiliary_probabilities).numpy(),
    )


def predictions_from_threshold(
    abnormal_probabilities: np.ndarray,
    subtype_probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    subtype_predictions = subtype_probabilities.argmax(axis=1) + 1
    return np.where(abnormal_probabilities >= threshold, subtype_predictions, 0)


def find_best_threshold(
    labels: np.ndarray,
    abnormal_probabilities: np.ndarray,
    subtype_probabilities: np.ndarray,
    cfg: Config,
):
    thresholds = np.arange(
        cfg.threshold_min,
        cfg.threshold_max + cfg.threshold_step * 0.5,
        cfg.threshold_step,
    )

    best_threshold = 0.5
    best_metrics = None
    best_score = -1.0

    for threshold in thresholds:
        predictions = predictions_from_threshold(
            abnormal_probabilities,
            subtype_probabilities,
            float(threshold),
        )
        metrics = calc_icbhi_score(labels, predictions)
        score = float(metrics[0])

        better = score > best_score + 1e-12
        tie = abs(score - best_score) <= 1e-12
        closer_to_half = abs(threshold - 0.5) < abs(best_threshold - 0.5)
        if better or (tie and closer_to_half):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("阈值搜索失败。")
    return best_threshold, best_metrics


@torch.inference_mode()
def evaluate_with_calibration(
    model: nn.Module,
    frontend: DualResolutionFrontend,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
):
    labels, abnormal_probs, subtype_probs, _ = collect_outputs(
        model, frontend, loader, device
    )
    threshold, metrics = find_best_threshold(
        labels, abnormal_probs, subtype_probs, cfg
    )
    return metrics, threshold


def clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: Optional[ModelEMA],
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_score: float,
    cfg: Config,
    selected_source: str,
    selected_state_dict: Dict[str, torch.Tensor],
    selected_threshold: float,
    selected_metrics,
    skipped_batches_total: int,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.module.state_dict() if ema is not None else None,
            "selected_source": selected_source,
            "selected_state_dict": selected_state_dict,
            "selected_threshold": float(selected_threshold),
            "selected_metrics": metrics_to_dict(selected_metrics, selected_threshold),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": float(best_score),
            "config": asdict(cfg),
            "class_names": CLASS_NAMES,
            "model_type": "HATS-Net-hierarchical-v2",
            "skipped_batches_total": int(skipped_batches_total),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = Config(seed=args.seed, split_seed=args.split_seed)

    if args.audio_dir is not None:
        cfg.audio_dir = args.audio_dir
    if args.csv_path is not None:
        cfg.csv_path = args.csv_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.disable_amp:
        cfg.use_amp = False
    if args.skip_official_test:
        cfg.skip_official_test = True
    if args.strict_patient_disjoint:
        cfg.strict_official_test_patient_disjoint = True

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")

    output_dir = Path(cfg.output_dir) / f"split_{cfg.split_seed}_seed_{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        f"HATS-Net | TrainSeed={cfg.seed} | SplitSeed={cfg.split_seed} | "
        f"Device={device} | AMP={amp_enabled} | Output={output_dir}"
    )
    print(
        f"DualMel=(512/{cfg.transient_hop}, 1024/{cfg.context_hop}) | "
        f"Fmax={cfg.f_max:.0f}Hz | NaturalShuffle | HierarchicalLoss | "
        f"ValThresholdSearch={cfg.threshold_min:.2f}-{cfg.threshold_max:.2f}"
    )
    print("=" * 88)

    train_loader, val_loader, test_loader = make_loaders(cfg)

    model = HierarchicalRespiratoryNet(input_channels=2).to(device)
    frontend = DualResolutionFrontend(cfg).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.use_ema else None

    subtype_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    auxiliary_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    best_score = -1.0
    no_improvement = 0
    skipped_batches_total = 0
    best_path = output_dir / "best_checkpoint.pth"
    last_path = output_dir / "last_checkpoint.pth"

    for epoch in range(cfg.epochs):
        model.train()
        frontend.train()

        running_total = 0.0
        running_binary = 0.0
        running_subtype = 0.0
        running_auxiliary = 0.0
        seen = 0
        skipped_this_epoch = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:02d}/{cfg.epochs} [Train]",
            leave=False,
        )

        for batch in progress:
            waveforms, labels, _, _ = unpack_batch(batch)
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            waveforms = add_waveform_augmentation(waveforms, cfg)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=False):
                features = frontend(waveforms.float(), augment=True)

            if not torch.isfinite(features).all():
                skipped_this_epoch += 1
                skipped_batches_total += 1
                continue

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                abnormal_logit, subtype_logits, auxiliary_logits = model(features)
                abnormal_targets = (labels != 0).float()
                binary_loss = balanced_binary_loss(abnormal_logit, abnormal_targets)

                abnormal_mask = labels != 0
                if abnormal_mask.any():
                    subtype_targets = labels[abnormal_mask] - 1
                    subtype_loss = subtype_criterion(
                        subtype_logits[abnormal_mask], subtype_targets
                    )
                else:
                    subtype_loss = abnormal_logit.sum() * 0.0

                auxiliary_loss = auxiliary_criterion(auxiliary_logits, labels)
                loss = (
                    cfg.binary_loss_weight * binary_loss
                    + cfg.subtype_loss_weight * subtype_loss
                    + cfg.auxiliary_loss_weight * auxiliary_loss
                )

            if not torch.isfinite(loss):
                skipped_this_epoch += 1
                skipped_batches_total += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_this_epoch += 1
                skipped_batches_total += 1
                scaler.update(max(float(scaler.get_scale()) / 2.0, 1.0))
                continue

            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < scale_before:
                skipped_this_epoch += 1
                skipped_batches_total += 1
                continue

            scheduler.step()
            if ema is not None:
                ema.update(model)

            batch_size = labels.size(0)
            seen += batch_size
            running_total += float(loss.detach()) * batch_size
            running_binary += float(binary_loss.detach()) * batch_size
            running_subtype += float(subtype_loss.detach()) * batch_size
            running_auxiliary += float(auxiliary_loss.detach()) * batch_size

            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                skip=skipped_this_epoch,
            )

        if seen == 0:
            raise RuntimeError("本轮没有有效参数更新。请使用 --disable-amp 或降低学习率。")

        raw_metrics, raw_threshold = evaluate_with_calibration(
            model, frontend, val_loader, device, cfg
        )
        if ema is not None:
            ema_metrics, ema_threshold = evaluate_with_calibration(
                ema.module, frontend, val_loader, device, cfg
            )
        else:
            ema_metrics, ema_threshold = None, None

        print(
            f"Epoch {epoch + 1:02d} | Train={running_total / seen:.4f} "
            f"(bin={running_binary / seen:.4f}, "
            f"sub={running_subtype / seen:.4f}, "
            f"aux={running_auxiliary / seen:.4f}) | Skipped={skipped_this_epoch}"
        )
        print_metrics("  Validation RAW |", raw_metrics, raw_threshold)
        if ema_metrics is not None:
            print_metrics("  Validation EMA |", ema_metrics, ema_threshold)

        if ema_metrics is not None and ema_metrics[0] > raw_metrics[0] + 1e-12:
            selected_source = "EMA"
            selected_module = ema.module
            selected_metrics = ema_metrics
            selected_threshold = float(ema_threshold)
        else:
            selected_source = "RAW"
            selected_module = model
            selected_metrics = raw_metrics
            selected_threshold = float(raw_threshold)

        selected_score = float(selected_metrics[0])
        selected_state = clone_state_dict(selected_module)
        print(
            f"  ✅ 本轮选择: {selected_source} "
            f"(Score={selected_score:.4f}, Threshold={selected_threshold:.2f})"
        )

        save_checkpoint(
            last_path,
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            best_score,
            cfg,
            selected_source,
            selected_state,
            selected_threshold,
            selected_metrics,
            skipped_batches_total,
        )

        if selected_score > best_score + 1e-6:
            best_score = selected_score
            no_improvement = 0
            save_checkpoint(
                best_path,
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_score,
                cfg,
                selected_source,
                selected_state,
                selected_threshold,
                selected_metrics,
                skipped_batches_total,
            )
            print(f"  🌟 保存最佳权重: {best_path}")
        else:
            no_improvement += 1
            print(f"  ⚠️ 未提升: {no_improvement}/{cfg.patience}")

        if no_improvement >= cfg.patience:
            print(f"🛑 触发早停：连续 {cfg.patience} 轮未提升。")
            break

    if cfg.skip_official_test:
        print("\n🧪 训练/冒烟测试完成，按要求跳过 official test。")
        return

    if not best_path.is_file():
        raise FileNotFoundError("未生成 best_checkpoint.pth。")

    print("\n🔍 配置冻结后，仅评估一次 official test...")
    checkpoint = safe_torch_load(best_path, device)
    model.load_state_dict(checkpoint["selected_state_dict"], strict=True)
    model.to(device).eval()
    selected_threshold = float(checkpoint["selected_threshold"])
    selected_source = checkpoint.get("selected_source", "UNKNOWN")
    print(
        f"📌 使用内部验证选中的权重来源: {selected_source} | "
        f"Threshold={selected_threshold:.2f}"
    )

    labels, abnormal_probs, subtype_probs, _ = collect_outputs(
        model, frontend, test_loader, device
    )
    predictions = predictions_from_threshold(
        abnormal_probs, subtype_probs, selected_threshold
    )
    test_metrics = calc_icbhi_score(labels, predictions)
    print_metrics("Official Test |", test_metrics, selected_threshold)

    result = {
        "seed": cfg.seed,
        "split_seed": cfg.split_seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_weight_source": selected_source,
        "best_internal_val_score": float(checkpoint["best_score"]),
        "selected_threshold": selected_threshold,
        "official_test": metrics_to_dict(test_metrics, selected_threshold),
        "skipped_batches_total": int(skipped_batches_total),
        "config": asdict(cfg),
    }
    result_path = output_dir / "final_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ 结果已保存: {result_path}")


if __name__ == "__main__":
    main()