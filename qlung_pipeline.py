"""Compact AST + QLung pipeline for this repository's ICBHI dataset.

Commands: prepare, train, evaluate. Official-test inference has no TTA.
QLung constants follow arXiv:2606.11915. Because the authors' repository is
currently unavailable, the paper-unspecified H_norm/R_norm implementation is
made explicit in quality_score().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

from dataset import CLASS_NAMES, ICBHIDataset, safe_torch_load
from metrics import calc_icbhi_score, calc_legacy_macro_ovr_score


AUDIO_DIR = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
CSV_PATH = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
MODEL_DIR = "pretrained/ast_audioset"
CACHE_DIR = "precomputed/qlung_ast"
TRAIN_CACHE = f"{CACHE_DIR}/ast_train.pth"
TEST_CACHE = f"{CACHE_DIR}/ast_test.pth"

# Paper settings.
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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.inference_mode()
def quality_score(waveforms: torch.Tensor) -> torch.Tensor:
    """AQS=clip(1-.7*Hnorm+.3*Rnorm,0,1), using 25-ms/10-ms frames."""
    x = waveforms.squeeze(1).float()
    window = torch.hann_window(400, dtype=x.dtype, device=x.device)
    stft = torch.stft(
        x,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        return_complex=True,
    )
    power = stft.abs().square().clamp_min(1e-12)
    probability = power / power.sum(1, keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(1) / math.log(power.size(1))
    h_norm = entropy.mean(1).clamp(0, 1)
    r_norm = x.square().mean(1).sqrt().clamp(0, 1)
    return (1.0 - 0.7 * h_norm + 0.3 * r_norm).clamp(0, 1)


def prepare_split(args, split: str, extractor: ASTFeatureExtractor) -> None:
    destination = Path(args.cache_dir) / f"ast_{split}.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        print(f"已存在，跳过: {destination}")
        return

    source = ICBHIDataset(
        data_path=args.audio_dir,
        split=split,
        metadatafile=args.csv_path,
        duration=8.0,
        samplerate=16000,
        cache=True,
    )
    shape = (len(source), int(extractor.max_length), int(extractor.num_mel_bins))
    features = torch.empty(shape, dtype=torch.float16)
    quality = torch.empty(len(source), dtype=torch.float32)
    for start in tqdm(range(0, len(source), args.feature_batch_size), desc=split):
        stop = min(start + args.feature_batch_size, len(source))
        waveforms = source.data[start:stop].float()
        raw = [item.squeeze(0).numpy() for item in waveforms]
        values = extractor(
            raw, sampling_rate=16000, return_tensors="pt"
        ).input_values
        expected = (stop - start,) + shape[1:]
        if tuple(values.shape) != expected:
            raise RuntimeError(f"AST 输入形状 {tuple(values.shape)} != {expected}")
        features[start:stop] = values.half()
        quality[start:stop] = quality_score(waveforms)

    torch.save(
        {
            "version": 1,
            "split": split,
            "features": features,
            "labels": source.labels.long(),
            "quality": quality,
            "patient_ids": source.patient_ids,
            "sample_ids": source.sample_ids,
        },
        destination,
    )
    print(
        f"保存: {destination} | shape={shape} | "
        f"AQS={quality.min():.4f}/{quality.mean():.4f}/{quality.max():.4f}"
    )


def run_prepare(args) -> None:
    extractor = ASTFeatureExtractor.from_pretrained(
        args.model_dir, local_files_only=True
    )
    print(
        f"AST extractor: length={extractor.max_length}, "
        f"mel={extractor.num_mel_bins}, mean={extractor.mean}, std={extractor.std}"
    )
    for split in args.splits:
        prepare_split(args, split, extractor)


class CachedFeatures(Dataset):
    def __init__(self, filename: str, split: str) -> None:
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {path}，请先运行 prepare。")
        data = safe_torch_load(path)
        if data.get("version") != 1 or data.get("split") != split:
            raise ValueError(f"缓存版本或 split 错误: {path}")
        self.x = data["features"].contiguous()
        self.y = data["labels"].long().contiguous()
        self.q = data["quality"].float().contiguous()
        if self.x.shape[1:] != (1024, 128) or len(self.x) != len(self.y):
            raise ValueError(f"缓存形状错误: {self.x.shape}")
        print(
            f"加载 {split}: {len(self.y)} samples | "
            f"classes={torch.bincount(self.y, minlength=4).tolist()}"
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.q[index]


def spec_augment(x: torch.Tensor, freq_width: int = 48, time_width: int = 192):
    x = x.clone()
    _, time_steps, freq_bins = x.shape
    for item in x:
        width = int(torch.randint(0, min(freq_width, freq_bins) + 1, (1,)).item())
        if width:
            start = int(torch.randint(0, freq_bins - width + 1, (1,)).item())
            item[:, start : start + width] = 0
        width = int(torch.randint(0, min(time_width, time_steps) + 1, (1,)).item())
        if width:
            start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
            item[start : start + width] = 0
    return x


class AngularClassifier(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, dimension))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor):
        features = F.normalize(features, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(features, weights).clamp(-1, 1)
        return ANGULAR_SCALE * cosine, cosine


class QLungAST(nn.Module):
    def __init__(self, model_dir: str, pretrained: bool) -> None:
        super().__init__()
        if pretrained:
            self.ast = ASTModel.from_pretrained(
                model_dir, local_files_only=True, use_safetensors=True
            )
        else:
            config = ASTConfig.from_pretrained(model_dir, local_files_only=True)
            self.ast = ASTModel(config)
        self.head = AngularClassifier(int(self.ast.config.hidden_size))

    def forward(self, x: torch.Tensor):
        features = self.ast(input_values=x, return_dict=True).pooler_output
        logits, cosine = self.head(features)
        return logits, cosine


class QLungLoss(nn.Module):
    def __init__(self, counts: torch.Tensor) -> None:
        super().__init__()
        frequency = counts.float() / counts.sum()
        scale = TARGET_MARGIN / math.log(4)
        self.register_buffer("class_margin", scale * (-frequency.log()))

    def forward(self, logits, cosine, labels, quality):
        ce = F.cross_entropy(logits, labels)
        mq = QUALITY_SCALE * quality.clamp(0, 1)
        mc = self.class_margin[labels]
        margin = GAMMA * mq + (1 - GAMMA) * mc

        cosine = cosine.float().clamp(-1 + 1e-6, 1 - 1e-6)
        target = cosine.gather(1, labels[:, None]).squeeze(1)
        sine = (1 - target.square()).clamp_min(1e-7).sqrt()
        target_with_margin = target * margin.cos() - sine * margin.sin()
        dfam_logits = DFAM_SCALE * cosine
        dfam_logits = dfam_logits.scatter(
            1, labels[:, None], (DFAM_SCALE * target_with_margin)[:, None]
        )
        dfam = F.cross_entropy(dfam_logits, labels)
        return ce + LAMBDA_DFAM * dfam, ce, dfam, margin.mean()


def cpu_state(model: nn.Module):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def run_train(args) -> None:
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    dataset = CachedFeatures(args.train_cache, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = QLungAST(args.model_dir, pretrained=True).to(device)
    counts = torch.bincount(dataset.y, minlength=4)
    criterion = QLungLoss(counts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scaler = scaler_for(amp)
    output = Path(args.output_dir) / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"QLung | seed={args.seed} | device={device} | AMP={amp}")
    print("完整 official train；official test 不参与训练或选权重；TTA=False")
    print(f"class margin={criterion.class_margin.cpu().numpy().round(4).tolist()}")
    print("=" * 76)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        seen = skipped = 0
        bar = tqdm(loader, desc=f"seed {args.seed} epoch {epoch}/{args.epochs}")
        for features, labels, quality in bar:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            if device.type == "cpu":
                features = features.float()
            features = spec_augment(features)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits, cosine = model(features)
                loss, ce, dfam, margin = criterion(logits, cosine, labels, quality)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = len(labels)
            sums += np.array(
                [float(loss.detach()), float(ce.detach()), float(dfam.detach()),
                 float(margin.detach())]
            ) * count
            seen += count
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}", skipped=skipped)
        if seen == 0:
            raise RuntimeError("一个 epoch 内没有任何有效更新。")
        means = sums / seen
        record = {
            "epoch": epoch,
            "loss": means[0],
            "ce": means[1],
            "dfam": means[2],
            "margin": means[3],
            "skipped": skipped,
        }
        history.append(record)
        write_json(output / "history.json", history)
        print(
            f"epoch={epoch:02d} loss={means[0]:.4f} ce={means[1]:.4f} "
            f"dfam={means[2]:.4f} margin={means[3]:.4f} skipped={skipped}"
        )

    checkpoint = output / "final_checkpoint.pth"
    torch.save(
        {
            "version": 1,
            "method": "QLung-AST",
            "seed": args.seed,
            "epoch": args.epochs,
            "model_state_dict": cpu_state(model),
            "history": history,
            "paper_parameters": {
                "lambda": LAMBDA_DFAM,
                "gamma": GAMMA,
                "target_margin": TARGET_MARGIN,
                "angular_scale": ANGULAR_SCALE,
                "dfam_scale": DFAM_SCALE,
                "quality_scale": QUALITY_SCALE,
                "learning_rate": 5e-5,
                "batch_size": args.batch_size,
                "tta": False,
            },
        },
        checkpoint,
    )
    print(f"训练完成: {checkpoint}")


def load_model(path: Path, model_dir: str, device: torch.device):
    checkpoint = safe_torch_load(path)
    if checkpoint.get("method") != "QLung-AST":
        raise ValueError(f"不是 QLung-AST 权重: {path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"权重中没有 model_state_dict: {path}")
    model = QLungAST(model_dir, pretrained=False).to(device)
    model.load_state_dict(
        {str(name).removeprefix("module."): value for name, value in state.items()},
        strict=True,
    )
    return model.eval(), int(checkpoint["seed"])


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sum = matrix.sum(1, keepdims=True)
    normalized = np.divide(
        matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum != 0
    )
    text = np.empty_like(matrix, dtype=object)
    for row in range(4):
        for column in range(4):
            text[row, column] = f"{normalized[row, column]:.1%}\n({matrix[row, column]})"
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        normalized, annot=text, fmt="", cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis
    )
    axis.set(title="QLung AST Official Test (No TTA)", xlabel="Predicted", ylabel="True")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


@torch.inference_mode()
def run_evaluate(args) -> None:
    paths = [Path(value).expanduser().resolve() for value in args.checkpoint]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"checkpoint 不完整: {paths}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    dataset = CachedFeatures(args.test_cache, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    loaded = [load_model(path, args.model_dir, device) for path in paths]
    models = [item[0] for item in loaded]
    seeds = [item[1] for item in loaded]
    labels_all, ensemble_all = [], []
    individual_all = [[] for _ in models]
    for features, labels, _ in tqdm(loader, desc="official test, no TTA"):
        features = features.to(device, non_blocking=True)
        if device.type == "cpu":
            features = features.float()
        probability_sum = 0
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            for index, model in enumerate(models):
                logits, _ = model(features)
                probability = logits.float().softmax(1)
                probability_sum = probability_sum + probability
                individual_all[index].extend(probability.argmax(1).cpu().tolist())
        ensemble_all.extend(probability_sum.argmax(1).cpu().tolist())
        labels_all.extend(labels.tolist())

    individuals = []
    for seed, prediction in zip(seeds, individual_all):
        result = calc_icbhi_score(labels_all, prediction)
        individuals.append({"seed": seed, **result.to_dict()})
        print(
            f"seed={seed} score={result.score:.4f} "
            f"SE={result.sensitivity:.4f} SP={result.specificity:.4f}"
        )
    result = calc_icbhi_score(labels_all, ensemble_all)
    legacy = calc_legacy_macro_ovr_score(labels_all, ensemble_all)
    report = classification_report(
        labels_all, ensemble_all, labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    print("=" * 76)
    print(f"models={len(models)} seeds={seeds} TTA=False")
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
            "method": "QLung-AST probability ensemble",
            "checkpoints": [str(path) for path in paths],
            "seeds": seeds,
            "tta": False,
            "official_icbhi": result.to_dict(),
            "legacy_macro_ovr": legacy,
            "individual_models": individuals,
        },
    )
    print(f"结果保存到: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="AST + QLung for ICBHI")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--audio-dir", default=AUDIO_DIR)
    prepare.add_argument("--csv-path", default=CSV_PATH)
    prepare.add_argument("--model-dir", default=MODEL_DIR)
    prepare.add_argument("--cache-dir", default=CACHE_DIR)
    prepare.add_argument("--splits", nargs="+", default=["train", "test"])
    prepare.add_argument("--feature-batch-size", type=int, default=16)
    prepare.add_argument("--force", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--model-dir", default=MODEL_DIR)
    train.add_argument("--train-cache", default=TRAIN_CACHE)
    train.add_argument("--output-dir", default="runs/qlung_ast")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--no-amp", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", nargs="+", required=True)
    evaluate.add_argument("--model-dir", default=MODEL_DIR)
    evaluate.add_argument("--test-cache", default=TEST_CACHE)
    evaluate.add_argument("--output-dir", default="evaluation/qlung_ast")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--num-workers", type=int, default=2)
    evaluate.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if not set(args.splits).issubset({"train", "test"}):
            raise ValueError("splits 只能包含 train/test。")
        run_prepare(args)
    elif args.command == "train":
        run_train(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()