"""Train DS-MSAN with a patient-disjoint internal validation split."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASS_NAMES, ICBHIDataset, ICBHIView
from frontend import FrontendConfig, MelFrontend
from metrics import ICBHIResult, calc_icbhi_score
from model import InnovativeResNet


@dataclass
class Config:
    audio_dir: str = (
        "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
    )
    csv_path: str = (
        "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
    )
    output_dir: str = "runs/dsmsan"

    seed: int = 24923
    epochs: int = 80
    batch_size: int = 64
    num_workers: int = 8
    patience: int = 15
    val_ratio: float = 0.20
    split_candidates: int = 300

    duration: float = 8.0
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 50.0
    f_max: float = 2500.0
    top_db: float = 80.0
    freq_mask_param: int = 15
    time_mask_param: int = 35
    normalize_spectrogram: bool = False

    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 2.0
    focal_gamma: float = 1.5
    class_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.5, 2.0)
    mixup_probability: float = 0.50
    mixup_alpha: float = 0.20

    use_amp: bool = True
    cache_dataset: bool = True

    def frontend_config(self) -> FrontendConfig:
        return FrontendConfig(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            top_db=self.top_db,
            freq_mask_param=self.freq_mask_param,
            time_mask_param=self.time_mask_param,
            normalize=self.normalize_spectrogram,
        )


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Train DS-MSAN on ICBHI with patient-disjoint validation."
    )
    parser.add_argument("--audio-dir", default=defaults.audio_dir)
    parser.add_argument("--csv-path", default=defaults.csv_path)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    return Config(
        audio_dir=args.audio_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patience=args.patience,
        val_ratio=args.val_ratio,
        learning_rate=args.learning_rate,
        use_amp=not args.disable_amp,
        cache_dataset=not args.no_cache,
    )


def validate_config(config: Config) -> None:
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs 和 batch_size 必须为正整数。")
    if config.num_workers < 0:
        raise ValueError("num_workers 不能为负数。")
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("val_ratio 必须位于 (0, 1) 区间。")
    if not 0.0 <= config.mixup_probability <= 1.0:
        raise ValueError("mixup_probability 必须位于 [0, 1] 区间。")
    if config.mixup_alpha <= 0:
        raise ValueError("mixup_alpha 必须为正数。")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class FocalLoss(nn.Module):
    """Standard multi-class focal loss with optional per-class alpha."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma 不能为负数。")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction 必须为 mean、sum 或 none。")
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", alpha.float())
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probabilities = target_log_probs.exp()
        loss = -((1.0 - probabilities).pow(self.gamma)) * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _distribution(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=4).astype(np.float64)
    return counts / max(1.0, counts.sum())


def make_patient_split(
    dataset: ICBHIDataset,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a patient-disjoint split with a representative class balance."""

    indices = np.arange(len(dataset), dtype=np.int64)
    labels = dataset.labels.cpu().numpy()
    groups = np.asarray(dataset.patient_ids, dtype=object)
    if np.unique(groups).size < 2:
        raise RuntimeError("至少需要两名患者才能划分训练集和验证集。")

    target_distribution = _distribution(labels)
    splitter = GroupShuffleSplit(
        n_splits=config.split_candidates,
        test_size=config.val_ratio,
        random_state=config.seed,
    )

    best_split = None
    best_objective = float("inf")
    for train_positions, val_positions in splitter.split(indices, labels, groups):
        val_labels = labels[val_positions]
        distribution_error = float(
            np.abs(_distribution(val_labels) - target_distribution).sum()
        )
        size_error = abs(len(val_positions) / len(indices) - config.val_ratio)
        missing_class_penalty = 10.0 if np.unique(val_labels).size < 4 else 0.0
        objective = distribution_error + size_error + missing_class_penalty
        if objective < best_objective:
            best_objective = objective
            best_split = (train_positions, val_positions)

    if best_split is None:
        raise RuntimeError("无法生成患者级训练/验证划分。")
    train_indices, val_indices = best_split
    train_patients = set(groups[train_indices])
    val_patients = set(groups[val_indices])
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise RuntimeError(f"训练集和验证集患者重叠: {sorted(overlap)}")

    print(
        f"✅ 患者级划分 | train={len(train_indices)} cycles/"
        f"{len(train_patients)} patients | val={len(val_indices)} cycles/"
        f"{len(val_patients)} patients"
    )
    print(f"📊 全部 train 类别比例: {target_distribution.round(4).tolist()}")
    print(
        "📊 内部 val 类别比例: "
        f"{_distribution(labels[val_indices]).round(4).tolist()}"
    )
    return train_indices, val_indices


def make_loaders(
    config: Config,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    official_train = ICBHIDataset(
        data_path=config.audio_dir,
        split="train",
        metadatafile=config.csv_path,
        duration=config.duration,
        samplerate=config.sample_rate,
        cache=config.cache_dataset,
    )
    train_indices, val_indices = make_patient_split(official_train, config)
    train_dataset = ICBHIView(official_train, train_indices)
    val_dataset = ICBHIView(official_train, val_indices)

    common = {
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed + 1),
        **common,
    )
    return train_loader, val_loader


@torch.inference_mode()
def evaluate_validation(
    model: nn.Module,
    frontend: MelFrontend,
    loader: DataLoader,
    device: torch.device,
) -> ICBHIResult:
    model.eval()
    frontend.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        spectrograms = frontend(waveforms, augment=False)
        predictions = model(spectrograms).argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
    return calc_icbhi_score(all_labels, all_predictions)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_score: float,
    config: Config,
    validation: ICBHIResult,
) -> None:
    torch.save(
        {
            "checkpoint_format_version": 2,
            "epoch": int(epoch),
            "model_name": "InnovativeResNet/DS-MSAN",
            "model_state_dict": _cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_score": float(best_val_score),
            "metric": "official_icbhi",
            "validation_metrics": validation.to_dict(),
            "config": asdict(config),
            "frontend_config": config.frontend_config().to_dict(),
            "class_names": list(CLASS_NAMES),
        },
        path,
    )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def print_metrics(prefix: str, metrics: ICBHIResult) -> None:
    recalls = ", ".join(
        f"{name}={value:.4f}"
        for name, value in zip(CLASS_NAMES, metrics.class_recall)
    )
    print(
        f"{prefix} Score={metrics.score:.4f} | SE={metrics.sensitivity:.4f} | "
        f"SP={metrics.specificity:.4f} | Recall[{recalls}]"
    )
    print(metrics.confusion_matrix)


def main() -> None:
    config = parse_args()
    validate_config(config)
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    output_dir = Path(config.output_dir) / f"seed_{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 76)
    print(
        f"Seed={config.seed} | Device={device} | AMP={amp_enabled} | "
        f"Output={output_dir}"
    )
    print("official test 不参与训练、早停或选权重。")
    print("=" * 76)

    train_loader, val_loader = make_loaders(
        config,
        pin_memory=device.type == "cuda",
    )
    model = InnovativeResNet(num_classes=4).to(device)
    frontend = MelFrontend(config.frontend_config()).to(device)
    criterion = FocalLoss(
        alpha=torch.tensor(config.class_weights, dtype=torch.float32),
        gamma=config.focal_gamma,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    scaler = make_grad_scaler(amp_enabled)

    best_val_score = -1.0
    epochs_without_improvement = 0
    best_path = output_dir / "best_checkpoint.pth"
    last_path = output_dir / "last_checkpoint.pth"

    for epoch in range(config.epochs):
        model.train()
        frontend.train()
        running_loss = 0.0
        seen_samples = 0
        skipped_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:02d}/{config.epochs} [Train]",
            leave=False,
        )
        for waveforms, labels, _, _ in progress:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            linear_mel = frontend.linear_mel(waveforms)
            use_mixup = (
                waveforms.size(0) > 1
                and torch.rand((), device=device).item() < config.mixup_probability
            )
            if use_mixup:
                mix_weight = float(
                    np.random.beta(config.mixup_alpha, config.mixup_alpha)
                )
                permutation = torch.randperm(waveforms.size(0), device=device)
                linear_mel = (
                    mix_weight * linear_mel
                    + (1.0 - mix_weight) * linear_mel[permutation]
                )
            else:
                mix_weight = 1.0
                permutation = None

            spectrograms = frontend.finish(linear_mel, augment=True)
            if not torch.isfinite(spectrograms).all():
                skipped_batches += 1
                continue

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(spectrograms)
                if permutation is None:
                    loss = criterion(logits, labels)
                else:
                    loss = (
                        mix_weight * criterion(logits, labels)
                        + (1.0 - mix_weight) * criterion(logits, labels[permutation])
                    )

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.grad_clip,
            )
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue

            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels.size(0))
            running_loss += float(loss.detach().float()) * batch_size
            seen_samples += batch_size
            progress.set_postfix(
                loss=f"{float(loss.detach().float()):.4f}",
                skip=skipped_batches,
            )

        if seen_samples == 0:
            raise RuntimeError("当前 epoch 没有成功更新任何参数。")
        scheduler.step()

        validation = evaluate_validation(model, frontend, val_loader, device)
        print(
            f"Epoch {epoch + 1:02d} | Train Loss={running_loss / seen_samples:.4f} "
            f"| LR={optimizer.param_groups[0]['lr']:.3e} | Skipped={skipped_batches}"
        )
        print_metrics("Internal Validation |", validation)

        if validation.score > best_val_score + 1e-8:
            best_val_score = validation.score
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_val_score,
                config,
                validation,
            )
            print(f"  🌟 保存最佳内部验证权重: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"  ⚠️ 未提升: {epochs_without_improvement}/{config.patience}"
            )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            best_val_score,
            config,
            validation,
        )
        if epochs_without_improvement >= config.patience:
            print(f"🛑 连续 {config.patience} 轮未提升，触发早停。")
            break

    print("=" * 76)
    print(f"✅ 训练结束，最佳内部验证 Score={best_val_score:.4f}")
    print(f"最佳权重: {best_path}")
    print("请使用 evaluate.py 在 official test 上独立评估一次。")


if __name__ == "__main__":
    main()