"""Patient-stratified cross-validation training for DS-MSAN."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import CLASS_NAMES, ICBHIDataset, ICBHIView
from frontend import FrontendConfig, MelFrontend
from metrics import calc_icbhi_score
from model import InnovativeResNet


DEFAULT_AUDIO_DIR = (
    "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
)
DEFAULT_CSV_PATH = (
    "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one patient-stratified DS-MSAN fold."
    )
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", default="runs/dsmsan_cv")
    parser.add_argument("--seed", type=int, default=24923)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--sampling-power", type=float, default=0.50)
    parser.add_argument("--mixup-probability", type=float, default=0.25)
    parser.add_argument("--mixup-alpha", type=float, default=0.20)
    parser.add_argument("--ema-decay", type=float, default=0.995)

    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_folds < 2:
        raise ValueError("num_folds 至少为 2。")
    if not 0 <= args.fold < args.num_folds:
        raise ValueError("fold 必须位于 [0, num_folds)。")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs 和 batch_size 必须为正整数。")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数。")
    if args.patience <= 3:
        raise ValueError("patience 必须大于学习率调度器 patience。")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label_smoothing 必须位于 [0, 1)。")
    if not 0 <= args.sampling_power <= 1:
        raise ValueError("sampling_power 必须位于 [0, 1]。")
    if not 0 <= args.mixup_probability <= 1:
        raise ValueError("mixup_probability 必须位于 [0, 1]。")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema_decay 必须位于 [0, 1)。")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_fold(dataset, num_folds, fold, seed):
    indices = np.arange(len(dataset), dtype=np.int64)
    labels = dataset.labels.cpu().numpy().astype(np.int64)
    patients = np.asarray(dataset.patient_ids, dtype=object)

    if len(np.unique(patients)) < num_folds:
        raise RuntimeError("患者数量少于交叉验证折数。")

    splitter = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )
    folds = list(splitter.split(indices, labels, patients))
    train_indices, val_indices = folds[fold]

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)

    train_patients = set(patients[train_indices].tolist())
    val_patients = set(patients[val_indices].tolist())

    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise RuntimeError(f"训练和验证患者重叠: {sorted(overlap)}")

    expected_classes = {0, 1, 2, 3}
    train_classes = set(labels[train_indices].tolist())
    val_classes = set(labels[val_indices].tolist())

    if train_classes != expected_classes:
        raise RuntimeError(f"训练折缺少类别: {sorted(train_classes)}")
    if val_classes != expected_classes:
        raise RuntimeError(f"验证折缺少类别: {sorted(val_classes)}")

    return train_indices, val_indices


def make_sampler(labels, power, seed):
    labels = labels.detach().cpu().long()
    counts = torch.bincount(labels, minlength=4).double()

    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"训练折缺少类别: {missing}")

    class_weights = (counts.sum() / counts).pow(power)
    sample_weights = class_weights[labels]

    expected = counts * class_weights
    expected = expected / expected.sum()

    print(f"原始类别数量: {counts.long().tolist()}")
    print(f"平滑采样比例: {expected.numpy().round(4).tolist()}")

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def augment_waveforms(waveforms):
    """Training-only gain, time shift, and additive noise."""

    augmented = waveforms.float().clone()
    batch_size = augmented.size(0)
    device = augmented.device

    # 随机增益，概率 0.5，范围 ±6 dB
    gain_mask = (
        torch.rand(batch_size, 1, 1, device=device) < 0.50
    )
    gain_db = (
        torch.rand(batch_size, 1, 1, device=device) * 12.0 - 6.0
    )
    gain = torch.pow(10.0, gain_db / 20.0)
    augmented = torch.where(gain_mask, augmented * gain, augmented)

    # 随机循环平移，概率 0.5，最大 0.5 秒
    shift_mask = torch.rand(batch_size, device=device) < 0.50
    shifts = torch.randint(
        low=-8000,
        high=8001,
        size=(batch_size,),
        device=device,
    )

    shifted = []
    for index in range(batch_size):
        if bool(shift_mask[index]):
            shifted.append(
                torch.roll(
                    augmented[index],
                    shifts=int(shifts[index].item()),
                    dims=-1,
                )
            )
        else:
            shifted.append(augmented[index])

    augmented = torch.stack(shifted, dim=0)

    # 随机噪声，概率 0.35，SNR 20–35 dB
    noise_mask = (
        torch.rand(batch_size, 1, 1, device=device) < 0.35
    )
    snr_db = (
        torch.rand(batch_size, 1, 1, device=device) * 15.0 + 20.0
    )

    signal_rms = (
        augmented.square()
        .mean(dim=(-2, -1), keepdim=True)
        .sqrt()
        .clamp_min(1e-6)
    )
    noise_rms = signal_rms / torch.pow(10.0, snr_db / 20.0)
    noise = torch.randn_like(augmented) * noise_rms
    augmented = augmented + noise * noise_mask

    return augmented.clamp(-1.0, 1.0)


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)

        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source_state = model.state_dict()

        for name, averaged in self.module.state_dict().items():
            source = source_state[name].detach()

            if torch.is_floating_point(averaged):
                averaged.mul_(self.decay).add_(
                    source,
                    alpha=1.0 - self.decay,
                )
            else:
                averaged.copy_(source)


def cpu_state_dict(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


@torch.inference_mode()
def validate(model, frontend, loader, device):
    model.eval()
    frontend.eval()

    labels_all = []
    predictions_all = []

    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        spectrograms = frontend(waveforms, augment=False)
        predictions = model(spectrograms).argmax(dim=1)

        labels_all.extend(labels.tolist())
        predictions_all.extend(predictions.cpu().tolist())

    return calc_icbhi_score(labels_all, predictions_all)


def save_checkpoint(
    path,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_epoch,
    best_score,
    args,
    frontend_config,
    metrics,
    fold_seed,
):
    config = vars(args).copy()
    config.update(
        {
            "duration": 8.0,
            "sample_rate": 16000,
        }
    )

    torch.save(
        {
            "checkpoint_format_version": 3,
            "model_name": "InnovativeResNet/DS-MSAN",
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "fold": int(args.fold),
            "num_folds": int(args.num_folds),
            "training_seed": int(fold_seed),
            "selected_weights": "ema_state_dict",
            "model_state_dict": cpu_state_dict(model),
            "ema_state_dict": cpu_state_dict(ema.module),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_score": float(best_score),
            "validation_metrics": metrics.to_dict(),
            "metric": "official_icbhi",
            "config": config,
            "frontend_config": frontend_config.to_dict(),
            "class_names": list(CLASS_NAMES),
        },
        path,
    )


def main():
    args = parse_args()
    validate_args(args)

    fold_seed = args.seed + args.fold * 1009
    seed_everything(fold_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = (
        device.type == "cuda" and not args.disable_amp
    )

    output_dir = (
        Path(args.output_dir)
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"Fold={args.fold}/{args.num_folds - 1} | "
        f"Seed={fold_seed} | Device={device} | AMP={amp_enabled}"
    )
    print(f"Output={output_dir}")
    print("official test 不参与训练、早停和权重选择。")
    print("=" * 76)

    dataset = ICBHIDataset(
        data_path=args.audio_dir,
        split="train",
        metadatafile=args.csv_path,
        duration=8.0,
        samplerate=16000,
        cache=not args.no_cache,
    )

    train_indices, val_indices = make_fold(
        dataset,
        num_folds=args.num_folds,
        fold=args.fold,
        seed=args.seed,
    )

    train_dataset = ICBHIView(dataset, train_indices)
    val_dataset = ICBHIView(dataset, val_indices)

    train_patients = sorted(set(train_dataset.patient_ids))
    val_patients = sorted(set(val_dataset.patient_ids))

    split_data = {
        "fold": args.fold,
        "train_indices": train_indices.tolist(),
        "validation_indices": val_indices.tolist(),
        "train_patients": train_patients,
        "validation_patients": val_patients,
        "train_class_counts": torch.bincount(
            train_dataset.labels,
            minlength=4,
        ).tolist(),
        "validation_class_counts": torch.bincount(
            val_dataset.labels,
            minlength=4,
        ).tolist(),
    }

    (output_dir / "split.json").write_text(
        json.dumps(split_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"train={len(train_dataset)} cycles/"
        f"{len(train_patients)} patients"
    )
    print(
        f"validation={len(val_dataset)} cycles/"
        f"{len(val_patients)} patients"
    )

    sampler = make_sampler(
        train_dataset.labels,
        power=args.sampling_power,
        seed=fold_seed,
    )

    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        generator=torch.Generator().manual_seed(fold_seed + 1),
        **loader_common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(fold_seed + 2),
        **loader_common,
    )

    frontend_config = FrontendConfig(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=128,
        f_min=50.0,
        f_max=2500.0,
        top_db=80.0,
        freq_mask_param=12,
        time_mask_param=25,
        normalize=False,
    )

    frontend = MelFrontend(frontend_config).to(device)
    model = InnovativeResNet(num_classes=4).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-5,
    )

    scaler = make_scaler(amp_enabled)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    best_path = output_dir / "best_checkpoint.pth"
    last_path = output_dir / "last_checkpoint.pth"

    for epoch in range(args.epochs):
        model.train()
        frontend.train()

        loss_sum = 0.0
        sample_count = 0
        skipped_batches = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Fold {args.fold} | "
                f"Epoch {epoch + 1:02d}/{args.epochs}"
            ),
            leave=False,
        )

        for waveforms, labels, _, _ in progress:
            waveforms = waveforms.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            waveforms = augment_waveforms(waveforms)
            optimizer.zero_grad(set_to_none=True)

            linear_mel = frontend.linear_mel(waveforms)

            use_mixup = (
                waveforms.size(0) > 1
                and float(torch.rand((), device=device))
                < args.mixup_probability
            )

            if use_mixup:
                mix_weight = float(
                    np.random.beta(
                        args.mixup_alpha,
                        args.mixup_alpha,
                    )
                )
                permutation = torch.randperm(
                    waveforms.size(0),
                    device=device,
                )
                linear_mel = (
                    mix_weight * linear_mel
                    + (1.0 - mix_weight)
                    * linear_mel[permutation]
                )
            else:
                mix_weight = 1.0
                permutation = None

            spectrograms = frontend.finish(
                linear_mel,
                augment=True,
            )

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
                        mix_weight
                        * criterion(logits, labels)
                        + (1.0 - mix_weight)
                        * criterion(
                            logits,
                            labels[permutation],
                        )
                    )

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0,
            )

            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue

            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            current_batch = labels.size(0)
            loss_sum += (
                float(loss.detach().float())
                * current_batch
            )
            sample_count += current_batch

            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                skipped=skipped_batches,
            )

        if sample_count == 0:
            raise RuntimeError(
                "当前 epoch 没有成功更新任何参数。"
            )

        validation = validate(
            ema.module,
            frontend,
            val_loader,
            device,
        )

        improved = validation.score > best_score + 1e-8

        if improved:
            best_score = validation.score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation.score)
        new_lr = optimizer.param_groups[0]["lr"]

        train_loss = loss_sum / sample_count

        print(
            f"Fold {args.fold} | Epoch {epoch + 1:02d} | "
            f"Loss={train_loss:.4f} | "
            f"LR={new_lr:.3e} | "
            f"Skipped={skipped_batches}"
        )
        print(
            f"Validation | Score={validation.score:.4f} | "
            f"SE={validation.sensitivity:.4f} | "
            f"SP={validation.specificity:.4f}"
        )
        print(validation.confusion_matrix)

        if new_lr < old_lr:
            print(
                f"学习率降低: {old_lr:.3e} -> {new_lr:.3e}"
            )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": new_lr,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "validation": validation.to_dict(),
            }
        )

        (output_dir / "history.json").write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if improved:
            save_checkpoint(
                best_path,
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_epoch,
                best_score,
                args,
                frontend_config,
                validation,
                fold_seed,
            )
            print(f"保存最佳 EMA 权重: {best_path}")
        else:
            print(
                "未提升: "
                f"{epochs_without_improvement}/"
                f"{args.patience}"
            )

        save_checkpoint(
            last_path,
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            best_epoch,
            best_score,
            args,
            frontend_config,
            validation,
            fold_seed,
        )

        if epochs_without_improvement >= args.patience:
            print("触发早停。")
            break

    summary = {
        "fold": args.fold,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 76)
    print(
        f"Fold {args.fold} 完成 | "
        f"Best Score={best_score:.4f} "
        f"@ Epoch {best_epoch}"
    )
    print(f"Best checkpoint: {best_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()