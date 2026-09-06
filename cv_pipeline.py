"""Patient-level cross-validation training and ensemble evaluation for DS-MSAN."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import CLASS_NAMES, ICBHIDataset, ICBHIView
from frontend import FrontendConfig, MelFrontend
from metrics import calc_icbhi_score, calc_legacy_macro_ovr_score
from model import InnovativeResNet


DEFAULT_AUDIO_DIR = "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
DEFAULT_CSV_PATH = (
    "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def frontend_config() -> FrontendConfig:
    return FrontendConfig(
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


def search_patient_folds(
    dataset: ICBHIDataset,
    num_folds: int,
    seed: int,
    max_candidates: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, float]:
    """寻找患者不重叠、每个验证折都包含四类的划分。"""

    indices = np.arange(len(dataset), dtype=np.int64)
    labels = dataset.labels.cpu().numpy().astype(np.int64)
    patients = np.asarray(dataset.patient_ids, dtype=object)

    unique_patients = set(patients.tolist())
    if len(unique_patients) < num_folds:
        raise RuntimeError("患者数量少于交叉验证折数。")

    expected_classes = set(range(len(CLASS_NAMES)))

    target_distribution = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    ).astype(np.float64)
    target_distribution /= target_distribution.sum()

    target_size_ratio = 1.0 / num_folds

    best_folds = None
    best_seed = None
    best_objective = float("inf")
    valid_candidates = 0

    for offset in range(max_candidates):
        candidate_seed = seed + offset

        splitter = StratifiedGroupKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=candidate_seed,
        )

        candidate_folds = []
        validation_patients = set()
        objective = 0.0
        valid = True

        for train_indices, val_indices in splitter.split(
            indices,
            labels,
            patients,
        ):
            train_indices = np.asarray(train_indices, dtype=np.int64)
            val_indices = np.asarray(val_indices, dtype=np.int64)

            train_patients = set(patients[train_indices].tolist())
            val_patients = set(patients[val_indices].tolist())

            if train_patients.intersection(val_patients):
                valid = False
                break

            if validation_patients.intersection(val_patients):
                valid = False
                break

            validation_patients.update(val_patients)

            train_classes = set(labels[train_indices].tolist())
            val_classes = set(labels[val_indices].tolist())

            if train_classes != expected_classes:
                valid = False
                break

            if val_classes != expected_classes:
                valid = False
                break

            val_distribution = np.bincount(
                labels[val_indices],
                minlength=len(CLASS_NAMES),
            ).astype(np.float64)
            val_distribution /= val_distribution.sum()

            distribution_error = np.abs(
                val_distribution - target_distribution
            ).sum()

            size_error = abs(
                len(val_indices) / len(indices) - target_size_ratio
            )

            objective += (
                float(distribution_error)
                + 2.0 * float(size_error)
            )

            candidate_folds.append(
                (train_indices, val_indices)
            )

        if not valid:
            continue

        if validation_patients != unique_patients:
            continue

        if len(candidate_folds) != num_folds:
            continue

        valid_candidates += 1

        if objective < best_objective:
            best_folds = candidate_folds
            best_seed = candidate_seed
            best_objective = objective

        if valid_candidates >= 25:
            break

    if best_folds is None or best_seed is None:
        raise RuntimeError(
            f"尝试 {max_candidates} 个随机种子后，"
            "无法生成每折均包含四类的患者级划分。"
        )

    return best_folds, best_seed, best_objective


def make_sampler(
    labels: torch.Tensor,
    power: float,
    seed: int,
):
    labels = labels.detach().cpu().long()

    counts = torch.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    ).double()

    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"训练折缺少类别: {missing}")

    # power=0 表示原始分布，power=1 表示完全均衡。
    # 默认 0.25，只做温和补偿，避免模型塌缩到少数类别。
    class_weights = (counts.sum() / counts).pow(power)
    sample_weights = class_weights[labels]

    expected = counts * class_weights
    expected /= expected.sum()

    print(f"原始类别数量: {counts.long().tolist()}")
    print(f"平滑采样比例: {expected.numpy().round(4).tolist()}")

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def augment_waveforms(
    waveforms: torch.Tensor,
) -> torch.Tensor:
    """训练阶段使用的轻量波形增强。"""

    augmented = waveforms.float().clone()
    batch_size = augmented.size(0)
    device = augmented.device

    # 35% 概率进行 -4dB 到 +4dB 的增益变化。
    gain_mask = (
        torch.rand(batch_size, 1, 1, device=device) < 0.35
    )
    gain_db = (
        torch.rand(batch_size, 1, 1, device=device) * 8.0 - 4.0
    )
    gain = torch.pow(10.0, gain_db / 20.0)

    augmented = torch.where(
        gain_mask,
        augmented * gain,
        augmented,
    )

    # 25% 概率循环平移，最大约 0.25 秒。
    shift_mask = torch.rand(batch_size, device=device) < 0.25
    shifts = torch.randint(
        -4000,
        4001,
        (batch_size,),
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

    # 20% 概率添加 25dB 到 40dB 的轻微噪声。
    noise_mask = (
        torch.rand(batch_size, 1, 1, device=device) < 0.20
    )
    snr_db = (
        torch.rand(batch_size, 1, 1, device=device) * 15.0
        + 25.0
    )

    signal_rms = (
        augmented.square()
        .mean(dim=(-2, -1), keepdim=True)
        .sqrt()
        .clamp_min(1e-6)
    )

    noise_rms = signal_rms / torch.pow(
        10.0,
        snr_db / 20.0,
    )

    augmented = (
        augmented
        + torch.randn_like(augmented)
        * noise_rms
        * noise_mask
    )

    return augmented.clamp(-1.0, 1.0)


class ModelEMA:
    """带预热的指数滑动平均模型。"""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
    ) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0

        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
    ) -> None:
        self.updates += 1

        # 前期降低 EMA 衰减，避免 EMA 长时间保留随机初始化。
        warm_decay = (
            (1.0 + self.updates)
            / (10.0 + self.updates)
        )
        decay = min(self.decay, warm_decay)

        source_state = model.state_dict()

        for name, averaged in self.module.state_dict().items():
            source = source_state[name].detach()

            if torch.is_floating_point(averaged):
                averaged.mul_(decay).add_(
                    source,
                    alpha=1.0 - decay,
                )
            else:
                averaged.copy_(source)


@torch.inference_mode()
def validate(
    model,
    frontend,
    loader,
    device,
):
    model.eval()
    frontend.eval()

    labels_all = []
    predictions_all = []

    for waveforms, labels, _, _ in loader:
        waveforms = waveforms.to(
            device,
            non_blocking=True,
        )

        spectrograms = frontend(
            waveforms,
            augment=False,
        )

        predictions = model(
            spectrograms
        ).argmax(dim=1)

        labels_all.extend(labels.tolist())
        predictions_all.extend(
            predictions.cpu().tolist()
        )

    return calc_icbhi_score(
        labels_all,
        predictions_all,
    )


def cpu_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


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
    front_config,
    metrics,
    fold_seed,
    split_seed,
):
    stored_config = vars(args).copy()
    stored_config.update(
        {
            "duration": 8.0,
            "sample_rate": 16000,
        }
    )

    torch.save(
        {
            "checkpoint_format_version": 4,
            "model_name": "InnovativeResNet/DS-MSAN",
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "fold": int(args.fold),
            "num_folds": int(args.num_folds),
            "training_seed": int(fold_seed),
            "split_seed": int(split_seed),
            "selected_weights": "ema_state_dict",
            "model_state_dict": cpu_state_dict(model),
            "ema_state_dict": cpu_state_dict(ema.module),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_score": float(best_score),
            "validation_metrics": metrics.to_dict(),
            "metric": "official_icbhi",
            "config": stored_config,
            "frontend_config": front_config.to_dict(),
            "class_names": list(CLASS_NAMES),
        },
        path,
    )


def run_train(args) -> None:
    if (
        args.num_folds < 2
        or not 0 <= args.fold < args.num_folds
    ):
        raise ValueError(
            "fold 必须位于 [0, num_folds)，"
            "且 num_folds 至少为 2。"
        )

    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.num_workers < 0
    ):
        raise ValueError("训练参数无效。")

    if not 0.0 <= args.sampling_power <= 1.0:
        raise ValueError(
            "sampling-power 必须位于 [0, 1]。"
        )

    if not 0.0 <= args.mixup_probability <= 1.0:
        raise ValueError(
            "mixup-probability 必须位于 [0, 1]。"
        )

    if args.mixup_alpha <= 0:
        raise ValueError(
            "mixup-alpha 必须大于 0。"
        )

    fold_seed = (
        args.seed
        + args.fold * 1009
    )

    seed_everything(fold_seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.disable_amp
    )

    output_dir = (
        Path(args.output_dir)
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print(
        f"Fold={args.fold}/{args.num_folds - 1} | "
        f"Seed={fold_seed} | "
        f"Device={device} | "
        f"AMP={amp_enabled}"
    )
    print(f"Output={output_dir}")
    print(
        "official test 不参与训练、早停或权重选择。"
    )
    print("=" * 76)

    dataset = ICBHIDataset(
        data_path=args.audio_dir,
        split="train",
        metadatafile=args.csv_path,
        duration=8.0,
        samplerate=16000,
        cache=not args.no_cache,
    )

    folds, split_seed, split_objective = (
        search_patient_folds(
            dataset,
            args.num_folds,
            args.seed,
            args.split_candidates,
        )
    )

    train_indices, val_indices = folds[args.fold]
    labels_np = dataset.labels.cpu().numpy()

    print(
        f"选定划分 seed={split_seed} | "
        f"objective={split_objective:.4f}"
    )

    print(
        f"Fold {args.fold} | train="
        f"{np.bincount(labels_np[train_indices], minlength=4).tolist()} | "
        f"val="
        f"{np.bincount(labels_np[val_indices], minlength=4).tolist()}"
    )

    train_dataset = ICBHIView(
        dataset,
        train_indices,
    )
    val_dataset = ICBHIView(
        dataset,
        val_indices,
    )

    train_patients = sorted(
        set(train_dataset.patient_ids)
    )
    val_patients = sorted(
        set(val_dataset.patient_ids)
    )

    if set(train_patients).intersection(val_patients):
        raise RuntimeError(
            "训练和验证患者重叠。"
        )

    write_json(
        output_dir / "split.json",
        {
            "fold": args.fold,
            "split_seed": split_seed,
            "train_indices": train_indices.tolist(),
            "validation_indices": val_indices.tolist(),
            "train_patients": train_patients,
            "validation_patients": val_patients,
        },
    )

    sampler = make_sampler(
        train_dataset.labels,
        args.sampling_power,
        fold_seed,
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
        generator=torch.Generator().manual_seed(
            fold_seed + 1
        ),
        **loader_common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(
            fold_seed + 2
        ),
        **loader_common,
    )

    front_config = frontend_config()

    frontend = MelFrontend(
        front_config
    ).to(device)

    model = InnovativeResNet(
        num_classes=4
    ).to(device)

    ema = ModelEMA(
        model,
        args.ema_decay,
    )

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

    scaler = make_scaler(
        amp_enabled
    )

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    best_path = (
        output_dir
        / "best_checkpoint.pth"
    )

    last_path = (
        output_dir
        / "last_checkpoint.pth"
    )

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

            if not args.disable_waveform_augment:
                waveforms = augment_waveforms(
                    waveforms
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            linear_mel = frontend.linear_mel(
                waveforms
            )

            use_mixup = (
                waveforms.size(0) > 1
                and float(
                    torch.rand(
                        (),
                        device=device,
                    )
                )
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

            if not torch.isfinite(
                spectrograms
            ).all():
                skipped_batches += 1
                continue

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(
                    spectrograms
                )

                if permutation is None:
                    loss = criterion(
                        logits,
                        labels,
                    )
                else:
                    loss = (
                        mix_weight
                        * criterion(
                            logits,
                            labels,
                        )
                        + (1.0 - mix_weight)
                        * criterion(
                            logits,
                            labels[permutation],
                        )
                    )

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            scaler.scale(
                loss
            ).backward()

            # 每个 batch 最多调用一次 unscale_。
            scaler.unscale_(optimizer)

            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0,
            )

            if not torch.isfinite(
                gradient_norm
            ):
                optimizer.zero_grad(
                    set_to_none=True
                )

                # unscale_ 后即使跳过 optimizer.step，
                # 也必须调用 scaler.update() 重置状态。
                scaler.update()

                skipped_batches += 1
                continue

            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            current_batch = int(
                labels.size(0)
            )

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

        improved = (
            validation.score
            > best_score + 1e-8
        )

        if improved:
            best_score = validation.score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        old_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        scheduler.step(
            validation.score
        )

        new_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        train_loss = (
            loss_sum / sample_count
        )

        print(
            f"Fold {args.fold} | "
            f"Epoch {epoch + 1:02d} | "
            f"Loss={train_loss:.4f} | "
            f"LR={new_lr:.3e} | "
            f"Skipped={skipped_batches}"
        )

        print(
            f"Validation | "
            f"Score={validation.score:.4f} | "
            f"SE={validation.sensitivity:.4f} | "
            f"SP={validation.specificity:.4f}"
        )

        print(
            validation.confusion_matrix
        )

        if new_lr < old_lr:
            print(
                f"学习率降低: "
                f"{old_lr:.3e} -> {new_lr:.3e}"
            )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": new_lr,
                "skipped_batches": skipped_batches,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "validation": validation.to_dict(),
            }
        )

        write_json(
            output_dir / "history.json",
            history,
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
                front_config,
                validation,
                fold_seed,
                split_seed,
            )

            print(
                f"保存最佳 EMA 权重: "
                f"{best_path}"
            )
        else:
            print(
                f"未提升: "
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
            front_config,
            validation,
            fold_seed,
            split_seed,
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print("触发早停。")
            break

    write_json(
        output_dir / "summary.json",
        {
            "fold": args.fold,
            "split_seed": split_seed,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "checkpoint": str(
                best_path.resolve()
            ),
        },
    )

    print("=" * 76)
    print(
        f"Fold {args.fold} 完成 | "
        f"Best Score={best_score:.4f} "
        f"@ Epoch {best_epoch}"
    )
    print(
        f"Best checkpoint: {best_path}"
    )
    print("=" * 76)


def safe_load(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def extract_state_dict(
    checkpoint,
) -> dict[str, torch.Tensor]:
    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            "checkpoint 必须为字典。"
        )

    for key in (
        "ema_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        candidate = checkpoint.get(key)

        if (
            isinstance(candidate, Mapping)
            and candidate
        ):
            state_dict = candidate
            break
    else:
        state_dict = checkpoint

    cleaned = {}

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"checkpoint 中的 "
                f"{name!r} 不是 Tensor。"
            )

        cleaned[
            str(name).removeprefix("module.")
        ] = tensor

    return cleaned


def checkpoint_config(
    checkpoint,
) -> tuple[FrontendConfig, float]:
    if not isinstance(
        checkpoint,
        Mapping,
    ):
        return frontend_config(), 8.0

    values = checkpoint.get(
        "frontend_config"
    )
    config = checkpoint.get(
        "config"
    )

    if not isinstance(values, Mapping):
        values = {}

    if not isinstance(config, Mapping):
        config = {}

    return (
        FrontendConfig.from_mapping(values),
        float(config.get("duration", 8.0)),
    )


@torch.inference_mode()
def ensemble_predict(
    models,
    frontend,
    loader,
    device,
):
    for model in models:
        model.eval()

    frontend.eval()

    labels_all = []
    predictions_all = []

    for waveforms, labels, _, _ in tqdm(
        loader,
        desc="Official Test",
    ):
        waveforms = waveforms.to(
            device,
            non_blocking=True,
        )

        spectrograms = frontend(
            waveforms,
            augment=False,
        )

        # 不使用 TTA，只对五折模型的原始 logits 求平均。
        logits = torch.stack(
            [
                model(spectrograms)
                for model in models
            ],
            dim=0,
        ).mean(dim=0)

        predictions_all.extend(
            logits.argmax(
                dim=1
            ).cpu().tolist()
        )

        labels_all.extend(
            labels.tolist()
        )

    return labels_all, predictions_all


def save_confusion_matrix(
    confusion: np.ndarray,
    path: Path,
) -> None:
    totals = confusion.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        confusion.astype(np.float64),
        totals,
        out=np.zeros_like(
            confusion,
            dtype=np.float64,
        ),
        where=totals != 0,
    )

    annotations = np.empty_like(
        confusion,
        dtype=object,
    )

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            annotations[row, column] = (
                f"{normalized[row, column]:.1%}\n"
                f"({confusion[row, column]})"
            )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1,
        linecolor="black",
        ax=axis,
    )

    axis.set_title(
        "Official Test Ensemble"
    )
    axis.set_xlabel(
        "Predicted class"
    )
    axis.set_ylabel(
        "True class"
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_evaluate(args) -> None:
    checkpoint_paths = [
        Path(value).expanduser().resolve()
        for value in args.checkpoint
    ]

    if len(set(checkpoint_paths)) != len(
        checkpoint_paths
    ):
        raise ValueError(
            "checkpoint 路径存在重复。"
        )

    missing = [
        path
        for path in checkpoint_paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"找不到 checkpoint: {missing}"
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "当前没有可用 CUDA。"
        )

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            or (
                args.device == "auto"
                and torch.cuda.is_available()
            )
        )
        else "cpu"
    )

    models = []
    front_config = None
    duration = None
    checkpoint_folds = []
    checkpoint_split_seed = None

    for path in checkpoint_paths:
        checkpoint = safe_load(path)

        if isinstance(checkpoint, Mapping):
            fold_value = checkpoint.get("fold")
            split_seed_value = checkpoint.get(
                "split_seed"
            )

            if fold_value is not None:
                checkpoint_folds.append(
                    int(fold_value)
                )

            if split_seed_value is not None:
                split_seed_value = int(
                    split_seed_value
                )

                if checkpoint_split_seed is None:
                    checkpoint_split_seed = (
                        split_seed_value
                    )
                elif (
                    split_seed_value
                    != checkpoint_split_seed
                ):
                    raise ValueError(
                        "checkpoint 使用了不同的 "
                        "患者级划分，不能集成。"
                    )

        current_frontend, current_duration = (
            checkpoint_config(checkpoint)
        )

        if front_config is None:
            front_config = current_frontend
            duration = current_duration
        elif (
            current_frontend != front_config
            or not math.isclose(
                current_duration,
                duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "所有 checkpoint 必须使用相同的"
                "音频前端配置。"
            )

        model = InnovativeResNet(
            num_classes=4
        ).to(device)

        model.load_state_dict(
            extract_state_dict(checkpoint),
            strict=True,
        )

        model.eval()
        models.append(model)

    if (
        checkpoint_folds
        and len(set(checkpoint_folds))
        != len(checkpoint_folds)
    ):
        raise ValueError(
            "发现重复的交叉验证 fold。"
        )

    if front_config is None or duration is None:
        raise RuntimeError(
            "没有成功加载模型。"
        )

    frontend = MelFrontend(
        front_config
    ).to(device).eval()

    test_dataset = ICBHIDataset(
        data_path=args.audio_dir,
        split="test",
        metadatafile=args.csv_path,
        duration=duration,
        samplerate=front_config.sample_rate,
        cache=not args.no_cache,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    labels, predictions = ensemble_predict(
        models,
        frontend,
        test_loader,
        device,
    )

    official = calc_icbhi_score(
        labels,
        predictions,
    )

    legacy = calc_legacy_macro_ovr_score(
        labels,
        predictions,
    )

    report_text = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print("=" * 72)
    print(f"Device               : {device}")
    print(f"Models               : {len(models)}")

    if checkpoint_folds:
        print(
            f"Folds                : "
            f"{sorted(checkpoint_folds)}"
        )

    print(
        f"Official ICBHI Score : "
        f"{official.score:.4f}"
    )
    print(
        f"Sensitivity          : "
        f"{official.sensitivity:.4f}"
    )
    print(
        f"Specificity          : "
        f"{official.specificity:.4f}"
    )
    print(
        f"Legacy Macro-OvR     : "
        f"{legacy['score']:.4f}"
    )
    print(report_text)
    print("Confusion matrix:")
    print(official.confusion_matrix)

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_confusion_matrix(
        official.confusion_matrix,
        output_dir
        / "ensemble_confusion_matrix.png",
    )

    write_json(
        output_dir / "metrics.json",
        {
            "checkpoints": [
                str(path)
                for path in checkpoint_paths
            ],
            "folds": sorted(
                checkpoint_folds
            ),
            "ensemble_size": len(models),
            "sample_count": len(labels),
            "official_icbhi": official.to_dict(),
            "legacy_macro_ovr": legacy,
            "classification_report": report_dict,
        },
    )

    print(
        f"结果已保存至: {output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DS-MSAN patient-level "
            "cross-validation pipeline"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train = subparsers.add_parser(
        "train",
        help="训练一个交叉验证 fold",
    )

    train.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
    )
    train.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
    )
    train.add_argument(
        "--output-dir",
        default="runs/dsmsan_cv_v2",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=24923,
    )
    train.add_argument(
        "--num-folds",
        type=int,
        default=5,
    )
    train.add_argument(
        "--fold",
        type=int,
        required=True,
    )
    train.add_argument(
        "--split-candidates",
        type=int,
        default=500,
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    train.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    train.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    train.add_argument(
        "--patience",
        type=int,
        default=15,
    )
    train.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    train.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )
    train.add_argument(
        "--label-smoothing",
        type=float,
        default=0.03,
    )
    train.add_argument(
        "--sampling-power",
        type=float,
        default=0.25,
    )
    train.add_argument(
        "--mixup-probability",
        type=float,
        default=0.15,
    )
    train.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.20,
    )
    train.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
    )
    train.add_argument(
        "--disable-waveform-augment",
        action="store_true",
    )
    train.add_argument(
        "--disable-amp",
        action="store_true",
    )
    train.add_argument(
        "--no-cache",
        action="store_true",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="在 official test 上评估折模型集成",
    )

    evaluate.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
    )
    evaluate.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
    )
    evaluate.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
    )
    evaluate.add_argument(
        "--output-dir",
        default="evaluation/dsmsan_cv_v2",
    )
    evaluate.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    evaluate.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    evaluate.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    evaluate.add_argument(
        "--no-cache",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        raise RuntimeError(
            f"未知命令: {args.command}"
        )


if __name__ == "__main__":
    main()