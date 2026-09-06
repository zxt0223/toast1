"""Evaluate a DS-MSAN checkpoint once on the official ICBHI test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASS_NAMES, ICBHIDataset
from frontend import FrontendConfig, MelFrontend
from metrics import calc_icbhi_score, calc_legacy_macro_ovr_score
from model import InnovativeResNet


REPOSITORY_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPOSITORY_DIR / (
    "InnovativeResNet_best_DS_MSAN_seed_24923_score_0.6439.pth"
)
DEFAULT_AUDIO_DIR = (
    "/group/chenjinming/zxt/gemini/data/icbhi_dataset/audio_test_data"
)
DEFAULT_CSV_PATH = (
    "/group/chenjinming/zxt/gemini/data/icbhi_dataset/metadata/metadata.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DS-MSAN on the official ICBHI test split."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-shift", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 CUDA 设备。")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint 必须是字典，当前为 {type(checkpoint)}")

    state_dict = None
    for key in (
        "selected_state_dict",
        "ema_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and candidate:
            state_dict = candidate
            break
    if state_dict is None:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(
                "无法把 checkpoint 解释为 state_dict；"
                f"键 {key!r} 的值不是 Tensor。"
            )
        cleaned[str(key).removeprefix("module.")] = value
    return cleaned


def checkpoint_configs(checkpoint) -> tuple[FrontendConfig, float]:
    if not isinstance(checkpoint, Mapping):
        return FrontendConfig(), 8.0

    stored_config = checkpoint.get("config")
    stored_frontend = checkpoint.get("frontend_config")
    if not isinstance(stored_config, Mapping):
        stored_config = {}
    if not isinstance(stored_frontend, Mapping):
        stored_frontend = stored_config

    frontend_config = FrontendConfig.from_mapping(stored_frontend)
    duration = float(stored_config.get("duration", 8.0))
    return frontend_config, duration


def safe_time_shift(spectrograms: torch.Tensor, shift_frames: int) -> torch.Tensor:
    """Shift along time and fill exposed frames with the spectrogram floor."""

    if shift_frames == 0:
        return spectrograms
    time_length = int(spectrograms.size(-1))
    floor = spectrograms.amin(dim=(-2, -1), keepdim=True)
    if abs(shift_frames) >= time_length:
        return floor.expand_as(spectrograms).clone()

    shifted = floor.expand_as(spectrograms).clone()
    if shift_frames > 0:
        shifted[..., shift_frames:] = spectrograms[..., :-shift_frames]
    else:
        shift = -shift_frames
        shifted[..., :-shift] = spectrograms[..., shift:]
    return shifted


@torch.inference_mode()
def predict(
    model: InnovativeResNet,
    frontend: MelFrontend,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
    tta_shift: int,
) -> tuple[list[int], list[int], list[int]]:
    model.eval()
    frontend.eval()
    labels_all: list[int] = []
    original_predictions: list[int] = []
    tta_predictions: list[int] = []

    for waveforms, labels, _, _ in tqdm(loader, desc="Official Test", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        spectrograms = frontend(waveforms, augment=False)
        original_probabilities = F.softmax(model(spectrograms), dim=1)
        original_predictions.extend(original_probabilities.argmax(dim=1).cpu().tolist())

        if use_tta:
            left_probabilities = F.softmax(
                model(safe_time_shift(spectrograms, -tta_shift)),
                dim=1,
            )
            right_probabilities = F.softmax(
                model(safe_time_shift(spectrograms, tta_shift)),
                dim=1,
            )
            probabilities = (
                0.60 * original_probabilities
                + 0.20 * left_probabilities
                + 0.20 * right_probabilities
            )
            tta_predictions.extend(probabilities.argmax(dim=1).cpu().tolist())

        labels_all.extend(labels.tolist())

    return labels_all, original_predictions, tta_predictions


def save_confusion_matrix(
    confusion: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion.astype(np.float64),
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals != 0,
    )
    annotations = np.empty_like(confusion, dtype=object)
    for row in range(4):
        for column in range(4):
            annotations[row, column] = (
                f"{normalized[row, column]:.1%}\n({confusion[row, column]})"
            )

    figure, axis = plt.subplots(figsize=(8, 6))
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
        cbar_kws={"label": "Row-normalized proportion"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def report(
    name: str,
    labels: list[int],
    predictions: list[int],
    output_dir: Path,
) -> dict:
    official = calc_icbhi_score(labels, predictions)
    legacy = calc_legacy_macro_ovr_score(labels, predictions)
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

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Official ICBHI Score : {official.score:.4f}")
    print(f"Sensitivity          : {official.sensitivity:.4f}")
    print(f"Specificity          : {official.specificity:.4f}")
    print(f"Legacy Macro-OvR     : {legacy['score']:.4f}")
    print(report_text)
    print("Confusion matrix:")
    print(official.confusion_matrix)

    safe_name = name.lower().replace(" ", "_")
    save_confusion_matrix(
        official.confusion_matrix,
        output_dir / f"{safe_name}_confusion_matrix.png",
        name,
    )
    return {
        "name": name,
        "official_icbhi": official.to_dict(),
        "legacy_macro_ovr": legacy,
        "classification_report": report_dict,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size 必须为正数，num_workers 不能为负数。")
    if args.tta_shift < 0:
        raise ValueError("tta_shift 不能为负数。")

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = safe_torch_load(checkpoint_path, device)
    frontend_config, duration = checkpoint_configs(checkpoint)

    model = InnovativeResNet(num_classes=4).to(device)
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint 与当前 DS-MSAN 结构不匹配；"
            "请确认没有使用已删除的实验模型。"
        ) from error
    model.eval()
    frontend = MelFrontend(frontend_config).to(device).eval()

    test_dataset = ICBHIDataset(
        data_path=args.audio_dir,
        split="test",
        metadatafile=args.csv_path,
        duration=duration,
        samplerate=frontend_config.sample_rate,
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

    print("=" * 72)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(f"Samples    : {len(test_dataset)}")
    print(f"TTA        : {args.tta}")
    print("=" * 72)
    labels, original_predictions, tta_predictions = predict(
        model,
        frontend,
        test_loader,
        device,
        use_tta=args.tta,
        tta_shift=args.tta_shift,
    )

    results = [
        report("Original", labels, original_predictions, output_dir)
    ]
    if args.tta:
        results.append(report("Conservative TTA", labels, tta_predictions, output_dir))

    payload = {
        "checkpoint": str(checkpoint_path),
        "sample_count": len(labels),
        "frontend_config": frontend_config.to_dict(),
        "duration": duration,
        "results": results,
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ 指标和混淆矩阵已保存到: {output_dir}")


if __name__ == "__main__":
    main()