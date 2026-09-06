from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ICBHIDataset
from model import HierarchicalRespiratoryNet
from train import (
    Config,
    DualResolutionFrontend,
    calc_icbhi_score,
    seed_everything,
    seed_worker,
    unpack_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 HATS-Net 层次化 ICBHI 模型")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="覆盖 checkpoint 保存的 validation 阈值；通常不要设置。",
    )
    parser.add_argument("--output-dir", default="evaluation/icbhi_hierarchical_v2")
    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def config_from_checkpoint(payload: Dict) -> Config:
    cfg = Config()
    valid_names = {field.name for field in fields(Config)}
    for name, value in payload.get("config", {}).items():
        if name in valid_names:
            setattr(cfg, name, value)
    return cfg


def load_models(paths: Sequence[str], device: torch.device):
    models: List[HierarchicalRespiratoryNet] = []
    thresholds: List[float] = []
    reference_cfg = None

    compatibility = [
        "sample_rate",
        "duration",
        "transient_n_fft",
        "transient_hop",
        "context_n_fft",
        "context_hop",
        "n_mels",
        "f_min",
        "f_max",
        "target_frames",
    ]

    for path in paths:
        payload = safe_torch_load(path, device)
        cfg = config_from_checkpoint(payload)
        if reference_cfg is None:
            reference_cfg = cfg
        else:
            for key in compatibility:
                if getattr(cfg, key) != getattr(reference_cfg, key):
                    raise ValueError(f"checkpoint 前端配置不一致: {key}")

        model = HierarchicalRespiratoryNet(input_channels=2).to(device)
        state = payload.get("selected_state_dict") or payload.get("model_state_dict")
        cleaned = {
            str(key).removeprefix("module."): value for key, value in state.items()
        }
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        models.append(model)
        thresholds.append(float(payload.get("selected_threshold", 0.5)))
        print(
            f"✅ 已加载: {path} | source={payload.get('selected_source', 'unknown')} | "
            f"threshold={thresholds[-1]:.2f}"
        )

    if reference_cfg is None:
        raise RuntimeError("没有加载任何 checkpoint。")
    return models, thresholds, reference_cfg


@torch.inference_mode()
def predict(models, frontend, loader, device, threshold: float):
    labels_all = []
    abnormal_sum = None
    subtype_sum = None

    for batch in loader:
        waveforms, labels, _, _ = unpack_batch(batch)
        waveforms = waveforms.to(device, non_blocking=True)
        features = frontend(waveforms, augment=False)

        batch_abnormal = torch.zeros(
            waveforms.size(0), device=device, dtype=torch.float32
        )
        batch_subtype = torch.zeros(
            waveforms.size(0), 3, device=device, dtype=torch.float32
        )
        for model in models:
            abnormal_logit, subtype_logits, _ = model(features)
            batch_abnormal += torch.sigmoid(abnormal_logit.float())
            batch_subtype += torch.softmax(subtype_logits.float(), dim=1)

        batch_abnormal /= len(models)
        batch_subtype /= len(models)

        labels_all.append(labels.cpu())
        if abnormal_sum is None:
            abnormal_sum = [batch_abnormal.cpu()]
            subtype_sum = [batch_subtype.cpu()]
        else:
            abnormal_sum.append(batch_abnormal.cpu())
            subtype_sum.append(batch_subtype.cpu())

    labels = torch.cat(labels_all).numpy()
    abnormal_probs = torch.cat(abnormal_sum).numpy()
    subtype_probs = torch.cat(subtype_sum).numpy()
    predictions = np.where(
        abnormal_probs >= threshold,
        subtype_probs.argmax(axis=1) + 1,
        0,
    )
    return labels, predictions


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("HATS-Net Confusion Matrix")

    for row in range(4):
        for col in range(4):
            ax.text(
                col,
                row,
                f"{normalized[row, col]:.1%}\n({cm[row, col]})",
                ha="center",
                va="center",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, stored_thresholds, cfg = load_models(args.checkpoints, device)
    seed_everything(cfg.seed)

    if args.threshold is None:
        threshold = float(np.mean(stored_thresholds))
    else:
        threshold = float(args.threshold)

    frontend = DualResolutionFrontend(cfg).to(device).eval()
    test_dataset = ICBHIDataset(
        data_path=cfg.audio_dir,
        split="test",
        metadatafile=cfg.csv_path,
        duration=cfg.duration,
        samplerate=cfg.sample_rate,
        training=False,
        cache=True,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    labels, predictions = predict(models, frontend, loader, device, threshold)
    score, se, sp, cm, recalls = calc_icbhi_score(labels, predictions)

    print("=" * 72)
    print(f"HATS-Net | Models={len(models)} | Threshold={threshold:.2f}")
    print(f"ICBHI Score : {score:.4f}")
    print(f"Sensitivity : {se:.4f}")
    print(f"Specificity : {sp:.4f}")
    print(cm)
    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    result = {
        "num_models": len(models),
        "threshold": threshold,
        "score": float(score),
        "sensitivity": float(se),
        "specificity": float(sp),
        "confusion_matrix": cm.tolist(),
        "class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, recalls)
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    print(f"✅ 结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()