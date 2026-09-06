"""Dual-factor QLung-AST pipeline for ICBHI 2017.

This V3 experiment keeps the validated V2 cache and backbone recipe, while
adding two explicit symptom tasks: crackle present/absent and wheeze
present/absent.  Four-class QLung predictions and the factorized predictions
are fused using parameters selected only from patient-disjoint OOF outputs.

Commands:
    check      validate the model heads and factor composition;
    tune       train one patient-disjoint fold and save every epoch's logits;
    summarize  choose a common epoch, fusion weight, and Normal log-bias;
    train      train one final full-official-train seed;
    evaluate   evaluate a frozen final ensemble once, without TTA.

The script depends on qlung_pipeline_v2.py, dataset.py, and metrics.py.  It
does not overwrite V2 checkpoints, logs, summaries, or evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import ASTConfig, ASTModel

import qlung_pipeline_v2 as v2
from dataset import CLASS_NAMES, safe_torch_load
from metrics import calc_icbhi_score, calc_legacy_macro_ovr_score


METHOD_NAME = "QLung-AST-DualFactor-V3"
FORMAT_VERSION = 1
INPUT_LENGTH = 798
MODEL_DIR = "pretrained/ast_audioset"
TRAIN_CACHE = "precomputed/qlung_ast_v2/ast_train_len798.pth"
TEST_CACHE = "precomputed/qlung_ast_v2/ast_test_len798.pth"
TUNE_DIR = "runs/qlung_factor_v3_tune"
TUNING_SUMMARY = f"{TUNE_DIR}/tuning_summary.json"
FINAL_DIR = "runs/qlung_factor_v3_final"

FACTOR_NAMES = ("Crackle", "Wheeze")
FACTOR_LAMBDA = 0.50
CONSISTENCY_LAMBDA = 0.20
FACTOR_DROPOUT = 0.10


def factor_targets(labels: torch.Tensor) -> torch.Tensor:
    """Convert 4-class labels into [crackle, wheeze] binary targets."""

    lookup = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
        device=labels.device,
    )
    return lookup[labels.long()]


def factor_class_probabilities(factor_logits: torch.Tensor) -> torch.Tensor:
    """Compose two Bernoulli symptom probabilities into four classes."""

    probability = factor_logits.float().sigmoid()
    crackle = probability[:, 0]
    wheeze = probability[:, 1]
    return torch.stack(
        [
            (1.0 - crackle) * (1.0 - wheeze),
            crackle * (1.0 - wheeze),
            (1.0 - crackle) * wheeze,
            crackle * wheeze,
        ],
        dim=1,
    )


def fused_probabilities(
    class_logits: torch.Tensor,
    factor_logits: torch.Tensor,
    class_weight: float,
) -> torch.Tensor:
    if not 0.0 <= class_weight <= 1.0:
        raise ValueError("class_weight 必须位于 [0, 1]。")
    class_probability = class_logits.float().softmax(dim=1)
    factor_probability = factor_class_probabilities(factor_logits)
    return (
        float(class_weight) * class_probability
        + (1.0 - float(class_weight)) * factor_probability
    )


def predictions_with_normal_log_bias(
    probabilities: torch.Tensor,
    normal_log_bias: float,
) -> torch.Tensor:
    adjusted = probabilities.float().clamp_min(1e-12).log()
    adjusted[:, 0] += float(normal_log_bias)
    return adjusted.argmax(dim=1)


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sum = matrix.sum(axis=1, keepdims=True)
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
        title="QLung AST Dual-Factor V3 Official Test (No TTA)",
        xlabel="Predicted",
        ylabel="True",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


class DualFactorAST(nn.Module):
    def __init__(
        self,
        model_dir: Union[str, os.PathLike],
        input_length: int = INPUT_LENGTH,
        factor_dropout: float = FACTOR_DROPOUT,
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
            v2.resize_pretrained_position_embeddings(self.ast, input_length)
        else:
            config = ASTConfig.from_pretrained(model_dir, local_files_only=True)
            config.max_length = int(input_length)
            self.ast = ASTModel(config)

        hidden_size = int(self.ast.config.hidden_size)
        self.class_head = v2.AngularClassifier(hidden_size)
        self.factor_norm = nn.LayerNorm(hidden_size)
        self.factor_dropout = nn.Dropout(float(factor_dropout))
        self.factor_head = nn.Linear(hidden_size, 2)
        nn.init.trunc_normal_(self.factor_head.weight, std=0.02)
        nn.init.zeros_(self.factor_head.bias)

        frequency_patches, time_patches = v2.patch_grid(
            self.ast.config,
            input_length,
        )
        expected_tokens = 2 + frequency_patches * time_patches
        actual_tokens = int(self.ast.embeddings.position_embeddings.size(1))
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"V3 position tokens={actual_tokens}, expected={expected_tokens}"
            )

    def forward(self, input_values: torch.Tensor):
        features = self.ast(
            input_values=input_values,
            return_dict=True,
        ).pooler_output
        class_logits, class_cosine = self.class_head(features)
        factor_logits = self.factor_head(
            self.factor_dropout(self.factor_norm(features))
        )
        return class_logits, class_cosine, factor_logits


class DualFactorLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor,
        factor_lambda: float,
        consistency_lambda: float,
    ) -> None:
        super().__init__()
        counts = class_counts.detach().float()
        if counts.ndim != 1 or len(counts) != 4 or bool((counts <= 0).any()):
            raise ValueError(f"class_counts 无效: {counts.tolist()}")
        self.qlung = v2.QLungLoss(counts)
        crackle_positive = counts[1] + counts[3]
        crackle_negative = counts[0] + counts[2]
        wheeze_positive = counts[2] + counts[3]
        wheeze_negative = counts[0] + counts[1]
        pos_weight = torch.stack(
            [
                crackle_negative / crackle_positive,
                wheeze_negative / wheeze_positive,
            ]
        )
        self.register_buffer("factor_pos_weight", pos_weight)
        self.factor_lambda = float(factor_lambda)
        self.consistency_lambda = float(consistency_lambda)

    def forward(
        self,
        class_logits: torch.Tensor,
        class_cosine: torch.Tensor,
        factor_logits: torch.Tensor,
        labels: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict:
        qlung_total, ce_loss, dfam_loss, margin = self.qlung(
            class_logits,
            class_cosine,
            labels,
            quality,
        )
        targets = factor_targets(labels)
        factor_loss = F.binary_cross_entropy_with_logits(
            factor_logits.float(),
            targets,
            pos_weight=self.factor_pos_weight,
        )
        class_probability = class_logits.float().softmax(dim=1)
        class_marginals = torch.stack(
            [
                class_probability[:, 1] + class_probability[:, 3],
                class_probability[:, 2] + class_probability[:, 3],
            ],
            dim=1,
        )
        consistency_loss = F.mse_loss(
            factor_logits.float().sigmoid(),
            class_marginals,
        )
        total = (
            qlung_total
            + self.factor_lambda * factor_loss
            + self.consistency_lambda * consistency_loss
        )
        return {
            "total": total,
            "ce": ce_loss,
            "dfam": dfam_loss,
            "factor": factor_loss,
            "consistency": consistency_loss,
            "margin": margin,
        }


def train_epoch(
    model: DualFactorAST,
    criterion: DualFactorLoss,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    description: str,
) -> dict:
    model.train()
    names = ("total", "ce", "dfam", "factor", "consistency", "margin")
    totals = {name: 0.0 for name in names}
    sample_count = 0
    skipped = 0
    progress = tqdm(loader, desc=description, leave=False)

    for features, labels, quality in progress:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        if device.type == "cpu":
            features = features.float()
        features = v2.spec_augment(features)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            class_logits, class_cosine, factor_logits = model(features)
            losses = criterion(
                class_logits,
                class_cosine,
                factor_logits,
                labels,
                quality,
            )
        loss = losses["total"]
        if not torch.isfinite(loss):
            skipped += 1
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current = int(labels.size(0))
        sample_count += current
        for name in names:
            totals[name] += float(losses[name].detach()) * current
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            skipped=skipped,
        )

    if sample_count == 0:
        raise RuntimeError("V3 epoch 没有成功更新任何 batch。")
    result = {name: totals[name] / sample_count for name in names}
    result["skipped"] = int(skipped)
    return result


@torch.inference_mode()
def validation_outputs(
    model: DualFactorAST,
    loader,
    device: torch.device,
    amp_enabled: bool,
):
    model.eval()
    class_logits_all = []
    factor_logits_all = []
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
            class_logits, _, factor_logits = model(features)
        class_logits_all.append(class_logits.float().cpu())
        factor_logits_all.append(factor_logits.float().cpu())
        labels_all.append(labels.long().cpu())
    return (
        torch.cat(class_logits_all),
        torch.cat(factor_logits_all),
        torch.cat(labels_all),
    )


def score_predictions(labels: torch.Tensor, predictions: torch.Tensor):
    return calc_icbhi_score(
        labels.long().tolist(),
        predictions.long().tolist(),
    )


def tune_postprocess(
    class_logits: torch.Tensor,
    factor_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha_steps: int,
    bias_min: float,
    bias_max: float,
    bias_steps: int,
) -> dict:
    """Tune class/factor fusion and one Normal log-probability bias."""

    if alpha_steps < 2:
        raise ValueError("alpha_steps 必须至少为 2。")
    if bias_steps < 2 or bias_max <= bias_min:
        raise ValueError("Normal bias 搜索范围无效。")

    labels = labels.long().cpu()
    normal_indices = labels == 0
    abnormal_indices = labels != 0
    if not bool(normal_indices.any()) or not bool(abnormal_indices.any()):
        raise ValueError("调参标签必须同时包含 Normal 和异常样本。")

    class_probability = class_logits.float().cpu().softmax(dim=1)
    factor_probability = factor_class_probabilities(factor_logits.cpu())
    alphas = torch.linspace(0.0, 1.0, alpha_steps)
    biases = torch.linspace(float(bias_min), float(bias_max), bias_steps)
    best = None

    for alpha_tensor in alphas:
        alpha = float(alpha_tensor)
        probability = (
            alpha * class_probability
            + (1.0 - alpha) * factor_probability
        ).clamp_min(1e-12)
        log_probability = probability.log()
        abnormal_value, abnormal_offset = log_probability[:, 1:].max(dim=1)
        abnormal_prediction = abnormal_offset + 1
        delta = abnormal_value - log_probability[:, 0]
        normal_prediction = biases[:, None] >= delta[None, :]

        specificity = normal_prediction[:, normal_indices].float().mean(dim=1)
        abnormal_correct = abnormal_prediction[abnormal_indices] == labels[
            abnormal_indices
        ]
        sensitivity = (
            (~normal_prediction[:, abnormal_indices])
            & abnormal_correct[None, :]
        ).float().mean(dim=1)
        scores = (specificity + sensitivity) / 2.0
        maximum = float(scores.max())
        candidates = torch.nonzero(
            torch.isclose(scores, scores.max(), atol=1e-12, rtol=0.0),
            as_tuple=False,
        ).flatten()
        for index_tensor in candidates:
            index = int(index_tensor)
            bias = float(biases[index])
            candidate = {
                "score": maximum,
                "sensitivity": float(sensitivity[index]),
                "specificity": float(specificity[index]),
                "class_weight": alpha,
                "normal_log_bias": bias,
            }
            if best is None:
                best = candidate
                continue
            key = (
                candidate["score"],
                -abs(candidate["normal_log_bias"]),
                -abs(candidate["class_weight"] - 0.5),
            )
            best_key = (
                best["score"],
                -abs(best["normal_log_bias"]),
                -abs(best["class_weight"] - 0.5),
            )
            if key > best_key:
                best = candidate

    probability = (
        best["class_weight"] * class_probability
        + (1.0 - best["class_weight"]) * factor_probability
    )
    prediction = predictions_with_normal_log_bias(
        probability,
        best["normal_log_bias"],
    )
    result = score_predictions(labels, prediction)
    best["metrics"] = result.to_dict()
    return best


def select_smoothed_epoch(
    epoch_results: Sequence[dict],
    smoothing_window: int,
    minimum_epoch: int,
) -> tuple[int, list[dict]]:
    if not epoch_results:
        raise ValueError("epoch_results 不能为空。")
    if smoothing_window <= 0 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window 必须是正奇数。")
    if minimum_epoch <= 0:
        raise ValueError("minimum_epoch 必须为正整数。")
    radius = smoothing_window // 2
    count = len(epoch_results)
    enriched = []
    for index, item in enumerate(epoch_results):
        start = max(0, index - radius)
        stop = min(count, index + radius + 1)
        smoothed = float(
            np.median(
                [epoch_results[pos]["score"] for pos in range(start, stop)]
            )
        )
        record = dict(item)
        record["smoothed_score"] = smoothed
        enriched.append(record)
    eligible = [
        item
        for item in enriched
        if item["epoch"] >= max(minimum_epoch, radius + 1)
        and item["epoch"] <= count - radius
    ]
    if not eligible:
        eligible = [item for item in enriched if item["epoch"] >= minimum_epoch]
    if not eligible:
        raise ValueError("没有满足 minimum_epoch 的候选轮次。")
    selected = max(
        eligible,
        key=lambda item: (
            item["smoothed_score"],
            item["score"],
            -item["epoch"],
        ),
    )
    return int(selected["epoch"]), enriched


def run_check(args) -> None:
    model = DualFactorAST(
        args.model_dir,
        args.input_length,
        args.factor_dropout,
        pretrained=True,
    )
    frequency, time = v2.patch_grid(model.ast.config, args.input_length)
    labels = torch.tensor([0, 1, 2, 3])
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    if not torch.equal(factor_targets(labels).cpu(), expected):
        raise RuntimeError("V3 factor target mapping 自检失败。")
    probability = factor_class_probabilities(torch.zeros(4, 2))
    if not torch.allclose(probability.sum(dim=1), torch.ones(4)):
        raise RuntimeError("V3 factor probability 自检失败。")
    print("=" * 76)
    print("V3 self-check 通过")
    print(
        f"AST grid={frequency}x{time} | "
        f"tokens={model.ast.embeddings.position_embeddings.size(1)}"
    )
    print("labels: Normal=[0,0], Crackle=[1,0], Wheeze=[0,1], Both=[1,1]")
    print("=" * 76)


def run_tune(args) -> None:
    if not 0 <= args.fold < args.num_folds:
        raise ValueError("fold 必须位于 [0, num_folds)。")
    fold_seed = args.seed + args.fold * 1009
    v2.seed_all(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output = Path(args.output_dir) / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)

    cache = v2.FeatureCache(args.train_cache, "train")
    if cache.input_length != args.input_length:
        raise ValueError("V3 train cache 与 input_length 不一致。")
    folds, split_seed, objective = v2.search_patient_folds(
        cache,
        args.num_folds,
        args.seed,
        args.split_candidates,
    )
    train_indices, validation_indices = folds[args.fold]
    calibrated_quality, calibration = v2.calibrate_quality(
        cache.raw_quality,
        train_indices,
        args.aqs_target_mean,
        args.aqs_target_std,
    )
    train_set = v2.FeatureView(cache, train_indices, calibrated_quality)
    validation_set = v2.FeatureView(
        cache,
        validation_indices,
        calibrated_quality,
    )
    if set(train_set.patient_ids).intersection(validation_set.patient_ids):
        raise RuntimeError("V3 tune 训练与验证患者重叠。")

    train_loader = v2.make_loader(
        train_set,
        args.batch_size,
        args.num_workers,
        True,
        device,
        fold_seed + 1,
    )
    validation_loader = v2.make_loader(
        validation_set,
        args.batch_size,
        args.num_workers,
        False,
        device,
        fold_seed + 2,
    )
    model = DualFactorAST(
        args.model_dir,
        cache.input_length,
        args.factor_dropout,
        pretrained=True,
    ).to(device)
    class_counts = torch.bincount(train_set.labels, minlength=4)
    criterion = DualFactorLoss(
        class_counts,
        args.factor_lambda,
        args.consistency_lambda,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = v2.scaler_for(amp_enabled)

    print("=" * 76)
    print(
        f"V3 tune fold={args.fold}/{args.num_folds - 1} | "
        f"seed={fold_seed} | device={device} | AMP={amp_enabled}"
    )
    print(f"split_seed={split_seed} | objective={objective:.4f}")
    print(
        f"train={torch.bincount(train_set.labels, minlength=4).tolist()} | "
        f"validation={torch.bincount(validation_set.labels, minlength=4).tolist()}"
    )
    print(
        f"factor pos_weight="
        f"{criterion.factor_pos_weight.detach().cpu().tolist()}"
    )
    print(f"AQS calibration={calibration}")
    print("official test 不参与 V3 训练、epoch 或融合参数选择。")
    print("=" * 76)

    history = []
    class_epochs = []
    factor_epochs = []
    reference_labels = None
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            f"V3 fold {args.fold} epoch {epoch}/{args.epochs}",
        )
        class_logits, factor_logits, labels = validation_outputs(
            model,
            validation_loader,
            device,
            amp_enabled,
        )
        if reference_labels is None:
            reference_labels = labels.clone()
        elif not torch.equal(reference_labels, labels):
            raise RuntimeError("V3 validation label 顺序发生变化。")
        class_epochs.append(class_logits)
        factor_epochs.append(factor_logits)

        class_result = score_predictions(labels, class_logits.argmax(dim=1))
        factor_prediction = factor_class_probabilities(factor_logits).argmax(dim=1)
        factor_result = score_predictions(labels, factor_prediction)
        fused_prediction = fused_probabilities(
            class_logits,
            factor_logits,
            0.5,
        ).argmax(dim=1)
        fused_result = score_predictions(labels, fused_prediction)
        record = {
            "epoch": epoch,
            "training": training,
            "class_only": class_result.to_dict(),
            "factor_only": factor_result.to_dict(),
            "fixed_half_fusion": fused_result.to_dict(),
        }
        history.append(record)
        v2.write_json(output / "history.json", history)
        print(
            f"fold={args.fold} epoch={epoch:02d} "
            f"loss={training['total']:.4f} "
            f"factor={training['factor']:.4f} "
            f"cons={training['consistency']:.4f} | "
            f"class={class_result.score:.4f} "
            f"factor={factor_result.score:.4f} "
            f"fused50={fused_result.score:.4f}"
        )

    destination = output / "epoch_outputs.pth"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "method": f"{METHOD_NAME}-tune",
            "fold": int(args.fold),
            "num_folds": int(args.num_folds),
            "split_seed": int(split_seed),
            "validation_indices": validation_indices.tolist(),
            "validation_labels": reference_labels,
            "class_logits": torch.stack(class_epochs),
            "factor_logits": torch.stack(factor_epochs),
            "epochs": int(args.epochs),
            "input_length": int(cache.input_length),
            "factor_lambda": float(args.factor_lambda),
            "consistency_lambda": float(args.consistency_lambda),
            "factor_dropout": float(args.factor_dropout),
            "learning_rate": float(args.learning_rate),
            "aqs_target_mean": float(args.aqs_target_mean),
            "aqs_target_std": float(args.aqs_target_std),
            "aqs_calibration": calibration,
            "official_test_used": False,
        },
        destination,
    )
    print("=" * 76)
    print(f"V3 fold {args.fold} 完成: {destination}")
    print("=" * 76)


def run_summarize(args) -> None:
    tune_dir = Path(args.tune_dir).expanduser().resolve()
    payloads = []
    split_seed = None
    validation_indices = []
    reference_config = None

    for fold in range(args.num_folds):
        path = tune_dir / f"fold_{fold}" / "epoch_outputs.pth"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 V3 fold 输出: {path}")
        payload = safe_torch_load(path)
        if (
            int(payload.get("format_version", -1)) != FORMAT_VERSION
            or payload.get("method") != f"{METHOD_NAME}-tune"
            or int(payload.get("fold", -1)) != fold
            or int(payload.get("num_folds", -1)) != args.num_folds
        ):
            raise ValueError(f"V3 fold 输出不兼容: {path}")
        labels = payload.get("validation_labels")
        class_logits = payload.get("class_logits")
        factor_logits = payload.get("factor_logits")
        if not all(
            torch.is_tensor(value)
            for value in (labels, class_logits, factor_logits)
        ):
            raise TypeError(f"V3 fold 输出缺少 Tensor: {path}")
        expected_samples = len(payload["validation_indices"])
        expected_epochs = int(payload["epochs"])
        if tuple(class_logits.shape) != (expected_epochs, expected_samples, 4):
            raise ValueError(f"V3 class_logits 形状错误: {path}")
        if tuple(factor_logits.shape) != (expected_epochs, expected_samples, 2):
            raise ValueError(f"V3 factor_logits 形状错误: {path}")
        if tuple(labels.shape) != (expected_samples,):
            raise ValueError(f"V3 validation_labels 形状错误: {path}")
        current_seed = int(payload["split_seed"])
        if split_seed is None:
            split_seed = current_seed
        elif current_seed != split_seed:
            raise ValueError("V3 各折 split_seed 不一致。")
        config = (
            int(payload["epochs"]),
            int(payload["input_length"]),
            float(payload["factor_lambda"]),
            float(payload["consistency_lambda"]),
            float(payload["factor_dropout"]),
            float(payload["learning_rate"]),
            float(payload["aqs_target_mean"]),
            float(payload["aqs_target_std"]),
        )
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError("V3 各折训练配置不一致。")
        validation_indices.extend(int(x) for x in payload["validation_indices"])
        payloads.append(payload)

    if len(validation_indices) != len(set(validation_indices)):
        raise RuntimeError("V3 OOF validation indices 存在重复。")
    if sorted(validation_indices) != list(range(len(validation_indices))):
        raise RuntimeError("V3 OOF validation indices 未完整覆盖训练集。")

    epoch_count = reference_config[0]
    epoch_results = []
    for epoch_index in range(epoch_count):
        class_logits = torch.cat(
            [payload["class_logits"][epoch_index].float() for payload in payloads]
        )
        factor_logits = torch.cat(
            [payload["factor_logits"][epoch_index].float() for payload in payloads]
        )
        labels = torch.cat(
            [payload["validation_labels"].long() for payload in payloads]
        )
        tuned = tune_postprocess(
            class_logits,
            factor_logits,
            labels,
            args.alpha_steps,
            args.bias_min,
            args.bias_max,
            args.bias_steps,
        )
        epoch_results.append(
            {
                "epoch": epoch_index + 1,
                "score": float(tuned["score"]),
                "sensitivity": float(tuned["sensitivity"]),
                "specificity": float(tuned["specificity"]),
                "class_weight": float(tuned["class_weight"]),
                "normal_log_bias": float(tuned["normal_log_bias"]),
                "metrics": tuned["metrics"],
            }
        )

    selected_epoch, curve = select_smoothed_epoch(
        epoch_results,
        args.smoothing_window,
        args.minimum_epoch,
    )
    selected = curve[selected_epoch - 1]
    summary = {
        "format_version": FORMAT_VERSION,
        "method": f"{METHOD_NAME}-OOF-tuning",
        "split_seed": int(split_seed),
        "num_folds": int(args.num_folds),
        "selected_final_epoch": int(selected_epoch),
        "selected_class_weight": float(selected["class_weight"]),
        "selected_factor_weight": float(1.0 - selected["class_weight"]),
        "selected_normal_log_bias": float(selected["normal_log_bias"]),
        "selected_oof_tuning_metrics": selected["metrics"],
        "selection_smoothed_score": float(selected["smoothed_score"]),
        "smoothing_window": int(args.smoothing_window),
        "smoothing_statistic": "median",
        "minimum_epoch": int(args.minimum_epoch),
        "input_length": int(reference_config[1]),
        "factor_lambda": float(reference_config[2]),
        "consistency_lambda": float(reference_config[3]),
        "factor_dropout": float(reference_config[4]),
        "learning_rate": float(reference_config[5]),
        "aqs_target_mean": float(reference_config[6]),
        "aqs_target_std": float(reference_config[7]),
        "epoch_curve": curve,
        "official_test_used": False,
        "warning": "OOF tuning score selects hyperparameters; it is not an unbiased test estimate.",
    }
    destination = tune_dir / "tuning_summary.json"
    v2.write_json(destination, summary)

    print("=" * 76)
    print("V3 common-epoch OOF Top 10 (按平滑选择分数)")
    top_candidates = [
        item
        for item in curve
        if item["epoch"] >= max(args.minimum_epoch, args.smoothing_window // 2 + 1)
        and item["epoch"] <= len(curve) - args.smoothing_window // 2
    ]
    if not top_candidates:
        top_candidates = curve
    top = sorted(
        top_candidates,
        key=lambda item: (item["smoothed_score"], item["score"]),
        reverse=True,
    )[:10]
    for item in top:
        print(
            f"epoch={item['epoch']:02d} | score={item['score']:.4f} | "
            f"smooth={item['smoothed_score']:.4f} | "
            f"SE={item['sensitivity']:.4f} SP={item['specificity']:.4f} | "
            f"class_w={item['class_weight']:.3f} "
            f"bias={item['normal_log_bias']:+.3f}"
        )
    print("-" * 76)
    print(
        f"selected epoch={selected_epoch} | "
        f"class_weight={selected['class_weight']:.3f} | "
        f"factor_weight={1.0 - selected['class_weight']:.3f} | "
        f"normal_log_bias={selected['normal_log_bias']:+.3f}"
    )
    print(
        f"OOF tuning Score={selected['score']:.4f} | "
        f"SE={selected['sensitivity']:.4f} | "
        f"SP={selected['specificity']:.4f}"
    )
    print(f"保存: {destination}")
    print("=" * 76)


def load_summary(path: Union[str, os.PathLike]) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 V3 tuning summary: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if (
        int(data.get("format_version", -1)) != FORMAT_VERSION
        or data.get("method") != f"{METHOD_NAME}-OOF-tuning"
        or data.get("official_test_used") is not False
    ):
        raise ValueError("V3 tuning summary 不兼容。")
    if int(data.get("selected_final_epoch", 0)) <= 0:
        raise ValueError("V3 selected_final_epoch 无效。")
    class_weight = float(data.get("selected_class_weight", math.nan))
    normal_log_bias = float(data.get("selected_normal_log_bias", math.nan))
    if not 0.0 <= class_weight <= 1.0:
        raise ValueError("V3 selected_class_weight 无效。")
    if not math.isfinite(normal_log_bias):
        raise ValueError("V3 selected_normal_log_bias 无效。")
    return data


def run_train(args) -> None:
    summary = load_summary(args.tuning_summary)
    selected_epoch = int(summary["selected_final_epoch"])
    epochs = selected_epoch if args.epochs is None else int(args.epochs)
    if args.epochs is not None and epochs != selected_epoch:
        print(f"警告: 手动 epochs={epochs}，OOF 选择为 {selected_epoch}。")

    v2.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output = Path(args.output_dir) / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    cache = v2.FeatureCache(args.train_cache, "train")
    if cache.input_length != int(summary["input_length"]):
        raise ValueError("V3 final train cache 与 tuning summary 不一致。")
    all_indices = np.arange(len(cache), dtype=np.int64)
    calibrated_quality, calibration = v2.calibrate_quality(
        cache.raw_quality,
        all_indices,
        float(summary["aqs_target_mean"]),
        float(summary["aqs_target_std"]),
    )
    dataset = v2.FeatureView(cache, all_indices, calibrated_quality)
    loader = v2.make_loader(
        dataset,
        args.batch_size,
        args.num_workers,
        True,
        device,
        args.seed + 1,
    )
    model = DualFactorAST(
        args.model_dir,
        cache.input_length,
        float(summary["factor_dropout"]),
        pretrained=True,
    ).to(device)
    class_counts = torch.bincount(dataset.labels, minlength=4)
    criterion = DualFactorLoss(
        class_counts,
        float(summary["factor_lambda"]),
        float(summary["consistency_lambda"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(summary["learning_rate"]),
    )
    scaler = v2.scaler_for(amp_enabled)
    history = []

    print("=" * 76)
    print(
        f"V3 final train | seed={args.seed} | epochs={epochs} | "
        f"device={device} | AMP={amp_enabled}"
    )
    print(f"AQS calibration={calibration}")
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
            f"V3 seed {args.seed} epoch {epoch}/{epochs}",
        )
        training["epoch"] = epoch
        history.append(training)
        v2.write_json(output / "history.json", history)
        print(
            f"seed={args.seed} epoch={epoch:02d} "
            f"loss={training['total']:.4f} ce={training['ce']:.4f} "
            f"dfam={training['dfam']:.4f} "
            f"factor={training['factor']:.4f} "
            f"cons={training['consistency']:.4f}"
        )

    checkpoint = output / "final_checkpoint.pth"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "method": f"{METHOD_NAME}-final",
            "model_state_dict": v2.clone_cpu_state(model),
            "seed": int(args.seed),
            "epoch": int(epochs),
            "input_length": int(cache.input_length),
            "factor_dropout": float(summary["factor_dropout"]),
            "class_weight": float(summary["selected_class_weight"]),
            "normal_log_bias": float(summary["selected_normal_log_bias"]),
            "tuning_summary": str(
                Path(args.tuning_summary).expanduser().resolve()
            ),
            "history": history,
            "tta": False,
        },
        checkpoint,
    )
    print(f"V3 final 训练完成: {checkpoint}")


def load_final_model(path: Path, args, device: torch.device):
    checkpoint = safe_torch_load(path)
    if (
        int(checkpoint.get("format_version", -1)) != FORMAT_VERSION
        or checkpoint.get("method") != f"{METHOD_NAME}-final"
    ):
        raise ValueError(f"不是 V3 final checkpoint: {path}")
    model = DualFactorAST(
        args.model_dir,
        int(checkpoint["input_length"]),
        float(checkpoint["factor_dropout"]),
        pretrained=False,
    ).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"V3 checkpoint 缺少 model_state_dict: {path}")
    model.load_state_dict(
        {str(name).removeprefix("module."): value for name, value in state.items()},
        strict=True,
    )
    return (
        model.eval(),
        int(checkpoint["seed"]),
        int(checkpoint["input_length"]),
        float(checkpoint["class_weight"]),
        float(checkpoint["normal_log_bias"]),
    )


@torch.inference_mode()
def run_evaluate(args) -> None:
    summary = load_summary(args.tuning_summary)
    class_weight = float(summary["selected_class_weight"])
    normal_log_bias = float(summary["selected_normal_log_bias"])
    paths = [Path(value).expanduser().resolve() for value in args.checkpoint]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"V3 final checkpoint 不完整: {paths}")
    if len(paths) != len(set(paths)):
        raise ValueError("V3 evaluate 收到了重复 checkpoint。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    cache = v2.FeatureCache(args.test_cache, "test")
    loader = v2.make_loader(
        cache,
        args.batch_size,
        args.num_workers,
        False,
        device,
        0,
    )
    loaded = [load_final_model(path, args, device) for path in paths]
    seeds = [item[1] for item in loaded]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"V3 evaluate 收到了重复 seed: {seeds}")
    for _, _, length, weight, bias in loaded:
        if length != cache.input_length:
            raise ValueError("V3 checkpoint 与 test cache 长度不一致。")
        if not math.isclose(weight, class_weight, abs_tol=1e-9):
            raise ValueError("V3 checkpoint 与 summary 的融合权重不一致。")
        if not math.isclose(bias, normal_log_bias, abs_tol=1e-9):
            raise ValueError("V3 checkpoint 与 summary 的 bias 不一致。")
    models = [item[0] for item in loaded]

    labels_all = []
    ensemble_predictions = []
    individual_predictions = [[] for _ in models]
    for features, labels, _ in tqdm(loader, desc="V3 official test (no TTA)"):
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
                class_logits, _, factor_logits = model(features)
                probability = fused_probabilities(
                    class_logits.float(),
                    factor_logits.float(),
                    class_weight,
                )
                prediction = predictions_with_normal_log_bias(
                    probability,
                    normal_log_bias,
                )
                individual_predictions[index].extend(
                    prediction.cpu().tolist()
                )
                probability_sum = (
                    probability
                    if probability_sum is None
                    else probability_sum + probability
                )
        ensemble_prediction = predictions_with_normal_log_bias(
            probability_sum,
            normal_log_bias,
        )
        ensemble_predictions.extend(ensemble_prediction.cpu().tolist())
        labels_all.extend(labels.tolist())

    individuals = []
    for seed, prediction in zip(seeds, individual_predictions):
        result = calc_icbhi_score(labels_all, prediction)
        individuals.append({"seed": seed, "official_icbhi": result.to_dict()})
        print(
            f"V3 seed={seed} score={result.score:.4f} "
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
    print(f"V3 models={len(models)} seeds={seeds} TTA=False")
    print(
        f"class_weight={class_weight:.3f} | "
        f"factor_weight={1.0 - class_weight:.3f} | "
        f"normal_log_bias={normal_log_bias:+.3f}"
    )
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
    v2.write_json(
        output / "metrics.json",
        {
            "method": f"{METHOD_NAME} probability ensemble",
            "checkpoints": [str(path) for path in paths],
            "seeds": seeds,
            "input_length": int(cache.input_length),
            "class_weight": class_weight,
            "factor_weight": 1.0 - class_weight,
            "normal_log_bias": normal_log_bias,
            "parameters_selected_on": "patient-disjoint OOF official-train only",
            "tta": False,
            "official_icbhi": result.to_dict(),
            "legacy_macro_ovr": legacy,
            "individual_models": individuals,
        },
    )
    print(f"V3 结果保存到: {output}")


def add_model_arguments(parser) -> None:
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--input-length", type=int, default=INPUT_LENGTH)
    parser.add_argument("--factor-dropout", type=float, default=FACTOR_DROPOUT)


def add_training_arguments(parser) -> None:
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(description="QLung-AST Dual-Factor V3")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    add_model_arguments(check)

    tune = commands.add_parser("tune")
    add_model_arguments(tune)
    add_training_arguments(tune)
    tune.add_argument("--train-cache", default=TRAIN_CACHE)
    tune.add_argument("--output-dir", default=TUNE_DIR)
    tune.add_argument("--seed", type=int, default=24923)
    tune.add_argument("--num-folds", type=int, default=5)
    tune.add_argument("--fold", type=int, required=True)
    tune.add_argument("--split-candidates", type=int, default=500)
    tune.add_argument("--epochs", type=int, default=50)
    tune.add_argument("--learning-rate", type=float, default=5e-5)
    tune.add_argument("--factor-lambda", type=float, default=FACTOR_LAMBDA)
    tune.add_argument(
        "--consistency-lambda",
        type=float,
        default=CONSISTENCY_LAMBDA,
    )
    tune.add_argument("--aqs-target-mean", type=float, default=0.45)
    tune.add_argument("--aqs-target-std", type=float, default=0.08)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--tune-dir", default=TUNE_DIR)
    summarize.add_argument("--num-folds", type=int, default=5)
    summarize.add_argument("--alpha-steps", type=int, default=41)
    summarize.add_argument("--bias-min", type=float, default=-2.0)
    summarize.add_argument("--bias-max", type=float, default=2.0)
    summarize.add_argument("--bias-steps", type=int, default=161)
    summarize.add_argument("--smoothing-window", type=int, default=5)
    summarize.add_argument("--minimum-epoch", type=int, default=3)

    train = commands.add_parser("train")
    add_training_arguments(train)
    train.add_argument("--model-dir", default=MODEL_DIR)
    train.add_argument("--train-cache", default=TRAIN_CACHE)
    train.add_argument("--tuning-summary", default=TUNING_SUMMARY)
    train.add_argument("--output-dir", default=FINAL_DIR)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int)

    evaluate = commands.add_parser("evaluate")
    add_training_arguments(evaluate)
    evaluate.add_argument("--checkpoint", nargs="+", required=True)
    evaluate.add_argument("--model-dir", default=MODEL_DIR)
    evaluate.add_argument("--test-cache", default=TEST_CACHE)
    evaluate.add_argument("--tuning-summary", default=TUNING_SUMMARY)
    evaluate.add_argument(
        "--output-dir",
        default="evaluation/qlung_factor_v3_final",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if hasattr(args, "input_length") and args.input_length <= 16:
        raise ValueError("input-length 必须大于 AST patch size。")
    if hasattr(args, "factor_dropout") and not 0.0 <= args.factor_dropout < 1.0:
        raise ValueError("factor-dropout 必须位于 [0, 1)。")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch-size 必须为正整数。")
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("num-workers 不能为负数。")
    if hasattr(args, "epochs") and args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs 必须为正整数。")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise ValueError("learning-rate 必须为正数。")
    if hasattr(args, "factor_lambda") and args.factor_lambda < 0:
        raise ValueError("factor-lambda 不能为负数。")
    if hasattr(args, "consistency_lambda") and args.consistency_lambda < 0:
        raise ValueError("consistency-lambda 不能为负数。")
    if hasattr(args, "num_folds") and args.num_folds < 2:
        raise ValueError("num-folds 必须至少为 2。")
    if hasattr(args, "split_candidates") and args.split_candidates <= 0:
        raise ValueError("split-candidates 必须为正整数。")
    if hasattr(args, "seed") and args.seed < 0:
        raise ValueError("seed 不能为负数。")
    if hasattr(args, "aqs_target_mean"):
        if not 0.0 < args.aqs_target_mean < 1.0:
            raise ValueError("aqs-target-mean 必须位于 (0, 1)。")
        if not 0.0 < args.aqs_target_std < 0.5:
            raise ValueError("aqs-target-std 必须位于 (0, 0.5)。")
    if hasattr(args, "alpha_steps") and args.alpha_steps < 2:
        raise ValueError("alpha-steps 必须至少为 2。")
    if hasattr(args, "bias_steps"):
        if args.bias_steps < 2 or args.bias_max <= args.bias_min:
            raise ValueError("bias 搜索范围无效。")
    if hasattr(args, "smoothing_window"):
        if args.smoothing_window <= 0 or args.smoothing_window % 2 == 0:
            raise ValueError("smoothing-window 必须是正奇数。")
    if hasattr(args, "minimum_epoch") and args.minimum_epoch <= 0:
        raise ValueError("minimum-epoch 必须为正整数。")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.command == "check":
        run_check(args)
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