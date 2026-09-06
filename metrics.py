"""Evaluation metrics for four-class ICBHI respiratory-sound classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ("Normal", "Crackle", "Wheeze", "Both")


@dataclass(frozen=True)
class ICBHIResult:
    score: float
    sensitivity: float
    specificity: float
    confusion_matrix: np.ndarray
    class_recall: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_recall": {
                name: value for name, value in zip(CLASS_NAMES, self.class_recall)
            },
        }


def _confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> np.ndarray:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 的长度必须一致。")
    if len(y_true) == 0:
        raise ValueError("不能对空预测计算指标。")

    valid_labels = {0, 1, 2, 3}
    observed = {int(value) for value in y_true} | {int(value) for value in y_pred}
    invalid = observed.difference(valid_labels)
    if invalid:
        raise ValueError(f"发现无效类别标签: {sorted(invalid)}")

    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])


def calc_icbhi_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ICBHIResult:
    """
    Calculate the standard four-class ICBHI score.

    Specificity is Normal recall. Sensitivity is the number of correctly
    classified abnormal cycles divided by all abnormal cycles. The final score
    is their arithmetic mean.
    """

    cm = _confusion_matrix(y_true, y_pred)
    normal_total = int(cm[0].sum())
    abnormal_total = int(cm[1:].sum())
    specificity = float(cm[0, 0] / normal_total) if normal_total else 0.0
    abnormal_correct = int(np.diag(cm)[1:].sum())
    sensitivity = (
        float(abnormal_correct / abnormal_total) if abnormal_total else 0.0
    )
    recalls = tuple(
        float(cm[index, index] / cm[index].sum()) if cm[index].sum() else 0.0
        for index in range(4)
    )
    return ICBHIResult(
        score=(sensitivity + specificity) / 2.0,
        sensitivity=sensitivity,
        specificity=specificity,
        confusion_matrix=cm,
        class_recall=recalls,
    )


def calc_legacy_macro_ovr_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict:
    """Reproduce the repository's historical, non-standard 0.6439 metric."""

    cm = _confusion_matrix(y_true, y_pred)
    total = int(cm.sum())
    sensitivities = []
    specificities = []

    for class_index in range(4):
        true_positive = int(cm[class_index, class_index])
        false_negative = int(cm[class_index].sum() - true_positive)
        false_positive = int(cm[:, class_index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        sensitivities.append(
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        specificities.append(
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        )

    macro_sensitivity = float(np.mean(sensitivities))
    macro_specificity = float(np.mean(specificities))
    return {
        "score": (macro_sensitivity + macro_specificity) / 2.0,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "class_sensitivity": [float(value) for value in sensitivities],
        "class_specificity": [float(value) for value in specificities],
        "confusion_matrix": cm.tolist(),
        "warning": "Legacy Macro-OvR metric; not the standard ICBHI score.",
    }