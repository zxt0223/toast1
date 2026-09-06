# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

from qlung_factor_pipeline import (
    factor_class_probabilities,
    factor_targets,
    select_smoothed_epoch,
    tune_postprocess,
)


def test_factor_target_mapping():
    labels = torch.tensor([0, 1, 2, 3])
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    assert torch.equal(factor_targets(labels), expected)


def test_factor_composition_has_expected_class_order_and_unit_sum():
    logits = torch.tensor(
        [
            [-10.0, -10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
            [10.0, 10.0],
        ]
    )
    probability = factor_class_probabilities(logits)
    assert probability.argmax(dim=1).tolist() == [0, 1, 2, 3]
    assert torch.allclose(probability.sum(dim=1), torch.ones(4), atol=1e-6)


def test_postprocess_can_select_factor_branch():
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    wrong_class_logits = torch.zeros(8, 4)
    wrong_class_logits[:, 0] = 5.0
    factor_logits = torch.tensor(
        [
            [-8.0, -8.0],
            [-7.0, -7.0],
            [8.0, -8.0],
            [7.0, -7.0],
            [-8.0, 8.0],
            [-7.0, 7.0],
            [8.0, 8.0],
            [7.0, 7.0],
        ]
    )
    tuned = tune_postprocess(
        wrong_class_logits,
        factor_logits,
        labels,
        alpha_steps=11,
        bias_min=-1.0,
        bias_max=1.0,
        bias_steps=21,
    )
    assert tuned["class_weight"] < 1.0
    assert tuned["score"] == pytest.approx(1.0)


def test_smoothed_epoch_selection_ignores_single_epoch_spike():
    scores = [0.50, 0.51, 0.80, 0.52, 0.53, 0.60, 0.61, 0.62, 0.61]
    curve = [
        {
            "epoch": index + 1,
            "score": score,
            "metrics": {},
            "class_weight": 0.5,
            "normal_log_bias": 0.0,
            "sensitivity": score,
            "specificity": score,
        }
        for index, score in enumerate(scores)
    ]
    selected, enriched = select_smoothed_epoch(curve, 5, 3)
    assert selected in {7, 8}
    assert all("smoothed_score" in item for item in enriched)