from __future__ import annotations

import numpy as np
import torch

from src.baselines import forecast
from src.benchmark import recursive_predict
from src.model import ModelConfig, ResidualLSTM
from src.training import wape


def test_baseline_regression_and_wape() -> None:
    history = list(map(float, range(700)))
    assert forecast(history, 3, "last") == [699.0, 699.0, 699.0]
    assert forecast(history, 3, "daily") == [676.0, 677.0, 678.0]
    assert wape(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 1.0 / 3.0


def test_recursive_rollout_uses_predictions_as_history() -> None:
    model = ResidualLSTM(ModelConfig(1, 0, 2, 1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    targets = np.stack(
        [np.arange(700, dtype=np.float32), np.arange(700, dtype=np.float32) + 10]
    )[..., None]
    result = recursive_predict(
        model,
        target_history=targets,
        feature_history=np.empty((2, 168, 0), dtype=np.float32),
        future_features=np.empty((2, 336, 0), dtype=np.float32),
        target_means=targets.mean(axis=1),
        target_stds=targets.std(axis=1),
        history_length=168,
        block_size=24,
        variant="target_only",
        device=torch.device("cpu"),
    )
    assert result.shape == (2, 336, 1)
    assert np.isfinite(result).all()
    assert (result >= 0).all()

