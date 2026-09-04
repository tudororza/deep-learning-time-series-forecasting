from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import Preprocessor, WindowDataset, seasonal_fallback, validate_frame


def test_preprocessor_is_causal_and_uses_only_fit_statistics() -> None:
    fit = pd.DataFrame(
        {
            "series_id": ["a", "a", "b", "b"],
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h").tolist() * 2,
            "value": [1.0, np.nan, 3.0, 5.0],
        }
    )
    processor = Preprocessor.fit(fit, ["value"], passthrough_features=set())
    assert processor.medians["value"] == 3.0
    assert processor.indicator_features == ["value"]
    transformed = processor.transform(fit)
    assert transformed.shape == (4, 2)
    assert transformed[1, 0] == pytest.approx(transformed[0, 0])
    assert transformed[1, 1] == 1.0
    outlier = fit.iloc[[0]].copy()
    outlier["value"] = 10_000.0
    processor.transform(outlier)
    assert processor.medians["value"] == 3.0


def test_validation_rejects_duplicate_and_nonhourly_rows() -> None:
    duplicate = pd.DataFrame(
        {
            "series_id": ["a", "a"],
            "timestamp": ["2024-01-01", "2024-01-01"],
            "x": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_frame(duplicate, feature_columns=["x"])


def test_seasonal_fallback_priority() -> None:
    history = np.arange(672, dtype=np.float32)
    prediction = seasonal_fallback(history, 24)[:, 0]
    expected = np.stack([history[-672 + offset :: 168].mean() for offset in range(24)])
    np.testing.assert_allclose(prediction, expected)
    weekly = seasonal_fallback(np.arange(200, dtype=np.float32), 24)[:, 0]
    np.testing.assert_allclose(weekly, np.arange(32, 56, dtype=np.float32))
    last = seasonal_fallback(np.array([2.0, 3.0], dtype=np.float32), 3)[:, 0]
    np.testing.assert_allclose(last, [3.0, 3.0, 3.0])


def test_window_dataset_shapes_and_no_future_target_in_history() -> None:
    features = np.zeros((2, 720, 3), dtype=np.float32)
    targets = np.arange(2 * 720, dtype=np.float32).reshape(2, 720, 1)
    means = targets[:, :700].mean(axis=1)
    stds = targets[:, :700].std(axis=1)
    dataset = WindowDataset(
        features,
        targets,
        means,
        stds,
        history_length=168,
        horizon=24,
        stride=6,
        forecast_end=720,
        variant="full",
    )
    history, future, series, baseline, scale, truth = dataset[0]
    assert history.shape == (168, 4)
    assert future.shape == (24, 3)
    assert series.ndim == 0
    assert baseline.shape == truth.shape == (24, 1)
    assert scale.shape == (1,)

