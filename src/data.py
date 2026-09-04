"""Leakage-safe preprocessing and window datasets for forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ID_COLUMNS = ("series_id", "timestamp")
PASSTHROUGH_FEATURES = {
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "maintenance_known",
    "zone_sin",
    "zone_cos",
}


def validate_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str] = (),
    require_hourly: bool = True,
) -> pd.DataFrame:
    required = set(ID_COLUMNS) | set(feature_columns) | set(target_columns)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    clean = frame.copy()
    clean["series_id"] = clean["series_id"].astype(str)
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise")
    if clean[list(ID_COLUMNS)].duplicated().any():
        raise ValueError("Duplicate (series_id, timestamp) rows found")
    clean = clean.sort_values(list(ID_COLUMNS), kind="stable").reset_index(drop=True)
    if require_hourly:
        for series_id, group in clean.groupby("series_id", sort=False):
            differences = group["timestamp"].diff().dropna()
            if not differences.eq(pd.Timedelta(hours=1)).all():
                raise ValueError(f"Series {series_id} is not strictly hourly")
    for target in target_columns:
        values = pd.to_numeric(clean[target], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Target {target} contains missing or non-finite values")
        clean[target] = values
    return clean


@dataclass
class Preprocessor:
    feature_columns: list[str]
    passthrough_features: list[str]
    indicator_features: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    @property
    def output_features(self) -> list[str]:
        return self.feature_columns + [f"{name}__missing" for name in self.indicator_features]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        passthrough_features: set[str] | None = None,
    ) -> "Preprocessor":
        passthrough = passthrough_features if passthrough_features is not None else PASSTHROUGH_FEATURES
        ordered = list(feature_columns)
        numeric = frame[ordered].apply(pd.to_numeric, errors="coerce")
        indicator_features = [name for name in ordered if numeric[name].isna().any()]
        filled = numeric.groupby(frame["series_id"], sort=False).ffill()
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for name in ordered:
            median = float(numeric[name].median())
            if not np.isfinite(median):
                raise ValueError(f"Feature {name} has no finite training values")
            medians[name] = median
            values = filled[name].fillna(median).astype(np.float64)
            if name not in passthrough:
                means[name] = float(values.mean())
                std = float(values.std(ddof=0))
                stds[name] = std if std > 1e-8 else 1.0
        return cls(
            feature_columns=ordered,
            passthrough_features=[name for name in ordered if name in passthrough],
            indicator_features=indicator_features,
            medians=medians,
            means=means,
            stds=stds,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Input is missing model features: {sorted(missing)}")
        numeric = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        indicators = {
            name: numeric[name].isna().to_numpy(dtype=np.float32)
            for name in self.indicator_features
        }
        filled = numeric.groupby(frame["series_id"], sort=False).ffill()
        result: list[np.ndarray] = []
        passthrough = set(self.passthrough_features)
        for name in self.feature_columns:
            values = filled[name].fillna(self.medians[name]).to_numpy(dtype=np.float32)
            if name not in passthrough:
                values = (values - self.means[name]) / self.stds[name]
            result.append(values)
        result.extend(indicators[name] for name in self.indicator_features)
        output = np.column_stack(result).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise ValueError("Preprocessing produced non-finite feature values")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "passthrough_features": self.passthrough_features,
            "indicator_features": self.indicator_features,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "Preprocessor":
        return cls(**state)


def fit_target_stats(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-series, per-target mean and safe standard deviation."""
    means = targets.mean(axis=1, dtype=np.float64).astype(np.float32)
    stds = targets.std(axis=1, dtype=np.float64).astype(np.float32)
    stds[stds < 1e-6] = 1.0
    return means, stds


def seasonal_fallback(history: np.ndarray, horizon: int) -> np.ndarray:
    """Four-week mean, weekly persistence, or last value for one/many targets."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) >= 672:
        recent = values[-672:].reshape(4, 168, values.shape[1])
        pattern = recent.mean(axis=0)
        return np.concatenate([pattern] * ((horizon + 167) // 168), axis=0)[:horizon]
    if len(values) >= 168:
        pattern = values[-168:]
        return np.concatenate([pattern] * ((horizon + 167) // 168), axis=0)[:horizon]
    if not len(values):
        raise ValueError("Cannot create a baseline without target history")
    return np.repeat(values[-1:], horizon, axis=0)


class WindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Lazy windows over dense [series, time, feature] arrays."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_means: np.ndarray,
        target_stds: np.ndarray,
        *,
        history_length: int,
        horizon: int,
        stride: int,
        forecast_end: int,
        variant: str = "full",
        minimum_history: int = 672,
    ) -> None:
        if features.ndim != 3 or targets.ndim != 3:
            raise ValueError("features and targets must be [series, time, channels]")
        if features.shape[:2] != targets.shape[:2]:
            raise ValueError("features and targets must share series/time dimensions")
        if variant not in {"full", "target_only"}:
            raise ValueError(f"Unknown variant: {variant}")
        self.features = features
        self.targets = targets
        self.target_means = target_means
        self.target_stds = target_stds
        self.history_length = history_length
        self.horizon = horizon
        self.variant = variant
        first = max(history_length, minimum_history)
        last = forecast_end - horizon
        self.indices = [
            (series, start)
            for series in range(features.shape[0])
            for start in range(first, last + 1, stride)
        ]
        if not self.indices:
            raise ValueError("No training windows fit the requested configuration")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        series, start = self.indices[index]
        target_history = self.targets[series, start - self.history_length : start]
        normalized = (target_history - self.target_means[series]) / self.target_stds[series]
        if self.variant == "full":
            history = np.concatenate(
                [normalized, self.features[series, start - self.history_length : start]],
                axis=1,
            )
            future = self.features[series, start : start + self.horizon]
        else:
            history = normalized
            future = np.empty((self.horizon, 0), dtype=np.float32)
        baseline = seasonal_fallback(self.targets[series, :start], self.horizon)
        truth = self.targets[series, start : start + self.horizon]
        return (
            torch.from_numpy(history.astype(np.float32, copy=False)),
            torch.from_numpy(future.astype(np.float32, copy=False)),
            torch.tensor(series, dtype=torch.long),
            torch.from_numpy(baseline),
            torch.from_numpy(self.target_stds[series]),
            torch.from_numpy(truth.astype(np.float32, copy=False)),
        )


def dense_arrays(
    frame: pd.DataFrame,
    transformed_features: np.ndarray,
    target_columns: Sequence[str],
) -> tuple[list[str], list[pd.Timestamp], np.ndarray, np.ndarray]:
    """Convert a balanced panel into dense series-major arrays."""
    working = frame.copy()
    working["__row"] = np.arange(len(working))
    series_ids = sorted(working["series_id"].unique().tolist())
    groups = [working.loc[working["series_id"].eq(series_id)] for series_id in series_ids]
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError("All series must contain the same number of timestamps")
    reference = groups[0]["timestamp"].tolist()
    if any(group["timestamp"].tolist() != reference for group in groups[1:]):
        raise ValueError("All series must share the same timestamp grid")
    feature_array = np.stack(
        [transformed_features[group["__row"].to_numpy()] for group in groups]
    ).astype(np.float32)
    target_array = np.stack(
        [group[list(target_columns)].to_numpy(dtype=np.float32) for group in groups]
    )
    return series_ids, reference, feature_array, target_array

