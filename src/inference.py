"""CPU-safe inference helpers used by the submission entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .benchmark import _dense_features, recursive_predict
from .data import Preprocessor, dense_arrays, validate_frame
from .model import ModelConfig, ResidualLSTM


def discover_csv_files(
    input_dir: Path, feature_columns: list[str]
) -> tuple[Path | None, Path, Path | None]:
    """Return history, future-covariate, and forecast-index paths by schema."""
    history_candidates: list[tuple[int, Path]] = []
    future_candidates: list[tuple[int, Path]] = []
    index_candidates: list[tuple[int, Path]] = []
    required_features = set(feature_columns)
    for path in sorted(input_dir.rglob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if not {"series_id", "timestamp"}.issubset(columns):
            continue
        size = path.stat().st_size
        if "target" in columns:
            history_candidates.append((size, path))
        elif required_features.issubset(columns):
            future_candidates.append((size, path))
        elif {"series_id", "timestamp"}.issubset(columns):
            index_candidates.append((size, path))
    if not future_candidates:
        raise FileNotFoundError(
            f"No CSV in {input_dir} contains series_id, timestamp, and all model features"
        )
    history = max(history_candidates, default=(0, None))[1]
    future = max(future_candidates)[1]
    index = max(index_candidates, default=(0, None))[1]
    return history, future, index


def _ordered_output(
    predictions: np.ndarray,
    series_ids: list[str],
    timestamps: list[pd.Timestamp],
    index: pd.DataFrame,
) -> pd.DataFrame:
    lookup = {
        (series_id, timestamp): float(predictions[series_no, time_no, 0])
        for series_no, series_id in enumerate(series_ids)
        for time_no, timestamp in enumerate(timestamps)
    }
    ordered = index[["series_id", "timestamp"]].copy()
    ordered["series_id"] = ordered["series_id"].astype(str)
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    keys = list(zip(ordered["series_id"], ordered["timestamp"], strict=True))
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"Forecast index asks for an unavailable prediction: {missing[0]}")
    ordered["prediction"] = [lookup[key] for key in keys]
    if ordered[list(("series_id", "timestamp"))].duplicated().any():
        raise ValueError("Forecast index contains duplicate rows")
    values = ordered["prediction"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Inference produced invalid predictions")
    return ordered


def run_inference(input_dir: Path, output_file: Path, checkpoint_path: Path) -> None:
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format_version") != 1 or checkpoint.get("dataset") != "benchmark":
        raise ValueError("Unsupported checkpoint format")
    preprocessor = Preprocessor.from_dict(checkpoint["preprocessor"])
    series_ids = list(checkpoint["series_ids"])
    history_path, future_path, index_path = discover_csv_files(
        input_dir, preprocessor.feature_columns
    )

    future_raw = pd.read_csv(future_path)
    future_frame = validate_frame(
        future_raw, feature_columns=preprocessor.feature_columns
    )
    future_values = preprocessor.transform(future_frame)
    timestamps, future_features = _dense_features(future_frame, future_values, series_ids)

    target_history = checkpoint["target_history"].cpu().numpy().astype(np.float32)
    feature_history = checkpoint["feature_history"].cpu().numpy().astype(np.float32)
    if history_path is not None:
        history_raw = pd.read_csv(history_path)
        history_frame = validate_frame(
            history_raw,
            feature_columns=preprocessor.feature_columns,
            target_columns=["target"],
        )
        history_values = preprocessor.transform(history_frame)
        history_series, _, history_features, history_targets = dense_arrays(
            history_frame, history_values, ["target"]
        )
        if history_series != series_ids:
            raise ValueError("History series do not match checkpoint series IDs")
        target_history = history_targets[:, -672:]
        feature_history = history_features[:, -int(checkpoint["history_length"]):]

    model = ResidualLSTM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to("cpu")
    predictions = recursive_predict(
        model,
        target_history=target_history,
        feature_history=feature_history,
        future_features=future_features,
        target_means=checkpoint["target_means"].cpu().numpy().astype(np.float32),
        target_stds=checkpoint["target_stds"].cpu().numpy().astype(np.float32),
        history_length=int(checkpoint["history_length"]),
        block_size=int(checkpoint["forecast_horizon"]),
        variant=str(checkpoint["variant"]),
        device=torch.device("cpu"),
    )
    if index_path is not None:
        index = pd.read_csv(index_path)
    else:
        index = future_frame[["series_id", "timestamp"]]
    output = _ordered_output(predictions, series_ids, timestamps, index)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Wrote {len(output):,} rows to {output_file}")
