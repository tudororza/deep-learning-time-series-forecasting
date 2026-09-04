"""Benchmark training, recursive validation, and checkpoint serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines import forecast as baseline_forecast
from .data import (
    Preprocessor,
    WindowDataset,
    dense_arrays,
    fit_target_stats,
    seasonal_fallback,
    validate_frame,
)
from .device import DeviceReport, preflight, write_report
from .model import ModelConfig, ResidualLSTM
from .training import fit_fixed_epochs, train_model, wape


TARGET = "target"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in {"series_id", "timestamp", TARGET}]


def _dense_features(
    frame: pd.DataFrame,
    transformed: np.ndarray,
    expected_series: list[str],
) -> tuple[list[pd.Timestamp], np.ndarray]:
    working = frame.copy()
    working["__row"] = np.arange(len(working))
    groups = []
    for series_id in expected_series:
        group = working.loc[working["series_id"].eq(series_id)]
        if group.empty:
            raise ValueError(f"Missing future rows for {series_id}")
        groups.append(group)
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError("Future series have unequal lengths")
    reference = groups[0]["timestamp"].tolist()
    if any(group["timestamp"].tolist() != reference for group in groups[1:]):
        raise ValueError("Future series do not share a timestamp grid")
    values = np.stack([transformed[group["__row"].to_numpy()] for group in groups]).astype(np.float32)
    return reference, values


def recursive_predict(
    model: ResidualLSTM,
    *,
    target_history: np.ndarray,
    feature_history: np.ndarray,
    future_features: np.ndarray,
    target_means: np.ndarray,
    target_stds: np.ndarray,
    history_length: int,
    block_size: int,
    variant: str,
    device: torch.device,
) -> np.ndarray:
    """Roll forward in blocks, appending predictions as target history."""
    model.eval()
    rolling_targets = np.asarray(target_history, dtype=np.float32).copy()
    rolling_features = np.asarray(feature_history, dtype=np.float32).copy()
    outputs: list[np.ndarray] = []
    total_horizon = future_features.shape[1]
    with torch.no_grad():
        for start in range(0, total_horizon, block_size):
            stop = min(start + block_size, total_horizon)
            horizon = stop - start
            target_tail = rolling_targets[:, -history_length:]
            normalized = (target_tail - target_means[:, None, :]) / target_stds[:, None, :]
            if variant == "full":
                feature_tail = rolling_features[:, -history_length:]
                history = np.concatenate([normalized, feature_tail], axis=2)
                future = future_features[:, start:stop]
            else:
                history = normalized
                future = np.empty((len(target_tail), horizon, 0), dtype=np.float32)
            baseline = np.stack(
                [seasonal_fallback(rolling_targets[index], horizon) for index in range(len(rolling_targets))]
            )
            prediction = model(
                torch.from_numpy(history).to(device),
                torch.from_numpy(future).to(device),
                torch.arange(len(history), dtype=torch.long, device=device),
                torch.from_numpy(baseline).to(device),
                torch.from_numpy(target_stds).to(device),
            ).clamp_min(0.0)
            block = prediction.cpu().numpy().astype(np.float32)
            outputs.append(block)
            rolling_targets = np.concatenate([rolling_targets, block], axis=1)
            rolling_features = np.concatenate(
                [rolling_features, future_features[:, start:stop]], axis=1
            )
    return np.concatenate(outputs, axis=1)


def baseline_scores(targets: np.ndarray, cutoff: int, horizon: int) -> dict[str, float]:
    truth = targets[:, cutoff : cutoff + horizon, 0]
    scores: dict[str, float] = {}
    for method in ("last", "daily", "weekly", "weekly_mean"):
        prediction = np.stack(
            [baseline_forecast(series[:cutoff, 0].tolist(), horizon, method) for series in targets]
        )
        scores[method] = wape(truth, prediction)
    return scores


def _make_checkpoint(
    model: ResidualLSTM,
    *,
    preprocessor: Preprocessor,
    series_ids: list[str],
    target_means: np.ndarray,
    target_stds: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    training: dict[str, Any],
    device_report: DeviceReport,
) -> dict[str, Any]:
    history_length = int(config["history_length"])
    return {
        "format_version": 1,
        "dataset": "benchmark",
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": model.config_dict(),
        "preprocessor": preprocessor.to_dict(),
        "series_ids": series_ids,
        "target_means": torch.from_numpy(np.ascontiguousarray(target_means)),
        "target_stds": torch.from_numpy(np.ascontiguousarray(target_stds)),
        # Copy slices before torch.save so NumPy's much larger backing arrays are
        # not retained in the serialized storage.
        "target_history": torch.from_numpy(np.ascontiguousarray(targets[:, -672:])),
        "feature_history": torch.from_numpy(
            np.ascontiguousarray(features[:, -history_length:])
        ),
        "variant": config["variant"],
        "history_length": history_length,
        "forecast_horizon": int(config["forecast_horizon"]),
        "training": training,
        "device_report": asdict(device_report),
    }


def save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"saved checkpoint to {path}")


def write_prediction_frame(
    predictions: np.ndarray,
    series_ids: list[str],
    timestamps: list[pd.Timestamp],
    index_path: Path,
    output_path: Path,
) -> None:
    lookup = {
        (series_id, timestamp): float(predictions[series_index, time_index, 0])
        for series_index, series_id in enumerate(series_ids)
        for time_index, timestamp in enumerate(timestamps)
    }
    index = pd.read_csv(index_path)
    index["timestamp"] = pd.to_datetime(index["timestamp"], errors="raise")
    keys = list(zip(index["series_id"].astype(str), index["timestamp"], strict=True))
    unknown = [key for key in keys if key not in lookup]
    if unknown:
        raise ValueError(f"Forecast index contains timestamps not predicted, first: {unknown[0]}")
    output = index[["series_id", "timestamp"]].copy()
    output["prediction"] = [lookup[key] for key in keys]
    if not np.isfinite(output["prediction"]).all() or (output["prediction"] < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"wrote {len(output):,} predictions to {output_path}")


def train_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    train_path = Path(config["train_path"])
    raw = pd.read_csv(train_path)
    feature_columns = _feature_columns(raw)
    frame = validate_frame(raw, feature_columns=feature_columns, target_columns=[TARGET])
    series_count = frame["series_id"].nunique()
    steps = len(frame) // series_count
    validation_horizon = int(config["validation_horizon"])
    cutoff = steps - validation_horizon
    if cutoff <= 672:
        raise ValueError("Training split is too short for the four-week baseline")

    fit_mask = frame.groupby("series_id", sort=False).cumcount().lt(cutoff)
    preprocessor = Preprocessor.fit(frame.loc[fit_mask], feature_columns)
    transformed = preprocessor.transform(frame)
    series_ids, timestamps, features, targets = dense_arrays(frame, transformed, [TARGET])
    target_means, target_stds = fit_target_stats(targets[:, :cutoff])

    dataset = WindowDataset(
        features,
        targets,
        target_means,
        target_stds,
        history_length=int(config["history_length"]),
        horizon=int(config["forecast_horizon"]),
        stride=int(config["stride"]),
        forecast_end=cutoff,
        variant=config["variant"],
    )
    input_size = targets.shape[2] + (features.shape[2] if config["variant"] == "full" else 0)
    future_size = features.shape[2] if config["variant"] == "full" else 0
    model_config = ModelConfig(
        input_size=input_size,
        future_size=future_size,
        n_series=len(series_ids),
        output_size=targets.shape[2],
        hidden_size=int(config["hidden_size"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
    )
    model = ResidualLSTM(model_config)
    device, device_report = preflight()
    write_report(device_report, Path("outputs/device_report.json"))
    print(f"training on {device}; windows={len(dataset):,}")
    baseline_result = baseline_scores(targets, cutoff, validation_horizon)
    print("baseline WAPE:", json.dumps(baseline_result, indent=2))

    def evaluate(candidate: ResidualLSTM) -> float:
        prediction = recursive_predict(
            candidate,
            target_history=targets[:, :cutoff],
            feature_history=features[:, :cutoff],
            future_features=features[:, cutoff:],
            target_means=target_means,
            target_stds=target_stds,
            history_length=int(config["history_length"]),
            block_size=int(config["forecast_horizon"]),
            variant=config["variant"],
            device=device,
        )
        return wape(targets[:, cutoff:], prediction)

    result = train_model(
        model,
        dataset,
        device=device,
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        max_epochs=int(config["max_epochs"]),
        patience=int(config["patience"]),
        gradient_clip=float(config["gradient_clip"]),
        seed=int(config["seed"]),
        evaluate=evaluate,
        num_workers=int(config.get("num_workers", 0)),
    )
    model.load_state_dict(result.state_dict)

    if config.get("retrain_full", False):
        print(f"retraining on all public targets for {result.best_epoch} epochs")
        preprocessor = Preprocessor.fit(frame, feature_columns)
        transformed = preprocessor.transform(frame)
        series_ids, timestamps, features, targets = dense_arrays(frame, transformed, [TARGET])
        target_means, target_stds = fit_target_stats(targets)
        full_dataset = WindowDataset(
            features,
            targets,
            target_means,
            target_stds,
            history_length=int(config["history_length"]),
            horizon=int(config["forecast_horizon"]),
            stride=int(config["stride"]),
            forecast_end=steps,
            variant=config["variant"],
        )
        model = ResidualLSTM(model_config)
        retrained = fit_fixed_epochs(
            model,
            full_dataset,
            device=device,
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            epochs=result.best_epoch,
            gradient_clip=float(config["gradient_clip"]),
            seed=int(config["seed"]),
            num_workers=int(config.get("num_workers", 0)),
        )
        training_info = {
            "selected_epoch": result.best_epoch,
            "internal_wape": result.best_metric,
            "selection_seconds": result.elapsed_seconds,
            "retrain_seconds": retrained.elapsed_seconds,
            "batch_size": retrained.batch_size,
            "baseline_wape": baseline_result,
            "seed": int(config["seed"]),
        }
    else:
        training_info = {
            "selected_epoch": result.best_epoch,
            "internal_wape": result.best_metric,
            "selection_seconds": result.elapsed_seconds,
            "batch_size": result.batch_size,
            "baseline_wape": baseline_result,
            "seed": int(config["seed"]),
        }

    checkpoint = _make_checkpoint(
        model,
        preprocessor=preprocessor,
        series_ids=series_ids,
        target_means=target_means,
        target_stds=target_stds,
        features=features,
        targets=targets,
        config=config,
        training=training_info,
        device_report=device_report,
    )
    checkpoint_path = Path(config["checkpoint_path"])
    save_checkpoint(checkpoint, checkpoint_path)

    future_path = Path(config["validation_input_path"])
    future_raw = pd.read_csv(future_path)
    future_frame = validate_frame(future_raw, feature_columns=feature_columns)
    future_transformed = preprocessor.transform(future_frame)
    future_timestamps, future_features = _dense_features(future_frame, future_transformed, series_ids)
    validation_prediction = recursive_predict(
        model,
        target_history=targets,
        feature_history=features,
        future_features=future_features,
        target_means=target_means,
        target_stds=target_stds,
        history_length=int(config["history_length"]),
        block_size=int(config["forecast_horizon"]),
        variant=config["variant"],
        device=device,
    )
    write_prediction_frame(
        validation_prediction,
        series_ids,
        future_timestamps,
        Path(config["forecast_index_path"]),
        Path(config["predictions_path"]),
    )
    return training_info


def refit_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Refit an already-selected benchmark configuration on all public targets."""
    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Run model selection first; checkpoint not found: {checkpoint_path}"
        )
    selected: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    selected_epoch = int(selected["training"]["selected_epoch"])
    raw = pd.read_csv(Path(config["train_path"]))
    feature_columns = _feature_columns(raw)
    frame = validate_frame(raw, feature_columns=feature_columns, target_columns=[TARGET])
    preprocessor = Preprocessor.fit(frame, feature_columns)
    transformed = preprocessor.transform(frame)
    series_ids, _, features, targets = dense_arrays(frame, transformed, [TARGET])
    target_means, target_stds = fit_target_stats(targets)
    dataset = WindowDataset(
        features,
        targets,
        target_means,
        target_stds,
        history_length=int(config["history_length"]),
        horizon=int(config["forecast_horizon"]),
        stride=int(config["stride"]),
        forecast_end=targets.shape[1],
        variant=config["variant"],
    )
    model = ResidualLSTM(ModelConfig(**selected["model_config"]))
    device, device_report = preflight()
    write_report(device_report, Path("outputs/device_report.json"))
    result = fit_fixed_epochs(
        model,
        dataset,
        device=device,
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        epochs=selected_epoch,
        gradient_clip=float(config["gradient_clip"]),
        seed=int(config["seed"]),
        num_workers=int(config.get("num_workers", 0)),
    )
    training_info = dict(selected["training"])
    training_info.update(
        {
            "retrained_on_all_public_data": True,
            "retrain_seconds": result.elapsed_seconds,
            "retrain_batch_size": result.batch_size,
        }
    )
    checkpoint = _make_checkpoint(
        model,
        preprocessor=preprocessor,
        series_ids=series_ids,
        target_means=target_means,
        target_stds=target_stds,
        features=features,
        targets=targets,
        config=config,
        training=training_info,
        device_report=device_report,
    )
    save_checkpoint(checkpoint, checkpoint_path)

    future_raw = pd.read_csv(Path(config["validation_input_path"]))
    future_frame = validate_frame(future_raw, feature_columns=feature_columns)
    future_transformed = preprocessor.transform(future_frame)
    future_timestamps, future_features = _dense_features(
        future_frame, future_transformed, series_ids
    )
    predictions = recursive_predict(
        model,
        target_history=targets,
        feature_history=features,
        future_features=future_features,
        target_means=target_means,
        target_stds=target_stds,
        history_length=int(config["history_length"]),
        block_size=int(config["forecast_horizon"]),
        variant=config["variant"],
        device=device,
    )
    write_prediction_frame(
        predictions,
        series_ids,
        future_timestamps,
        Path(config["forecast_index_path"]),
        Path(config["predictions_path"]),
    )
    return training_info
