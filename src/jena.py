"""Jena weather preprocessing, baselines, LSTM training, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import Preprocessor, fit_target_stats, seasonal_fallback
from .device import preflight, write_report
from .model import ModelConfig, ResidualLSTM
from .training import train_model


TARGET_COLUMNS = ["T (degC)", "rh (%)"]
CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "year_sin", "year_cos"]


def prepare_jena(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "Date Time" not in raw:
        raise ValueError("Jena data is missing the 'Date Time' column")
    raw["timestamp"] = pd.to_datetime(
        raw.pop("Date Time"), format="%d.%m.%Y %H:%M:%S", errors="raise"
    )
    raw = raw.set_index("timestamp").sort_index()
    for name in ("wv (m/s)", "max. wv (m/s)"):
        if name in raw:
            raw.loc[raw[name] < 0, name] = np.nan
    if {"wv (m/s)", "wd (deg)"}.issubset(raw.columns):
        radians = np.deg2rad(raw["wd (deg)"])
        raw["wind_x"] = raw["wv (m/s)"] * np.cos(radians)
        raw["wind_y"] = raw["wv (m/s)"] * np.sin(radians)
        raw = raw.drop(columns=["wd (deg)"])
    hourly = raw.resample("1h").mean().ffill()
    timestamp = hourly.index
    hour = timestamp.hour.to_numpy()
    day_fraction = (
        timestamp.dayofyear.to_numpy() - 1 + hour / 24.0
    ) / 365.2425
    hourly["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    hourly["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    hourly["year_sin"] = np.sin(2 * np.pi * day_fraction)
    hourly["year_cos"] = np.cos(2 * np.pi * day_fraction)
    hourly = hourly.reset_index()
    hourly.insert(0, "series_id", "jena")
    if hourly[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Jena targets contain missing values after hourly aggregation")
    return hourly


class JenaWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        history_features: np.ndarray,
        future_features: np.ndarray,
        targets: np.ndarray,
        target_means: np.ndarray,
        target_stds: np.ndarray,
        *,
        history_length: int,
        horizon: int,
        stride: int,
        start: int,
        end: int,
    ) -> None:
        self.history_features = history_features
        self.future_features = future_features
        self.targets = targets
        self.target_means = target_means
        self.target_stds = target_stds
        self.history_length = history_length
        self.horizon = horizon
        first = max(start, history_length, 672)
        self.starts = list(range(first, end - horizon + 1, stride))
        if not self.starts:
            raise ValueError("No Jena windows fit this split")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        start = self.starts[index]
        target_history = self.targets[start - self.history_length : start]
        normalized = (target_history - self.target_means) / self.target_stds
        history = np.concatenate(
            [normalized, self.history_features[start - self.history_length : start]], axis=1
        )
        future = self.future_features[start : start + self.horizon]
        baseline = seasonal_fallback(self.targets[:start], self.horizon)
        truth = self.targets[start : start + self.horizon]
        return (
            torch.from_numpy(history.astype(np.float32, copy=False)),
            torch.from_numpy(future.astype(np.float32, copy=False)),
            torch.tensor(0, dtype=torch.long),
            torch.from_numpy(baseline),
            torch.from_numpy(self.target_stds.astype(np.float32, copy=False)),
            torch.from_numpy(truth.astype(np.float32, copy=False)),
        )


def _predict_starts(
    model: ResidualLSTM,
    history_features: np.ndarray,
    future_features: np.ndarray,
    targets: np.ndarray,
    target_means: np.ndarray,
    target_stds: np.ndarray,
    starts: list[int],
    history_length: int,
    horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for start in starts:
            normalized = (
                targets[start - history_length : start] - target_means
            ) / target_stds
            history = np.concatenate(
                [normalized, history_features[start - history_length : start]], axis=1
            )[None]
            future = future_features[start : start + horizon][None]
            baseline = seasonal_fallback(targets[:start], horizon)[None]
            prediction = model(
                torch.from_numpy(history.astype(np.float32)).to(device),
                torch.from_numpy(future.astype(np.float32)).to(device),
                torch.zeros(1, dtype=torch.long, device=device),
                torch.from_numpy(baseline).to(device),
                torch.from_numpy(target_stds[None].astype(np.float32)).to(device),
            )
            predictions.append(prediction.cpu().numpy()[0])
            truths.append(targets[start : start + horizon])
    return np.concatenate(predictions), np.concatenate(truths)


def _metrics(actual: np.ndarray, predicted: np.ndarray, target_stds: np.ndarray) -> dict[str, Any]:
    error = predicted - actual
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.square(error).mean(axis=0))
    return {
        "temperature_mae": float(mae[0]),
        "temperature_rmse": float(rmse[0]),
        "humidity_mae": float(mae[1]),
        "humidity_rmse": float(rmse[1]),
        "mean_normalized_mae": float(np.mean(mae / target_stds)),
    }


def _baseline_predictions(targets: np.ndarray, starts: list[int], horizon: int, lag: int | None) -> np.ndarray:
    output = []
    for start in starts:
        if lag is None:
            block = np.repeat(targets[start - 1 : start], horizon, axis=0)
        else:
            history = targets[:start].copy()
            values = []
            for _ in range(horizon):
                values.append(history[-lag])
                history = np.concatenate([history, history[-lag][None]], axis=0)
            block = np.asarray(values)
        output.append(block)
    return np.concatenate(output)


def train_jena(config: dict[str, Any]) -> dict[str, Any]:
    frame = prepare_jena(Path(config["data_path"]))
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp("2015-01-01")))
    validation_end = int(frame["timestamp"].searchsorted(pd.Timestamp("2016-01-01")))
    test_end = len(frame)
    history_columns = [
        name
        for name in frame.columns
        if name not in {"series_id", "timestamp", *TARGET_COLUMNS}
    ]
    preprocessor = Preprocessor.fit(
        frame.iloc[:train_end], history_columns, passthrough_features=set(CALENDAR_COLUMNS)
    )
    history_features = preprocessor.transform(frame)
    future_features = frame[CALENDAR_COLUMNS].to_numpy(dtype=np.float32)
    targets = frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    target_means, target_stds = fit_target_stats(targets[None, :train_end])
    target_means = target_means[0]
    target_stds = target_stds[0]
    dataset = JenaWindowDataset(
        history_features,
        future_features,
        targets,
        target_means,
        target_stds,
        history_length=int(config["history_length"]),
        horizon=int(config["forecast_horizon"]),
        stride=int(config["stride"]),
        start=0,
        end=train_end,
    )
    model = ResidualLSTM(
        ModelConfig(
            input_size=len(TARGET_COLUMNS) + history_features.shape[1],
            future_size=future_features.shape[1],
            n_series=1,
            output_size=len(TARGET_COLUMNS),
            hidden_size=int(config["hidden_size"]),
            embedding_dim=int(config["embedding_dim"]),
            dropout=float(config["dropout"]),
        )
    )
    device, report = preflight()
    write_report(report, Path("outputs/device_report.json"))
    validation_starts = list(
        range(train_end, validation_end - int(config["forecast_horizon"]) + 1, 24)
    )

    def evaluate(candidate: ResidualLSTM) -> float:
        prediction, truth = _predict_starts(
            candidate,
            history_features,
            future_features,
            targets,
            target_means,
            target_stds,
            validation_starts,
            int(config["history_length"]),
            int(config["forecast_horizon"]),
            device,
        )
        return _metrics(truth, prediction, target_stds)["mean_normalized_mae"]

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
    test_starts = list(range(validation_end, test_end - int(config["forecast_horizon"]) + 1, 24))
    prediction, truth = _predict_starts(
        model,
        history_features,
        future_features,
        targets,
        target_means,
        target_stds,
        test_starts,
        int(config["history_length"]),
        int(config["forecast_horizon"]),
        device,
    )
    metrics: dict[str, Any] = {"lstm": _metrics(truth, prediction, target_stds)}
    for name, lag in (("last", None), ("daily", 24), ("weekly", 168)):
        baseline = _baseline_predictions(targets, test_starts, int(config["forecast_horizon"]), lag)
        metrics[name] = _metrics(truth, baseline, target_stds)
    metrics["split"] = {
        "train": "2009-2014",
        "validation": "2015",
        "test": "2016",
        "test_windows": len(test_starts),
    }
    metrics["training"] = {
        "best_epoch": result.best_epoch,
        "validation_normalized_mae": result.best_metric,
        "elapsed_seconds": result.elapsed_seconds,
        "batch_size": result.batch_size,
        "device": asdict(report),
    }
    results_path = Path(config["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(metrics, indent=2) + "\n")
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "dataset": "jena",
            "model_state": result.state_dict,
            "model_config": model.config_dict(),
            "preprocessor": preprocessor.to_dict(),
            "target_means": torch.from_numpy(target_means),
            "target_stds": torch.from_numpy(target_stds),
            "training": metrics["training"],
        },
        checkpoint_path,
    )
    return metrics

