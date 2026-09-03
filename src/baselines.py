"""Target-only baselines for the operations forecasting benchmark.

This module intentionally uses only Python's standard library so that the first
end-to-end benchmark and prediction file can be reproduced before installing
the deep-learning dependencies.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_targets(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Read and chronologically sort target observations by series."""
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"series_id", "timestamp", "target"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            grouped[row["series_id"]].append((row["timestamp"], float(row["target"])))

    for observations in grouped.values():
        observations.sort(key=lambda item: item[0])
    return dict(grouped)


def recursive_seasonal(history: list[float], horizon: int, lag: int) -> list[float]:
    """Repeat a seasonal lag recursively for the requested horizon."""
    if len(history) < lag:
        raise ValueError(f"Need at least {lag} history values, got {len(history)}")
    context = list(history)
    predictions: list[float] = []
    for _ in range(horizon):
        prediction = context[-lag]
        predictions.append(prediction)
        context.append(prediction)
    return predictions


def seasonal_mean(
    history: list[float], horizon: int, lag: int = 168, seasons: int = 4
) -> list[float]:
    """Repeat the mean pattern from the most recent complete seasons."""
    required = lag * seasons
    if len(history) < required:
        raise ValueError(f"Need at least {required} history values, got {len(history)}")
    recent = history[-required:]
    pattern = [
        sum(recent[offset::lag]) / seasons
        for offset in range(lag)
    ]
    return [pattern[index % lag] for index in range(horizon)]


def forecast(history: list[float], horizon: int, method: str) -> list[float]:
    if method == "last":
        return [history[-1]] * horizon
    if method == "daily":
        return recursive_seasonal(history, horizon, lag=24)
    if method == "weekly":
        return recursive_seasonal(history, horizon, lag=168)
    if method == "weekly_mean":
        return seasonal_mean(history, horizon, lag=168, seasons=4)
    raise ValueError(f"Unknown method: {method}")


def wape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for target, prediction in zip(actual, predicted, strict=True):
        numerator += abs(target - prediction)
        denominator += abs(target)
    if denominator == 0:
        raise ValueError("WAPE is undefined because all targets are zero")
    return numerator / denominator


def run_backtest(train_path: Path, horizon: int) -> None:
    grouped = read_targets(train_path)
    methods = ("last", "daily", "weekly", "weekly_mean")
    actual: list[float] = []
    predictions = {method: [] for method in methods}

    for series_id in sorted(grouped):
        values = [value for _, value in grouped[series_id]]
        if len(values) <= horizon:
            raise ValueError(f"{series_id} has too little history for horizon {horizon}")
        history = values[:-horizon]
        truth = values[-horizon:]
        actual.extend(truth)
        for method in methods:
            predictions[method].extend(forecast(history, horizon, method))

    print(f"Backtest: last {horizon} training hours across {len(grouped)} series")
    for method in methods:
        print(f"{method:12s} WAPE={wape(actual, predictions[method]):.6f}")


def read_forecast_index(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"series_id", "timestamp"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append((row["series_id"], row["timestamp"]))
    return rows


def write_predictions(
    train_path: Path,
    forecast_index_path: Path,
    output_path: Path,
    method: str,
) -> None:
    grouped_targets = read_targets(train_path)
    index_rows = read_forecast_index(forecast_index_path)
    timestamps_by_series: dict[str, list[str]] = defaultdict(list)
    for series_id, timestamp in index_rows:
        timestamps_by_series[series_id].append(timestamp)

    prediction_lookup: dict[tuple[str, str], float] = {}
    for series_id, timestamps in timestamps_by_series.items():
        if series_id not in grouped_targets:
            raise ValueError(f"Forecast index contains unknown series: {series_id}")
        ordered_timestamps = sorted(timestamps)
        history = [value for _, value in grouped_targets[series_id]]
        series_predictions = forecast(history, len(ordered_timestamps), method)
        prediction_lookup.update(
            {
                (series_id, timestamp): prediction
                for timestamp, prediction in zip(
                    ordered_timestamps, series_predictions, strict=True
                )
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["series_id", "timestamp", "prediction"]
        )
        writer.writeheader()
        for series_id, timestamp in index_rows:
            writer.writerow(
                {
                    "series_id": series_id,
                    "timestamp": timestamp,
                    "prediction": prediction_lookup[(series_id, timestamp)],
                }
            )
    print(f"Wrote {len(index_rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Backtest on the end of train.csv")
    backtest.add_argument("--train", type=Path, default=Path("train.csv"))
    backtest.add_argument("--horizon", type=int, default=336)

    predict = subparsers.add_parser("predict", help="Create a prediction CSV")
    predict.add_argument("--train", type=Path, default=Path("train.csv"))
    predict.add_argument(
        "--forecast-index",
        type=Path,
        default=Path("forecast_index_validation.csv"),
    )
    predict.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_weekly.csv"),
    )
    predict.add_argument(
        "--method",
        choices=("last", "daily", "weekly", "weekly_mean"),
        default="weekly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "backtest":
        run_backtest(args.train, args.horizon)
    else:
        write_predictions(
            args.train,
            args.forecast_index,
            args.output,
            args.method,
        )


if __name__ == "__main__":
    main()
