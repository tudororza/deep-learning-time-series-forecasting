from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import Preprocessor
from src.inference import run_inference
from src.model import ModelConfig, ResidualLSTM


def test_exact_inference_contract(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    series_ids = ["a", "b"]
    future_times = pd.date_range("2024-02-01", periods=48, freq="h")
    future = pd.DataFrame(
        [
            {"series_id": series, "timestamp": timestamp, "x": float(index)}
            for series in series_ids
            for index, timestamp in enumerate(future_times)
        ]
    )
    future.to_csv(input_dir / "mystery_covariates.csv", index=False)
    index = future[["series_id", "timestamp"]].sample(frac=1, random_state=42)
    index.to_csv(input_dir / "mystery_index.csv", index=False)
    fit_frame = future.copy()
    processor = Preprocessor.fit(fit_frame, ["x"], passthrough_features=set())
    model = ResidualLSTM(ModelConfig(1, 0, 2, 1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    target_history = np.stack(
        [np.arange(672, dtype=np.float32) + offset for offset in (0, 10)]
    )[..., None]
    checkpoint = {
        "format_version": 1,
        "dataset": "benchmark",
        "model_state": model.state_dict(),
        "model_config": model.config_dict(),
        "preprocessor": processor.to_dict(),
        "series_ids": series_ids,
        "target_means": torch.from_numpy(target_history.mean(axis=1)),
        "target_stds": torch.from_numpy(target_history.std(axis=1)),
        "target_history": torch.from_numpy(target_history),
        "feature_history": torch.zeros(2, 168, 1),
        "variant": "target_only",
        "history_length": 168,
        "forecast_horizon": 24,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    output_path = tmp_path / "output" / "predictions.csv"
    run_inference(input_dir, output_path, checkpoint_path)
    output = pd.read_csv(output_path)
    assert list(output.columns) == ["series_id", "timestamp", "prediction"]
    assert len(output) == 96
    assert output[["series_id", "timestamp"]].astype(str).to_numpy().tolist() == index.astype(str).to_numpy().tolist()
    assert np.isfinite(output["prediction"]).all()
    assert (output["prediction"] >= 0).all()

