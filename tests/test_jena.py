from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.jena import CALENDAR_COLUMNS, TARGET_COLUMNS, prepare_jena


def test_jena_hourly_preparation_and_invalid_wind(tmp_path: Path) -> None:
    timestamps = pd.date_range("01.01.2014 00:00:00", periods=12, freq="10min")
    frame = pd.DataFrame(
        {
            "Date Time": timestamps.strftime("%d.%m.%Y %H:%M:%S"),
            "p (mbar)": np.arange(12, dtype=float) + 990,
            "T (degC)": np.arange(12, dtype=float),
            "rh (%)": np.arange(12, dtype=float) + 50,
            "wv (m/s)": [1.0] * 6 + [-9999.0] + [2.0] * 5,
            "max. wv (m/s)": [2.0] * 6 + [-9999.0] + [3.0] * 5,
            "wd (deg)": [0.0] * 12,
        }
    )
    path = tmp_path / "jena.csv"
    frame.to_csv(path, index=False)
    hourly = prepare_jena(path)
    assert len(hourly) == 2
    assert set(TARGET_COLUMNS).issubset(hourly.columns)
    assert set(CALENDAR_COLUMNS).issubset(hourly.columns)
    assert {"wind_x", "wind_y"}.issubset(hourly.columns)
    assert "wd (deg)" not in hourly.columns
    assert np.isfinite(hourly.select_dtypes(include=["number"]).to_numpy()).all()

