#!/usr/bin/env python3
"""Extract an archive and execute its required inference command in isolation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"predict.py", "checkpoint.pt", "requirements.txt", "README.md"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forecast-archive-check-") as temporary:
        root = Path(temporary) / "submission"
        output = Path(temporary) / "output" / "predictions.csv"
        root.mkdir()
        with zipfile.ZipFile(args.archive) as bundle:
            names = set(bundle.namelist())
            missing = REQUIRED.difference(names)
            if missing:
                raise ValueError(f"Archive is missing: {sorted(missing)}")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise ValueError("Archive contains an unsafe path")
            bundle.extractall(root)
        subprocess.run(
            [
                sys.executable,
                "predict.py",
                "--input_dir",
                str(args.input_dir.resolve()),
                "--output_file",
                str(output),
                "--checkpoint",
                str(root / "checkpoint.pt"),
            ],
            cwd=root,
            check=True,
        )
        frame = pd.read_csv(output)
        if list(frame.columns) != ["series_id", "timestamp", "prediction"]:
            raise ValueError(f"Wrong prediction schema: {list(frame.columns)}")
        if not np.isfinite(frame["prediction"]).all() or (frame["prediction"] < 0).any():
            raise ValueError("Predictions are not finite and nonnegative")
        print(f"Archive check passed: {len(frame):,} predictions")


if __name__ == "__main__":
    main()

