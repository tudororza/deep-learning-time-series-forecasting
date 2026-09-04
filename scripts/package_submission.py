#!/usr/bin/env python3
"""Build the self-contained model archive expected by private evaluation."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/benchmark_model_submission.zip")
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    project_root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forecast-submission-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        shutil.copy2(project_root / "predict.py", root / "predict.py")
        shutil.copy2(args.checkpoint, root / "checkpoint.pt")
        shutil.copy2(project_root / "requirements-inference.txt", root / "requirements.txt")
        shutil.copy2(project_root / "README.md", root / "README.md")
        shutil.copytree(
            project_root / "src",
            root / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

