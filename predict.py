#!/usr/bin/env python3
"""Generate benchmark predictions using the evaluator's required interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    run_inference(args.input_dir, args.output_file, args.checkpoint)


if __name__ == "__main__":
    main()

