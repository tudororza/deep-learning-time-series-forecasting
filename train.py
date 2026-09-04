#!/usr/bin/env python3
"""Train either the operations benchmark model or the Jena experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark import load_config, refit_benchmark, train_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--retrain-full",
        action="store_true",
        help="Retrain a selected benchmark model on all public training targets",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("dataset") == "benchmark":
        result = refit_benchmark(config) if args.retrain_full else train_benchmark(config)
    elif config.get("dataset") == "jena":
        from src.jena import train_jena

        result = train_jena(config)
    else:
        raise ValueError(f"Unknown dataset: {config.get('dataset')}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
