#!/usr/bin/env python3
"""Report whether this PyTorch environment can train on the available GPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.device import preflight, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/device_report.json"))
    args = parser.parse_args()
    _, report = preflight()
    write_report(report, args.output)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

