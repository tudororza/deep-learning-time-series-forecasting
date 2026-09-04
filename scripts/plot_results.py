#!/usr/bin/env python3
"""Create report-ready benchmark and Jena result figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = ["#8BA6BD", "#6C91AE", "#5A7F9D", "#4B708E", "#6E9F86", "#26735B"]


def benchmark_plot(output_dir: Path) -> None:
    with Path("outputs/benchmark_results.json").open() as stream:
        results = json.load(stream)
    labels = ["Last", "Daily", "Weekly", "4-week mean", "Target-only", "Full LSTM"]
    values = [
        results["baselines"]["last"],
        results["baselines"]["daily"],
        results["baselines"]["weekly"],
        results["baselines"]["weekly_mean"],
        results["target_only_lstm"]["wape"],
        results["full_lstm"]["wape"],
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    bars = axis.bar(labels, values, color=COLORS)
    axis.set_ylabel("WAPE (lower is better)")
    axis.set_title("Internal 336-hour benchmark holdout")
    axis.set_ylim(0, 0.62)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(output_dir / "benchmark_wape.png", dpi=220)
    plt.close(figure)


def jena_plot(output_dir: Path) -> None:
    with Path("outputs/jena_results.json").open() as stream:
        results = json.load(stream)
    methods = ["last", "daily", "weekly", "lstm"]
    labels = ["Last", "Daily", "Weekly", "LSTM"]
    temperature = [results[name]["temperature_mae"] for name in methods]
    humidity = [results[name]["humidity_mae"] for name in methods]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for axis, values, title, color in (
        (axes[0], temperature, "Temperature", "#2F6B9A"),
        (axes[1], humidity, "Relative humidity", "#287A5A"),
    ):
        bars = axis.bar(labels, values, color=color, alpha=0.88)
        axis.set_title(title)
        axis.set_ylabel("MAE")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    figure.suptitle("Jena 2016 test performance")
    figure.tight_layout()
    figure.savefig(output_dir / "jena_mae.png", dpi=220)
    plt.close(figure)


def main() -> None:
    output_dir = Path("output/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_plot(output_dir)
    jena_plot(output_dir)
    print(f"Created plots in {output_dir}")


if __name__ == "__main__":
    main()

