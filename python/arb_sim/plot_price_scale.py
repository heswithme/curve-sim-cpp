#!/usr/bin/env python3
"""
Plot price_scale vs candle midpoint (open+close)/2 from detailed-output.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <detailed-output.json>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    with open(filepath) as f:
        data = json.load(f)

    # Downsample to max 10k points for plotting
    MAX_POINTS = 1_000
    n = len(data)
    if n > MAX_POINTS:
        indices = np.linspace(0, n - 1, MAX_POINTS, dtype=int)
        data = [data[i] for i in indices]

    timestamps = np.array([entry["t"] for entry in data])
    price_scale = np.array([entry["price_scale"] for entry in data])
    midpoints = np.array([(entry["open"] + entry["close"]) / 2 for entry in data])

    # Convert timestamps to hours from start
    t_hours = (timestamps - timestamps[0]) / 3600

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top plot: prices
    ax1.plot(
        t_hours,
        midpoints,
        label="CEX midpoint (open+close)/2",
        alpha=0.7,
        linewidth=1,
    )
    ax1.plot(t_hours, price_scale, label="price_scale", alpha=0.7, linewidth=1)
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.set_title(f"Price Scale vs CEX Midpoint\n{filepath.name}")
    ax1.grid(True, alpha=0.3)

    # Bottom plot: relative difference
    rel_diff = (price_scale / midpoints - 1) * 100
    ax2.plot(t_hours, rel_diff, linewidth=0.5, color="red", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(1, color="gray", linestyle="--", linewidth=0.5, label="±1%")
    ax2.axhline(-1, color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(5, color="orange", linestyle="--", linewidth=0.5, label="±5%")
    ax2.axhline(-5, color="orange", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Relative diff (%)")
    ax2.set_xlabel("Time (hours)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save next to input file
    out_path = filepath.with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
