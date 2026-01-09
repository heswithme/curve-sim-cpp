#!/usr/bin/env python3
"""
Plot price_scale vs candle midpoint (open+close)/2 from detailed-output.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    token0 = np.array([entry["token0"] for entry in data])
    token1 = np.array([entry["token1"] for entry in data])

    # Pool imbalance: val0 / (val0 + val1) where val1 = token1 * midpoint
    val0 = token0
    val1 = token1 * midpoints
    pool_balance = val0 / (val0 + val1) * 100  # as percentage

    # Convert timestamps to datetime
    dates = [datetime.utcfromtimestamp(t) for t in timestamps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top plot: prices
    ax1.plot(
        dates,
        midpoints,
        label="CEX midpoint (open+close)/2",
        alpha=0.7,
        linewidth=1,
    )
    ax1.plot(dates, price_scale, label="price_scale", alpha=0.7, linewidth=1)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.set_title(f"Price Scale vs CEX Midpoint\n{filepath.name}")
    ax1.grid(True, alpha=0.3)

    # Bottom plot: relative difference (left axis) and pool balance (right axis)
    rel_diff = (price_scale / midpoints - 1) * 100
    ax2.plot(
        dates, rel_diff, linewidth=0.5, color="red", alpha=0.7, label="Price deviation"
    )
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(1, color="gray", linestyle="--", linewidth=0.5, label="±1%")
    ax2.axhline(-1, color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(5, color="orange", linestyle="--", linewidth=0.5, label="±5%")
    ax2.axhline(-5, color="orange", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Price deviation (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Secondary y-axis for pool balance
    ax3 = ax2.twinx()
    ax3.plot(
        dates,
        pool_balance,
        linewidth=0.8,
        color="blue",
        alpha=0.6,
        label="Pool balance",
    )
    ax3.axhline(50, color="blue", linestyle=":", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("Pool balance (%)", color="blue")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper right")

    # Format x-axis as datetime
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()  # Rotate date labels

    plt.tight_layout()

    # Save next to input file
    out_path = filepath.with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
