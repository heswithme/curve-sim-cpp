#!/usr/bin/env python3
"""
Plot price_scale vs CEX price from detailed-output.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

HAS_OUT_ARG = any(arg == "--out" or arg.startswith("--out=") for arg in sys.argv[1:])
if HAS_OUT_ARG:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


SEC_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
ROLLING_APY_WINDOW_S = 60.0 * 24.0 * 60.0 * 60.0


def _load_inspect_pool_meta(filepath):
    pool_path = filepath.with_name("inspect_pool.json")
    if not pool_path.exists():
        return {}
    try:
        payload = json.loads(pool_path.read_text())
    except Exception:
        return {}
    pools = payload.get("pools")
    if isinstance(pools, list) and pools:
        pool = pools[0].get("pool") if isinstance(pools[0], dict) else None
        if isinstance(pool, dict):
            return pool
    return {}


def _rolling_60d_net_apy(timestamps, vp, donation_apy, donation_frequency):
    if len(timestamps) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    elapsed = timestamps - timestamps[0]
    donation_apy = np.nan_to_num(donation_apy, nan=0.0)
    if donation_frequency and donation_frequency > 0.0:
        period_rate = donation_apy * donation_frequency / SEC_PER_YEAR
        donation_growth = np.power(1.0 + period_rate, elapsed / donation_frequency)
    else:
        donation_growth = np.power(1.0 + donation_apy, elapsed / SEC_PER_YEAR)

    with np.errstate(invalid="ignore", divide="ignore"):
        net_vp = vp / donation_growth

    rolling = np.full(len(timestamps), np.nan, dtype=float)
    floored = np.zeros(len(timestamps), dtype=bool)
    start = 0
    for i, ts in enumerate(timestamps):
        cutoff = ts - ROLLING_APY_WINDOW_S
        while start + 1 < len(timestamps) and timestamps[start + 1] <= cutoff:
            start += 1
        dt = ts - timestamps[start]
        if dt < ROLLING_APY_WINDOW_S or not (net_vp[start] > 0.0):
            continue
        growth = net_vp[i] / net_vp[start]
        if not np.isfinite(growth) or growth <= 0.0:
            rolling[i] = 0.0
            floored[i] = True
            continue
        apy = np.power(growth, SEC_PER_YEAR / dt) - 1.0
        rolling[i] = apy
        floored[i] = bool(np.isfinite(apy) and apy < 0.0)
    return rolling, floored


def main():
    parser = argparse.ArgumentParser(description="Plot price_scale vs CEX price")
    parser.add_argument("filepath", type=Path, help="Path to detailed-output.json")
    parser.add_argument("--no-save", action="store_true", help="Don't save PNG file")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    filepath = args.filepath
    with open(filepath) as f:
        data = json.load(f)

    timestamps = np.array([entry["t"] for entry in data])
    price_scale = np.array([entry["price_scale"] for entry in data])
    midpoints = np.array([(entry["open"] + entry["close"]) / 2 for entry in data])
    p_cex = np.array([entry.get("p_cex", mp) for entry, mp in zip(data, midpoints)])
    token0 = np.array([entry["token0"] for entry in data])
    token1 = np.array([entry["token1"] for entry in data])
    vp = np.array([entry["vp"] for entry in data])
    donation_apy = np.array([entry.get("donation_apy", 0.0) for entry in data])
    pool_meta = _load_inspect_pool_meta(filepath)
    donation_frequency = float(pool_meta.get("donation_frequency", 0.0) or 0.0)
    rolling_apy, rolling_floored = _rolling_60d_net_apy(
        timestamps,
        vp,
        donation_apy,
        donation_frequency,
    )

    # Pool imbalance: 4*x*y/(x+y)^2 where y = token1 * p_cex
    val0 = token0
    val1 = token1 * p_cex
    denom = val0 + val1
    imbalance = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    imbalance[mask] = 4.0 * val0[mask] * val1[mask] / (denom[mask] ** 2)
    imbalance *= 100.0  # as percentage

    rel_diff = (price_scale / p_cex - 1) * 100

    # Downsample to max 10k points for plotting after derived series are computed.
    MAX_POINTS = 10_000
    n = len(timestamps)
    if n > MAX_POINTS:
        indices = np.linspace(0, n - 1, MAX_POINTS, dtype=int)
        timestamps = timestamps[indices]
        price_scale = price_scale[indices]
        p_cex = p_cex[indices]
        imbalance = imbalance[indices]
        rel_diff = rel_diff[indices]
        rolling_apy = rolling_apy[indices]
        rolling_floored = rolling_floored[indices]

    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(t, timezone.utc) for t in timestamps]

    fig, (ax1, ax_apy, ax2) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Top plot: prices
    ax1.plot(
        dates,
        p_cex,
        label="CEX price",
        alpha=0.7,
        linewidth=1,
    )
    ax1.plot(dates, price_scale, label="price_scale", alpha=0.7, linewidth=1)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.set_title(f"Price Scale vs CEX Price\n{filepath.name}")
    ax1.grid(True, alpha=0.3)

    # Middle plot: rolling 60d annualized net APY.
    rolling_apy_pct = rolling_apy * 100.0
    ax_apy.plot(
        dates,
        rolling_apy_pct,
        linewidth=0.8,
        color="green",
        alpha=0.85,
        label="rolling 60d annualized net APY",
    )
    if np.any(rolling_floored & np.isfinite(rolling_apy_pct)):
        ax_apy.fill_between(
            dates,
            rolling_apy_pct,
            0.0,
            where=rolling_floored & np.isfinite(rolling_apy_pct),
            color="red",
            alpha=0.25,
            label="floored in GM",
        )
    ax_apy.axhline(0, color="black", linewidth=0.7)
    ax_apy.set_ylabel("60d net APY (%)")
    ax_apy.legend(loc="upper left")
    ax_apy.grid(True, alpha=0.3)

    # Bottom plot: relative difference (left axis) and pool balance (right axis)
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
        imbalance,
        linewidth=0.8,
        color="blue",
        alpha=0.6,
        label="Pool balance",
    )
    ax3.axhline(100, color="blue", linestyle=":", linewidth=0.5, alpha=0.5)
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

    if not args.no_save:
        out_path = args.out or filepath.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    if args.out is None:
        plt.show()


if __name__ == "__main__":
    main()
