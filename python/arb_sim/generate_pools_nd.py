#!/usr/bin/env python3
"""
Generate an N-dimensional grid of pool configurations (no CLI args).

- Pool parameters are specified in their native units:
  - Integers for fees (1e10), WAD-like fields (1e18), balances (1e18).
  - Floats allowed for harness-only fields like donation_apy (plain fraction).
- Values are stringified in the output JSON under the "pool" object.
- Supports linear or log spacing per dimension.

Writes a pretty JSON to python/arb_sim/run_data/pool_config.json with entries of
form {tag, pool, costs}.

Grid metadata uses x1..xN keys (no X/Y).
"""

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool

# -------------------- Grid Definition --------------------
# Example 3D grid: A, donation_apy, mid_fee (10x10x10)
# Keep names consistent with pool param keys.
# # DIM_NAMES = ["donation_apy", "A"]
DIM_NAMES = ["donation_apy", "A"]
DIM_MINS = [0.01, 2 * 10_000]
DIM_MAXS = [3.0, 100 * 10_000]
DIM_COUNTS = [16, 16]
DIM_LOG = [False, False]
DIM_AS_INT = [False, True]

# If true, force out_fee == mid_fee
FEE_EQUALIZE = False

# -------------------- Data Inputs --------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATAFILE = str(_SCRIPT_DIR / "trade_data" / "ethusd" / "ethusdt-2yup.json")
DEFAULT_COWSWAP_FILE = None
DEFAULT_COWSWAP_FEE_BPS = 0.0

START_TS = _first_candle_ts(DEFAULT_DATAFILE)
init_price = _initial_price_from_file(DEFAULT_DATAFILE)
init_liq = 10_000_000  # in coin0

INVERT_LIQ = False
if INVERT_LIQ:
    # inverse if necessary (make it coin1)
    init_liq *= init_price

# -------------------- Base Templates --------------------
BASE_POOL = {
    # All values are integers in their native units
    "initial_liquidity": [
        int(init_liq * 10**18 // 2),
        int(init_liq * 10**18 // 2 / init_price),
    ],
    "A": 3.5 * 10_000,
    "gamma": 10**14,  # unused in twocrypto
    "mid_fee": int(43 / 10_000 * 10**10),
    "out_fee": int(240 / 10_000 * 10**10),
    "fee_gamma": int(0.0023 * 10**18),
    "allowed_extra_profit": int(1e-12 * 10**18),
    "adjustment_step": int(0.005 * 10**18),  # 1%
    "ma_time": 866,
    "initial_price": int(init_price * 10**18),
    "start_timestamp": START_TS,
    # Donations (harness-only):
    # - donation_apy: plain fraction per year (0.05 => 5%).
    # - donation_frequency: seconds between donations.
    # - donation_coins_ratio: fraction of donation in coin1 (0=all coin0, 1=all coin1)
    "donation_apy": 0.01,
    "donation_frequency": int(7 * 86400),
    "donation_coins_ratio": 0.5,
}

BASE_COSTS = {
    "arb_fee_bps": 10.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1,
}


def _linspace(start: float, stop: float, num: int) -> List[float]:
    """Simple linspace implementation."""
    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _logspace(start: float, stop: float, num: int) -> List[float]:
    """Simple logspace implementation."""
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    log_vals = _linspace(log_start, log_stop, num)
    return [10**x for x in log_vals]


def _build_axis(
    min_v: float, max_v: float, n: int, logspace: bool, as_int: bool = False
) -> List[float]:
    if n <= 0:
        raise ValueError("grid size must be positive")
    if logspace:
        if min_v <= 0 or max_v <= 0:
            raise ValueError("logspace requires positive min/max")
        vals = _logspace(min_v, max_v, n)
    else:
        vals = _linspace(min_v, max_v, n)
    if as_int:
        vals = [round(x) for x in vals]
    return vals


def build_grid():
    if not (
        len(DIM_NAMES)
        == len(DIM_MINS)
        == len(DIM_MAXS)
        == len(DIM_COUNTS)
        == len(DIM_LOG)
        == len(DIM_AS_INT)
    ):
        raise ValueError("DIM_* arrays must be the same length")

    axes = [
        _build_axis(mn, mx, n, lg, ai)
        for mn, mx, n, lg, ai in zip(
            DIM_MINS, DIM_MAXS, DIM_COUNTS, DIM_LOG, DIM_AS_INT
        )
    ]

    pools = []
    for coords in itertools.product(*axes):
        pool = dict(BASE_POOL)
        tag_parts = []
        for name, value in zip(DIM_NAMES, coords):
            pool[name] = value
            tag_parts.append(f"{name}_{value}")

        # Enforce out_fee >= mid_fee if both exist
        if "mid_fee" in pool:
            mid_fee_val = int(pool.get("mid_fee", 0))
            cur_out_val = int(pool.get("out_fee", 0))
            pool["mid_fee"] = int(mid_fee_val)
            pool["out_fee"] = (
                mid_fee_val if FEE_EQUALIZE else max(mid_fee_val, cur_out_val) + 1
            )

        costs = dict(BASE_COSTS)
        tag = "__".join(tag_parts)
        pools.append({"tag": tag, "pool": strify_pool(pool), "costs": costs})

    return pools


def main():
    pools = build_grid()

    # Build grid metadata with x1..xN keys
    grid_meta = {}
    for idx, (name, mn, mx, cnt, lg) in enumerate(
        zip(DIM_NAMES, DIM_MINS, DIM_MAXS, DIM_COUNTS, DIM_LOG), start=1
    ):
        grid_meta[f"x{idx}"] = {
            "name": name,
            "min": mn,
            "max": mx,
            "n": cnt,
            "log": lg,
        }

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid": grid_meta,
            "datafile": DEFAULT_DATAFILE,
            "cowswap_file": DEFAULT_COWSWAP_FILE,
            "cowswap_fee_bps": DEFAULT_COWSWAP_FEE_BPS,
            "base_pool": strify_pool(BASE_POOL),
        },
        "pools": pools,
    }

    out_path = Path(__file__).resolve().parent / "run_data" / "pool_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(pools)} pool configs to {out_path}")


if __name__ == "__main__":
    main()
