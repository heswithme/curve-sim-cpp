#!/usr/bin/env python3
"""
Generate an N-dimensional grid of pool configurations.

Writes pretty JSON to python/arb_sim/run_data/pool_config.json.
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool

# -------------------- Grid Definition --------------------
# Each dimension: (name, min, max, count, log_scale, as_int)
DIMS = [
    ("donation_apy", 0.00, 0.1, 32, False, False),
    ("A", 10*10_000, 200*10_000, 32, False, True),
    # ("mid_fee", 1 / 10_000 * 10**10, 500 / 10_000 * 10**10, 64, False, True),
    ("out_fee", 20 / 10_000 * 10**10, 200 / 10_000 * 10**10, 10, False, True),
    ("fee_gamma", 0.001 * 10**18, 0.5 * 10**18, 10, True, True),
]
FEE_EQUALIZE = False  # If true, force out_fee == mid_fee

# DIMS = [
#     ("A", 100*10_000, 500*10_000, 1, False, True),
#     ("mid_fee", 1 / 10_000 * 10**10, 500 / 10_000 * 10**10, 1, False, True),
# ]
# FEE_EQUALIZE = True  # If true, force out_fee == mid_fee

# -------------------- Data Inputs --------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_DATAFILE = str(_SCRIPT_DIR / "trade_data" / "ethusd" / "ethusdt-2yup.json")
# DEFAULT_DATAFILE = str(_SCRIPT_DIR / "trade_data" / "usdchf" / "usdchf-20180101-20251231.json")
DEFAULT_DATAFILE = str(_SCRIPT_DIR / "trade_data" / "eurchf" / "eurchf-20180101-20251231.json")

DEFAULT_COWSWAP_FILE = None
DEFAULT_COWSWAP_FEE_BPS = 0.0

START_TS = _first_candle_ts(DEFAULT_DATAFILE)
INIT_PRICE = _initial_price_from_file(DEFAULT_DATAFILE)
INIT_LIQ = 10_000_000  # in coin0

# -------------------- Base Templates --------------------
BASE_POOL = {
    "initial_liquidity": [
        int(INIT_LIQ * 1e18 // 2),
        int(INIT_LIQ * 1e18 // 2 / INIT_PRICE),
    ],
    "A": int(3.5*10_000),
    "gamma": int(1e-4 * 10**18),
    "mid_fee": int(1 / 10_000 * 1e10),
    "out_fee": int(240 / 10_000 * 1e10),
    "fee_gamma": int(0.0023 * 1e18),
    "allowed_extra_profit": int(1e-12 * 10**18),
    "adjustment_step": int(0.001 * 10**18),
    "ma_time": 866,
    "initial_price": int(INIT_PRICE * 1e18),
    "start_timestamp": START_TS,
    "donation_apy": 0.05,
    "donation_frequency": 7 * 86400,
    "donation_coins_ratio": 0.5,
}

BASE_COSTS = {
    "arb_fee_bps": 10.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1,
}


def build_axis(min_v: float, max_v: float, n: int, log: bool, as_int: bool) -> list:
    """Build axis values using numpy linspace/logspace."""
    if n <= 0:
        raise ValueError("grid size must be positive")
    vals = np.geomspace(min_v, max_v, n) if log else np.linspace(min_v, max_v, n)
    return [int(round(v)) if as_int else float(v) for v in vals]


def build_grid() -> list:
    """Build all pool configurations from grid dimensions."""
    axes = [build_axis(*dim[1:]) for dim in DIMS]
    names = [dim[0] for dim in DIMS]

    pools = []
    for coords in itertools.product(*axes):
        pool = dict(BASE_POOL)
        tag_parts = []
        for name, val in zip(names, coords):
            pool[name] = val
            tag_parts.append(f"{name}_{val}")

        # Enforce out_fee >= mid_fee
        if "mid_fee" in pool:
            mid = int(pool["mid_fee"])
            out = int(pool.get("out_fee", 0))
            pool["mid_fee"] = mid
            pool["out_fee"] = mid if FEE_EQUALIZE else max(mid, out) + 1

        pools.append(
            {
                "tag": "__".join(tag_parts),
                "pool": strify_pool(pool),
                "costs": dict(BASE_COSTS),
            }
        )
    return pools


def main():
    pools = build_grid()

    grid_meta = {
        f"x{i}": {"name": d[0], "min": d[1], "max": d[2], "n": d[3], "log": d[4]}
        for i, d in enumerate(DIMS, 1)
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

    out_path = _SCRIPT_DIR / "run_data" / "pool_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(pools)} pool configs to {out_path}")


if __name__ == "__main__":
    main()
