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
    ("donation_apy", 0.00, 0.1, 10, False, False),
    ("A", 10 * 10_000, 200 * 10_000, 24, False, True),
    # ("mid_fee", 1 / 10_000 * 10**10, 10 / 10_000 * 10**10, 3, False, True),
    ("out_fee", 10 / 10_000 * 10**10, 200 / 10_000 * 10**10, 24, False, True),
    ("fee_gamma", 0.001 * 10**18, 0.5 * 10**18, 8, True, True),
    ("ma_time", 866, 3600 * 4 / np.log(2), 6, False, True),
]
FEE_EQUALIZE = False  # If true, force out_fee == mid_fee

# Manual grid override: if set, uses these exact values instead of DIMS
# Format: {"param_name": [val1, val2, ...], ...}
# Example:
# MANUAL_GRID = {
#     "A": [100_000, 200_000, 500_000],
#     "donation_apy": [0.0, 0.05, 0.1],
# }
MANUAL_GRID = None
N_DENSE = 64
MANUAL_GRID = {
    "out_fee": np.linspace(10 / 10_000 * 10**10, 100 / 10_000 * 10**10, N_DENSE),
    "A": np.linspace(10 * 10_000, 150 * 10_000, N_DENSE),
    "donation_apy": np.linspace(0.0, 0.2, N_DENSE),

    # # "mid_fee": [int(a / 10_000 * 10**10) for a in [1, 5]],
    # # "ma_time": [int(a / np.log(2)) for a in [600, 3600, 3600 * 4]],
    # "ma_time": [int(a / np.log(2)) for a in [600, 3600]],
    # # "donation_apy": [0.0, 0.025, 0.05], #, 0.075, 0.1],
    # "donation_apy": np.linspace(0.0, 0.1, 11),
    # # "fee_gamma": np.geomspace(0.001 * 10**18, 0.5 * 10**18, 8),
    # "fee_gamma": [int(a*10**18) for a in [0.001, 0.003, 0.01, 0.05, 0.3, 0.5, 1.0]],
    # "adjustment_step": [int(a * 10**18) for a in [0.001, 0.005]],
}
# DIMS = [
#     ("A", 100*10_000, 500*10_000, 1, False, True),
#     ("mid_fee", 1 / 10_000 * 10**10, 500 / 10_000 * 10**10, 1, False, True),
# ]
# FEE_EQUALIZE = True  # If true, force out_fee == mid_fee

# -------------------- Data Inputs --------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_DATAFILE = str(_SCRIPT_DIR / "trade_data" / "ethusd" / "ethusdt-2yup.json")
# DEFAULT_DATAFILE = str(
#     _SCRIPT_DIR / "trade_data" / "usdchf" / "usdchf-20180101-20251231.json"
# )
# DEFAULT_DATAFILE = str(
#     _SCRIPT_DIR / "trade_data" / "chfusd" / "chfusd-20180101-20251231.json"
# )
DEFAULT_DATAFILE = str(
    _SCRIPT_DIR / "trade_data" / "eurusd" / "eurusd-20180101-20251231.json"
)
# DEFAULT_DATAFILE = str(_SCRIPT_DIR / "trade_data" / "eurchf" / "eurchf-20180101-20251231.json")

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
    "A": int(3.5 * 10_000),
    "gamma": int(1e-4 * 10**18),
    "mid_fee": int(1 / 10_000 * 1e10),
    "out_fee": int(240 / 10_000 * 1e10),
    "fee_gamma": int(0.003 * 1e18),
    # "allowed_extra_profit": int(1e-12 * 10**18),
    "adjustment_step": int(0.005 * 10**18),
    # "adjustment_step": int(1e-7 * 10**18), # ONLY FOR OLD POOLS
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


def build_grid() -> tuple[list, dict]:
    """Build all pool configurations from grid dimensions.

    Returns:
        (pools, grid_meta) tuple
    """
    # Use MANUAL_GRID if set, otherwise generate from DIMS
    if MANUAL_GRID is not None:
        names = list(MANUAL_GRID.keys())
        # Convert numpy arrays to lists for JSON serialization
        axes = [
            list(v) if isinstance(v, np.ndarray) else list(v)
            for v in MANUAL_GRID.values()
        ]
        grid_meta = {
            f"x{i}": {"name": name, "values": [float(x) for x in vals]}
            for i, (name, vals) in enumerate(zip(names, axes), 1)
        }
    else:
        names = [dim[0] for dim in DIMS]
        axes = [build_axis(*dim[1:]) for dim in DIMS]
        grid_meta = {
            f"x{i}": {"name": d[0], "min": d[1], "max": d[2], "n": d[3], "log": d[4]}
            for i, d in enumerate(DIMS, 1)
        }

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
    return pools, grid_meta


def main():
    pools, grid_meta = build_grid()

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
