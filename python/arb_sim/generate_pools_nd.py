#!/usr/bin/env python3
"""Generate an N-dimensional grid of pool configurations.

This is intentionally an explicit script: edit the active setup block below,
then run it. The helpers only keep the repetitive unit conversions and JSON
assembly out of the way.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DATA_DIR = SCRIPT_DIR / "run_data"

A_MULTIPLIER = 10_000
FEE_SCALE = 10**10
WAD = 10**18


# -------------------- Active setup --------------------
# Edit this block for a run.


# Point these paths at the pair being simulated. Set COWSWAP_FILE to None when
# no organic trade replay file should be associated with the generated config.
DATAFILE = SCRIPT_DIR / "trade_data" / "btcusd" / "btcusd-2023-2026-filtered.json"
COWSWAP_FILE = None
COWSWAP_FEE_BPS = 0.0

# Use either START_TIME or LAST_YEARS. If START_TIME is set, it wins.
START_TIME: str | None = "01-01-2024"  # Unix timestamp or DD-MM-YYYY
LAST_YEARS: float | None = None  # 2.0

OUT_PATH = RUN_DATA_DIR / "pool_config.json"
EXPAND_POOLS = False

INIT_LIQ = 1_000_000  # coin0 notional
ARB_FEE_BPS = 2

BASE_DONATION_APY = 0.0
BASE_DONATION_FREQUENCY = 3600
BASE_DONATION_DURATION = 7 * 86400
BASE_DONATION_COINS_RATIO = 0.5
# Grids sequence:
# 1. A-mf-don, 64**3, fee_equalize=True. Find best A & rpf. Donation. May be twice (1a): broad + zoom
# 2. fix some center-region A & don, unset fee_equalize!, do mf-of-fg 48**3 + 3x3 for a-rpf
# 3. with mf-of-fg fixed, scan A-rpf-donation
#
# GRID 1:
# Search good A & donation candidate, fixed fee surface, rpf collapsed into donations for now
N_GRID = 96
FEE_EQUALIZE = True
GRID: dict[str, Any] = {
    "A": [int(a * A_MULTIPLIER) for a in np.linspace(2, 8, N_GRID)],  # [5, 6, 7]],
    "mid_fee": [
        int(round(a / 10_000 * FEE_SCALE))
        for a in np.linspace(50, 250, N_GRID)  #
    ],
    "donation_apy": np.linspace(0.0, 0.06, 20),  # [0.02, 0.04, 0.08],  #
}
# GRID 2:
# With fixed A&donation, define fee surface. A and donation will be rescanned later, but this scan fixes fees
N_GRID = 64
FEE_EQUALIZE = False
GRID: dict[str, Any] = {
    "mid_fee": [
        int(round(a / 10_000 * FEE_SCALE))
        for a in np.linspace(20, 150, N_GRID)  #
    ],
    "out_fee": [
        int(round(a / 10_000 * FEE_SCALE)) for a in np.linspace(101, 250, N_GRID)
    ],
    "fee_gamma": [
        int(round(a * WAD)) for a in np.logspace(np.log10(1e-5), np.log10(1e-1), N_GRID)
    ],
    "donation_apy": [0.038],  #
    "A": [int(a * A_MULTIPLIER) for a in [5]],
}
# GRID 3: fees fixed now, can play A-don-rpf
N_GRID = 64
FEE_EQUALIZE = False
GRID: dict[str, Any] = {
    "A": [int(a * A_MULTIPLIER) for a in np.linspace(3, 7, N_GRID)],
    "donation_apy": np.linspace(0.0, 0.06, N_GRID),
    "reserved_profit_fraction": [
        int(round(a * FEE_SCALE)) for a in np.linspace(0.2, 0.75, N_GRID)
    ],
    "mid_fee": [
        int(round(a / 10_000 * FEE_SCALE))
        for a in [85]  #
    ],
    "out_fee": [
        int(round(a / 10_000 * FEE_SCALE)) for a in [150]
    ],
    "fee_gamma": [
        int(round(a * WAD)) for a in [0.001]
    ]
    # "adjustment_step_min": [int(a * 10**18) for a in np.linspace(0.000001, 0.000002, N_DENSE)],
}

# Alternative manual-FX style recipe:
# DATAFILE = SCRIPT_DIR / "trade_data" / "<pair>" / "<candles>.json"
# COWSWAP_FILE = SCRIPT_DIR / "trade_data" / "<pair>" / "<cowswap>.csv"
# START_TIME = "01-09-2024"
# LAST_YEARS = None
# BASE_DONATION_APY = 0.07
# BASE_DONATION_FREQUENCY = 86400
# GRID = {
#     "A": [int(a * 10_000) for a in np.linspace(10, 200, 24)],
#     "mid_fee": [int(a / 10_000 * 10**10) for a in np.linspace(1, 60, 24)],
#     "donation_apy": [0.02 * i for i in range(6)],
# }

# -------------------- Helpers --------------------


def _parse_start_time(value: str | None) -> int | None:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return int(
        datetime.strptime(value, "%d-%m-%Y").replace(tzinfo=timezone.utc).timestamp()
    )


def _data_array(path: Path) -> list[Any]:
    with path.open("r") as f:
        root = json.load(f)
    arr = (
        root
        if isinstance(root, list)
        else root.get("candles") or root.get("data") or root.get("events")
    )
    if not arr:
        raise ValueError(f"No data array found in {path}")
    return arr


def _last_candle_ts(path: Path) -> int:
    for row in reversed(_data_array(path)):
        if not isinstance(row, list) or not row:
            continue
        ts = int(row[0])
        if ts > 10_000_000_000:
            ts //= 1000
        return ts
    raise ValueError(f"No candle timestamp found in {path}")


def _start_candle_ts_and_price(path: Path, start_ts: int | None) -> tuple[int, float]:
    if start_ts is None:
        return _first_candle_ts(str(path)), _initial_price_from_file(str(path))

    for row in _data_array(path):
        if not isinstance(row, list) or len(row) < 2:
            continue
        ts = int(row[0])
        if ts > 10_000_000_000:
            ts //= 1000
        if ts < start_ts:
            continue
        price = float(row[4] if len(row) >= 5 else row[1])
        return ts, price
    raise ValueError(f"No candle found at or after start_ts={start_ts} in {path}")


def fixed_fee_policy(fee_bps: float) -> dict[str, Any]:
    return {
        "kind": "fixed_fee",
        "fee_bps": float(fee_bps),
    }


def build_base_pool(
    *,
    datafile: Path,
    start_time: str | None,
    last_years: float | None,
    init_liq: float,
    donation_apy: float,
    donation_frequency: int,
    donation_duration: int,
    donation_coins_ratio: float,
) -> dict[str, Any]:
    requested_start_ts = _parse_start_time(start_time)
    if requested_start_ts is None and last_years is not None:
        requested_start_ts = _last_candle_ts(datafile) - int(last_years * 365 * 86400)

    start_ts, init_price = _start_candle_ts_and_price(datafile, requested_start_ts)
    print(f"START_TS: {start_ts}")
    print(f"INIT_PRICE: {init_price}")

    return {
        "initial_liquidity": [
            int(init_liq * 1e18 // 2),
            int(init_liq * 1e18 // 2 / init_price),
        ],
        # Fixed to Polygon pool 0xdcb72c163de84618417bec9aef7ae32b5336d70e.
        "A": int(50 * A_MULTIPLIER),
        "gamma": int(1e-4 * WAD),
        "mid_fee": int(4 / 10_000 * 10**10),
        "out_fee": int(30 / 10_000 * 10**10),
        "fee_gamma": int(0.1 * WAD),
        "adjustment_step_min": int(0.0000000001 * WAD),
        "adjustment_step_max": int(0.5 / 100 * WAD),
        "ma_time": int(600 / np.log(2)),
        "reserved_profit_fraction": int(0.5 * FEE_SCALE),
        "admin_fee": int(0.0 * FEE_SCALE),
        "policy": {"kind": "none"},
        "initial_price": int(init_price * WAD),
        "start_timestamp": start_ts,
        "donation_apy": donation_apy,
        "donation_frequency": donation_frequency,
        "donation_duration": donation_duration,
        "donation_coins_ratio": donation_coins_ratio,
    }


def build_costs(arb_fee_bps: float) -> dict[str, Any]:
    return {
        "arb_fee_bps": arb_fee_bps,
        "gas_coin0": 0.0,
        "use_volume_cap": False,
        "volume_cap_mult": 1,
    }


def set_dotted_value(obj: dict[str, Any], dotted_name: str, value: Any) -> None:
    cur = obj
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        next_obj = cur.setdefault(part, {})
        if not isinstance(next_obj, dict):
            next_obj = {}
            cur[part] = next_obj
        cur = next_obj
    cur[parts[-1]] = value


def set_grid_value(
    pool: dict[str, Any],
    costs: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    if name == "policy.fee_bps":
        pool["policy"] = fixed_fee_policy(float(value))
    elif name.startswith("costs."):
        set_dotted_value(costs, name.removeprefix("costs."), value)
    else:
        set_dotted_value(pool, name, value)


def json_grid_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def axis_values(values: Any) -> list[Any]:
    raw_values = values.tolist() if isinstance(values, np.ndarray) else list(values)
    return [json_grid_value(x) for x in raw_values]


def build_grid(
    *,
    base_pool: dict[str, Any],
    base_costs: dict[str, Any],
    grid: dict[str, Any],
    fee_equalize: bool,
    expand_pools: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    names = list(grid.keys())
    axes = [axis_values(v) for v in grid.values()]
    grid_meta = {
        f"x{i}": {"name": name, "values": vals}
        for i, (name, vals) in enumerate(zip(names, axes), 1)
    }
    pool_count = math.prod(len(vals) for vals in axes)

    pools = []
    if not expand_pools:
        return pools, grid_meta, pool_count

    for coords in itertools.product(*axes):
        pool = copy.deepcopy(base_pool)
        costs = dict(base_costs)
        tag_parts = []
        for name, val in zip(names, coords):
            set_grid_value(pool, costs, name, val)
            tag_parts.append(f"{name}_{val}")

        if "mid_fee" in names or "out_fee" in names:
            mid = int(pool["mid_fee"])
            out = int(pool.get("out_fee", 0))
            pool["mid_fee"] = mid
            pool["out_fee"] = mid if fee_equalize else max(mid, out)

        pools.append(
            {
                "tag": "__".join(tag_parts),
                "pool": strify_pool(pool),
                "costs": costs,
            }
        )

    return pools, grid_meta, pool_count


def build_config() -> dict[str, Any]:
    datafile = Path(DATAFILE)
    cowswap_file = Path(COWSWAP_FILE) if COWSWAP_FILE else None

    base_pool = build_base_pool(
        datafile=datafile,
        start_time=START_TIME,
        last_years=LAST_YEARS,
        init_liq=INIT_LIQ,
        donation_apy=BASE_DONATION_APY,
        donation_frequency=BASE_DONATION_FREQUENCY,
        donation_duration=BASE_DONATION_DURATION,
        donation_coins_ratio=BASE_DONATION_COINS_RATIO,
    )
    base_costs = build_costs(ARB_FEE_BPS)
    pools, grid_meta, pool_count = build_grid(
        base_pool=base_pool,
        base_costs=base_costs,
        grid=GRID,
        fee_equalize=FEE_EQUALIZE,
        expand_pools=EXPAND_POOLS,
    )

    config = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid": grid_meta,
            "datafile": str(datafile),
            "cowswap_file": str(cowswap_file) if cowswap_file else None,
            "cowswap_fee_bps": COWSWAP_FEE_BPS,
            "requested_start_time": START_TIME,
            "start_time": str(base_pool["start_timestamp"]),
            "last_years": LAST_YEARS,
            "base_pool": strify_pool(base_pool),
            "base_costs": dict(base_costs),
            "pool_count": pool_count,
            "compact_grid": not EXPAND_POOLS,
            "fee_equalize": FEE_EQUALIZE,
        },
    }
    if EXPAND_POOLS:
        config["pools"] = pools
    return config


def main() -> None:
    out = build_config()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    if not EXPAND_POOLS:
        print(f"Wrote compact grid for {out['meta']['pool_count']} pools to {OUT_PATH}")
        print("C++ will generate pool configs lazily from meta.base_pool + meta.grid.")
    else:
        print(f"Wrote {out['meta']['pool_count']} expanded pool configs to {OUT_PATH}")


if __name__ == "__main__":
    main()
