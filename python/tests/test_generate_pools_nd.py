import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from generate_pools_nd import (  # noqa: E402
    build_base_pool,
    build_grid,
)


def _base_pool() -> dict:
    return {
        "initial_liquidity": [10**21, 10**21],
        "A": 50 * 10_000,
        "gamma": 10**14,
        "mid_fee": int(4 / 10_000 * 10**10),
        "out_fee": int(30 / 10_000 * 10**10),
        "fee_gamma": 10**17,
        "adjustment_step_min": 10**12,
        "adjustment_step_max": 5 * 10**15,
        "ma_time": 866,
        "reserved_profit_fraction": 5 * 10**9,
        "admin_fee": 5 * 10**9,
        "policy": {"kind": "none"},
        "initial_price": 10**18,
        "start_timestamp": 1_700_000_000,
        "donation_apy": 0.10,
        "donation_frequency": 604800,
        "donation_coins_ratio": 0.5,
    }


def test_grid_uses_explicit_raw_values_and_fixed_fee_policy_axis() -> None:
    grid = {
        "A": [int(a * 10_000) for a in np.linspace(1, 2, 2)],
        "policy.fee_bps": np.linspace(10, 20, 2),
        "adjustment_step_min": [int(0.000001 * 10**18)],
    }

    pools, grid_meta = build_grid(
        base_pool=_base_pool(),
        base_costs={"arb_fee_bps": 10.0},
        grid=grid,
        fee_equalize=True,
    )

    assert len(pools) == 4
    assert grid_meta["x1"] == {"name": "A", "values": [10000.0, 20000.0]}
    assert grid_meta["x2"] == {"name": "policy.fee_bps", "values": [10.0, 20.0]}
    assert grid_meta["x3"] == {
        "name": "adjustment_step_min",
        "values": [10**12],
    }

    first_pool = pools[0]["pool"]
    assert first_pool["A"] == "10000"
    assert first_pool["adjustment_step_min"] == "1000000000000"
    assert first_pool["policy"] == {"kind": "fixed_fee", "fee_bps": 10.0}


def test_build_grid_does_not_share_nested_policy_state_between_pools() -> None:
    base_pool = _base_pool()
    grid = {"policy.fee_bps": np.linspace(10, 20, 2)}

    pools, _ = build_grid(
        base_pool=base_pool,
        base_costs={"arb_fee_bps": 10.0},
        grid=grid,
        fee_equalize=True,
    )

    assert base_pool["policy"] == {"kind": "none"}
    assert pools[0]["pool"]["policy"] == {"kind": "fixed_fee", "fee_bps": 10.0}
    assert pools[-1]["pool"]["policy"] == {"kind": "fixed_fee", "fee_bps": 20.0}


def test_base_pool_uses_near_zero_adjustment_step_min(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    candles.write_text("[[1700000000, 100.0, 100.0, 100.0, 100.0]]")

    pool = build_base_pool(
        datafile=candles,
        start_time=None,
        last_years=None,
        init_liq=1000.0,
        donation_apy=0.10,
        donation_frequency=604800,
        donation_coins_ratio=0.5,
    )

    assert pool["adjustment_step_min"] == 10**12
