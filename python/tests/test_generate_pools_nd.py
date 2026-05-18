import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from generate_pools_nd import (  # noqa: E402
    build_config,
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

    pools, grid_meta, pool_count = build_grid(
        base_pool=_base_pool(),
        base_costs={"arb_fee_bps": 10.0},
        grid=grid,
        fee_equalize=True,
        expand_pools=True,
    )

    assert len(pools) == 4
    assert pool_count == 4
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

    pools, _, pool_count = build_grid(
        base_pool=base_pool,
        base_costs={"arb_fee_bps": 10.0},
        grid=grid,
        fee_equalize=True,
        expand_pools=True,
    )

    assert pool_count == 2
    assert base_pool["policy"] == {"kind": "none"}
    assert pools[0]["pool"]["policy"] == {"kind": "fixed_fee", "fee_bps": 10.0}
    assert pools[-1]["pool"]["policy"] == {"kind": "fixed_fee", "fee_bps": 20.0}


def test_expanded_grid_supports_dotted_pool_and_cost_axes() -> None:
    grid = {
        "policy.fee_bps": [25],
        "costs.arb_fee_bps": [1.0, 2.0],
        "policy.custom_param": [7],
    }

    pools, grid_meta, pool_count = build_grid(
        base_pool=_base_pool(),
        base_costs={"arb_fee_bps": 10.0, "gas_coin0": 0.0},
        grid=grid,
        fee_equalize=False,
        expand_pools=True,
    )

    assert pool_count == 2
    assert grid_meta["x2"] == {"name": "costs.arb_fee_bps", "values": [1.0, 2.0]}
    assert pools[0]["pool"]["policy"] == {
        "kind": "fixed_fee",
        "fee_bps": 25.0,
        "custom_param": 7,
    }
    assert pools[0]["costs"]["arb_fee_bps"] == 1.0
    assert pools[1]["costs"]["arb_fee_bps"] == 2.0


def test_build_grid_supports_coupled_axis() -> None:
    coupled_pairs = [
        [int(5 * 10_000), int(round(0.15 * 10**10))],
        [int(7 * 10_000), int(round(0.20 * 10**10))],
        [int(10 * 10_000), int(round(0.25 * 10**10))],
    ]
    grid = {
        "mid_fee": [30_000_000, 40_000_000],
        "fee_gamma": [10**15, 2 * 10**15],
        ("A", "reserved_profit_fraction"): coupled_pairs,
    }

    pools, grid_meta, pool_count = build_grid(
        base_pool=_base_pool(),
        base_costs={"arb_fee_bps": 10.0},
        grid=grid,
        fee_equalize=False,
        expand_pools=True,
    )

    assert pool_count == 12
    assert len(pools) == 12
    assert grid_meta["x3"] == {
        "names": ["A", "reserved_profit_fraction"],
        "values": coupled_pairs,
    }
    observed_pairs = {
        (pool["pool"]["A"], pool["pool"]["reserved_profit_fraction"])
        for pool in pools
    }
    assert observed_pairs == {
        (str(a), str(reserved_profit_fraction))
        for a, reserved_profit_fraction in coupled_pairs
    }


def test_expanded_grid_raises_out_fee_to_mid_fee_floor() -> None:
    grid = {
        "mid_fee": [150_000_000],
        "out_fee": [101_000_000],
    }

    pools, _, pool_count = build_grid(
        base_pool=_base_pool(),
        base_costs={"arb_fee_bps": 10.0},
        grid=grid,
        fee_equalize=False,
        expand_pools=True,
    )

    assert pool_count == 1
    assert pools[0]["pool"]["mid_fee"] == "150000000"
    assert pools[0]["pool"]["out_fee"] == "150000000"


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
        donation_duration=604800,
        donation_coins_ratio=0.5,
    )

    assert pool["adjustment_step_min"] == 10**8


def test_build_config_defaults_to_compact_grid(monkeypatch, tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    candles.write_text("[[1700000000, 100.0, 100.0, 100.0, 100.0]]")

    import generate_pools_nd

    monkeypatch.setattr(generate_pools_nd, "DATAFILE", candles)
    monkeypatch.setattr(generate_pools_nd, "COWSWAP_FILE", None)
    monkeypatch.setattr(generate_pools_nd, "START_TIME", None)
    monkeypatch.setattr(generate_pools_nd, "LAST_YEARS", None)
    monkeypatch.setattr(generate_pools_nd, "EXPAND_POOLS", False)
    monkeypatch.setattr(
        generate_pools_nd,
        "GRID",
        {
            "A": [10_000, 20_000],
            "mid_fee": [10**7, 2 * 10**7],
        },
    )

    config = build_config()

    assert "pools" not in config
    assert config["meta"]["compact_grid"] is True
    assert config["meta"]["pool_count"] == 4
    assert config["meta"]["base_pool"]["A"] == str(50 * 10_000)
    assert config["meta"]["base_costs"]["arb_fee_bps"] == 2
    assert config["meta"]["grid"]["x1"] == {"name": "A", "values": [10_000, 20_000]}
    assert config["meta"]["grid"]["x2"] == {
        "name": "mid_fee",
        "values": [10**7, 2 * 10**7],
    }
