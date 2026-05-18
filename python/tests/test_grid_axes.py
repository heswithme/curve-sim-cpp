import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from find_local_maxima_orjson import _run_coord_values as local_coord_values  # noqa: E402
from find_ranked_maxima import (  # noqa: E402
    _build_env as ranked_build_env,
    _run_coord_values as ranked_coord_values,
)
from grid_axes import infer_grid_from_runs, pool_value  # noqa: E402


def _raw_run(a: str, fee_bps: float, donation_apy: float) -> dict:
    return {
        "params": {
            "pool": {
                "initial_liquidity": [10**21, 10**21],
                "A": a,
                "gamma": "100000000000000",
                "policy": {
                    "kind": "fixed_fee",
                    "fee_bps": fee_bps,
                    "fee": int(fee_bps * 1_000_000),
                },
                "donation_apy": donation_apy,
            },
        },
        "result": {"apy_net": 0.1, "avg_rel_price_diff": 0.001},
        "final_state": {},
    }


def test_infer_grid_axes_from_raw_harness_params_pool() -> None:
    runs = [
        _raw_run(a, fee_bps, donation_apy)
        for a in ("10000", "20000")
        for fee_bps in (10.0, 20.0)
        for donation_apy in (0.0, 0.2)
    ]

    names, values, x_keys = infer_grid_from_runs(runs)

    assert names == ["A", "policy.fee_bps", "donation_apy"]
    assert values == [[10000.0, 20000.0], [10.0, 20.0], [0.0, 0.2]]
    assert x_keys == ["x1", "x2", "x3"]


def test_ranking_tools_read_nested_raw_harness_axes() -> None:
    run = _raw_run("10000", 10.0, 0.2)
    names = ["A", "policy.fee_bps", "donation_apy"]

    assert ranked_coord_values(run, [], names) == [10000.0, 10.0, 0.2]
    assert local_coord_values(run, [], names) == [10000.0, 10.0, 0.2]
    assert ranked_build_env(run)["policy.fee_bps"] == 10.0


def test_pool_value_accepts_flat_dotted_cluster_axes() -> None:
    pool = {"A": "10000", "policy.fee_bps": 25.0}

    assert pool_value(pool, "A") == "10000"
    assert pool_value(pool, "policy.fee_bps") == 25.0
