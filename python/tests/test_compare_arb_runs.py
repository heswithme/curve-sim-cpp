import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from compare_arb_runs import compare_runs  # noqa: E402


def _artifact(
    pool_exec_ms: float,
    exec_ms: float,
    trades: int = 7,
    *,
    postprocessed: bool = False,
) -> dict:
    run = {
        "result": {
            "trades": trades,
            "pool_exec_ms": pool_exec_ms,
            "apy_net": 0.1,
        },
        "final_state": {"D": "1000000000000000000"},
        "params": {"pool": {"A": "10000"}},
    }
    if postprocessed:
        run.update({"x1_key": "A", "x1_val": "10000"})
    else:
        run["success"] = True

    return {
        "metadata": {
            "exec_ms": exec_ms,
            "candles_read_ms": exec_ms / 2,
            "harness_wall_ms": exec_ms + 1.0,
            "postprocess_ms": 2.0,
            "wall_ms": exec_ms + 3.0,
            "threads": 10,
            "n_pools": 1,
            "events": 10,
        },
        "runs": [run],
    }


def test_compare_runs_ignores_timing_fields() -> None:
    left = _artifact(1.0, 10.0)
    right = _artifact(2.0, 11.0)
    right["metadata"]["threads"] = 12
    left["metadata"]["quiet"] = False
    right["metadata"]["quiet"] = True
    left["metadata"]["quiet_harness"] = False
    right["metadata"]["quiet_harness"] = True

    assert compare_runs(left, right) is None


def test_compare_runs_reports_first_economic_diff() -> None:
    diff = compare_runs(_artifact(1.0, 10.0), _artifact(2.0, 11.0, trades=8))

    assert diff == "$.runs[0].result.trades: 7 != 8"


def test_compare_runs_ignores_raw_vs_postprocessed_wrapper_fields() -> None:
    raw = _artifact(1.0, 10.0)
    processed = _artifact(2.0, 11.0, postprocessed=True)
    processed["metadata"].update(
        {
            "candles_file": "/abs/path/btcusd-2023-2026.json",
            "grid": {"x1": {"name": "A"}},
            "pool_config_file": "/tmp/pools.json",
            "total_trades": 7,
        }
    )
    raw["metadata"]["candles_file"] = "python/arb_sim/trade_data/btcusd/btcusd-2023-2026.json"

    assert compare_runs(raw, processed) is None


def test_compare_runs_treats_numeric_string_formatting_as_equal() -> None:
    left = _artifact(1.0, 10.0)
    right = _artifact(2.0, 11.0)
    left["runs"][0]["params"]["pool"]["donation_apy"] = "0"
    right["runs"][0]["params"]["pool"]["donation_apy"] = "0.0"
    left["runs"][0]["params"]["pool"]["adjustment_step_min"] = "1e-6"
    right["runs"][0]["params"]["pool"]["adjustment_step_min"] = "0.000001"

    assert compare_runs(left, right) is None
