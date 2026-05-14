import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from yb_releverage import (  # noqa: E402
    SimulationConfig,
    load_trace,
    make_fee_grid,
    run_releverage,
)


def _flat_trace_rows() -> list[dict[str, float]]:
    return [
        {
            "t": 1_700_000_000 + i * 86_400,
            "token0": 1.0,
            "token1": 1.0,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "price_scale": 1.0,
            "price_oracle": 1.0,
            "profit": 0.0,
            "vp": 1.0,
            "vp_boosted": 1.0,
            "xcp": 1.0,
            "donation_apy": 0.0,
        }
        for i in range(3)
    ]


def test_flat_trace_has_zero_growth_without_donation(tmp_path: Path) -> None:
    trace_path = tmp_path / "detailed-output.json"
    trace_path.write_text(json.dumps(_flat_trace_rows()))

    trace = load_trace(trace_path, "price_scale", "profit")
    result = run_releverage(
        trace,
        SimulationConfig(
            fee=0.012,
            leverage=2.0,
            ext_fee=0.0,
            donation_apy=0.0,
            path_every=1,
        ),
    )

    assert result["final_growth"] == 1.0
    assert abs(result["apy"]) < 1e-12
    assert result["n_releverage_trades"] == 0
    assert len(result["path"]) == 3


def test_trace_donation_apy_is_loaded(tmp_path: Path) -> None:
    rows = _flat_trace_rows()
    for row in rows:
        row["donation_apy"] = 0.0475
    trace_path = tmp_path / "detailed-output.json"
    trace_path.write_text(json.dumps(rows))

    trace = load_trace(trace_path, "price_scale", "profit")

    assert trace.donation_apy == 0.0475


def test_fee_grid_single_and_scan_modes() -> None:
    assert make_fee_grid(0.012, 0.002, 0.05, 20, "log", scan=False) == [0.012]

    grid = make_fee_grid(None, 0.01, 0.03, 3, "linear", scan=False)
    assert grid == pytest.approx([0.01, 0.02, 0.03])
