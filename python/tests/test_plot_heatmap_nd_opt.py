import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

if not any(arg == "--out" or arg.startswith("--out=") for arg in sys.argv[1:]):
    sys.argv.append("--out=/tmp/test_plot_heatmap_nd_opt.png")

from plot_heatmap_nd_opt import (  # noqa: E402
    NDHeatmapExplorerOpt,
    _extract_nd_arrays,
    _format_axis_labels,
    _format_slider_value,
    _stringify_pool,
)


def test_stringify_pool_preserves_nested_policy_object() -> None:
    pool = {
        "A": 10000,
        "policy": {"kind": "fixed_fee", "fee_bps": 77.41935483870968},
    }

    assert _stringify_pool(pool) == {
        "A": "10000",
        "policy": {"kind": "fixed_fee", "fee_bps": "77.41935483870968"},
    }


def test_reserved_profit_fraction_axis_displays_as_fraction() -> None:
    labels, display_name = _format_axis_labels(
        "reserved_profit_fraction",
        [1_900_000_000.0, 5_000_000_000.0],
    )

    assert labels == ["0.19", "0.50"]
    assert display_name == "reserved_profit_fraction (÷1e10)"
    assert _format_slider_value("reserved_profit_fraction", 1_900_000_000.0) == "0.1900"


def test_extract_nd_arrays_infers_raw_harness_params_pool_axes() -> None:
    runs = []
    for a in ("10000", "20000"):
        for fee_bps in (10.0, 20.0):
            for donation_apy in (0.0, 0.2):
                runs.append(
                    {
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
                        "result": {"score": float(a) + fee_bps + donation_apy},
                        "final_state": {},
                    }
                )

    dim_names, dim_values, metric_arrays, *_ = _extract_nd_arrays(
        {"runs": runs},
        ["score"],
        price_thr_bps=1.0,
        imbalance_thr_pct=0.0,
    )

    assert dim_names == ["A", "policy.fee_bps", "donation_apy"]
    assert "policy.fee" not in dim_names
    assert dim_values == {
        "A": [10000.0, 20000.0],
        "policy.fee_bps": [10.0, 20.0],
        "donation_apy": [0.0, 0.2],
    }
    assert metric_arrays["score"].shape == (2, 2, 2)
    assert metric_arrays["score"][0, 0, 0] == 10010.0
    assert metric_arrays["score"][1, 1, 1] == 20020.2


def test_extract_nd_arrays_reads_cluster_flat_dotted_policy_axis() -> None:
    data = {
        "metadata": {
            "grid": {
                "x1": {"name": "A", "values": [10000.0, 20000.0]},
                "x2": {"name": "policy.fee_bps", "values": [10.0, 20.0]},
            }
        },
        "runs": [
            {
                "pool": {"A": str(a), "policy.fee_bps": fee_bps},
                "result": {"score": float(a) + fee_bps},
            }
            for a in (10000, 20000)
            for fee_bps in (10.0, 20.0)
        ],
    }

    dim_names, dim_values, metric_arrays, *_ = _extract_nd_arrays(
        data,
        ["score"],
        price_thr_bps=1.0,
        imbalance_thr_pct=0.0,
    )

    assert dim_names == ["A", "policy.fee_bps"]
    assert dim_values == {
        "A": [10000.0, 20000.0],
        "policy.fee_bps": [10.0, 20.0],
    }
    assert metric_arrays["score"].shape == (2, 2)
    assert metric_arrays["score"][1, 1] == 20020.0


def test_explorer_reads_inspect_flags_from_cluster_harness_args() -> None:
    data = {
        "metadata": {
            "grid": {
                "x1": {"name": "A", "values": [10000.0, 20000.0]},
                "x2": {"name": "mid_fee", "values": [10.0, 20.0]},
            },
            "harness_args": {
                "start_time": "1704067200",
                "disable_slippage_probes": True,
            },
        },
        "runs": [
            {
                "pool": {"A": str(a), "mid_fee": fee},
                "result": {"score": float(a) + fee},
            }
            for a in (10000, 20000)
            for fee in (10.0, 20.0)
        ],
    }

    explorer = NDHeatmapExplorerOpt(
        data,
        ["score"],
        ncol=1,
        cmap="viridis",
        max_ticks=8,
        clamp=False,
        price_thr_bps=1.0,
        max_price_thr_bps=1.0,
    )

    assert explorer.config_start_time == "1704067200"
    assert explorer.config_disable_slippage_probes is True
