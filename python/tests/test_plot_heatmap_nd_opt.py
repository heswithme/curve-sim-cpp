import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

if not any(arg == "--out" or arg.startswith("--out=") for arg in sys.argv[1:]):
    sys.argv.append("--out=/tmp/test_plot_heatmap_nd_opt.png")

from plot_heatmap_nd_opt import (  # noqa: E402
    INSPECT_FULL_DETAILED_INTERVAL,
    INSPECT_SPARSE_DETAILED_INTERVAL,
    INSPECT_YB_FEE,
    NDHeatmapExplorerOpt,
    _extract_nd_arrays,
    _format_axis_labels,
    _format_slider_value,
    _stringify_pool,
)


def test_inspect_interval_is_sparse_unless_yb_is_requested() -> None:
    explorer = SimpleNamespace()

    assert INSPECT_FULL_DETAILED_INTERVAL == 1
    assert INSPECT_SPARSE_DETAILED_INTERVAL == 1000
    assert NDHeatmapExplorerOpt._inspect_detailed_interval(explorer, True) == 1
    assert NDHeatmapExplorerOpt._inspect_detailed_interval(explorer, False) == 1000


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


def test_apy_masked_uses_optional_7d_skew_filter() -> None:
    data = {
        "metadata": {
            "grid": {
                "x1": {"name": "A", "values": [1.0, 2.0]},
                "x2": {"name": "mid_fee", "values": [10.0]},
            }
        },
        "runs": [
            {
                "x1_val": 1.0,
                "x2_val": 10.0,
                "result": {
                    "apy_net": 0.05,
                    "max_7d_rel_price_diff": 0.01,
                    "max_7d_skew": 0.75,
                },
            },
            {
                "x1_val": 2.0,
                "x2_val": 10.0,
                "result": {
                    "apy_net": 0.06,
                    "max_7d_rel_price_diff": 0.01,
                    "max_7d_skew": 0.85,
                },
            },
        ],
    }

    _, _, unfiltered, *_ = _extract_nd_arrays(
        data,
        ["apy_masked"],
        max_price_thr_bps=2000.0,
        skew_thr_pct=0.0,
    )
    _, _, filtered, *_ = _extract_nd_arrays(
        data,
        ["apy_masked"],
        max_price_thr_bps=2000.0,
        skew_thr_pct=80.0,
    )

    assert unfiltered["apy_masked"][:, 0].tolist() == [5.0, 6.0]
    assert filtered["apy_masked"][0, 0] == 5.0
    assert np.isnan(filtered["apy_masked"][1, 0])


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


def test_heatmap_plain_left_click_does_not_run_inspect(monkeypatch) -> None:
    data = {
        "metadata": {
            "grid": {
                "x1": {"name": "A", "values": [10000.0, 20000.0]},
                "x2": {"name": "mid_fee", "values": [10.0, 20.0]},
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

    called = {"metrics": 0, "inspect": 0}
    monkeypatch.setattr(
        explorer,
        "_update_metrics_window",
        lambda *_args, **_kwargs: called.__setitem__("metrics", called["metrics"] + 1),
    )
    monkeypatch.setattr(
        explorer,
        "_build_inspect_pool_config",
        lambda *_args, **_kwargs: {"pool": {}, "costs": {}},
    )
    monkeypatch.setattr(
        explorer,
        "_run_inspect_simulation",
        lambda *_args, **_kwargs: called.__setitem__("inspect", called["inspect"] + 1),
    )

    event = SimpleNamespace(
        button=1,
        key=None,
        inaxes=explorer.axes[0],
        xdata=10000.0,
        ydata=10.0,
    )
    explorer._on_click(event)

    assert called == {"metrics": 1, "inspect": 0}


def test_heatmap_shift_left_runs_inspect_with_yb_right_click_without_yb(monkeypatch) -> None:
    data = {
        "metadata": {
            "grid": {
                "x1": {"name": "A", "values": [10000.0, 20000.0]},
                "x2": {"name": "mid_fee", "values": [10.0, 20.0]},
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

    called = {"metrics": 0, "inspect": 0, "yb_flags": []}
    monkeypatch.setattr(
        explorer,
        "_update_metrics_window",
        lambda *_args, **_kwargs: called.__setitem__("metrics", called["metrics"] + 1),
    )
    monkeypatch.setattr(
        explorer,
        "_build_inspect_pool_config",
        lambda *_args, **_kwargs: {"pool": {}, "costs": {}},
    )
    monkeypatch.setattr(
        explorer,
        "_run_inspect_simulation",
        lambda *_args, **kwargs: (
            called.__setitem__("inspect", called["inspect"] + 1),
            called["yb_flags"].append(kwargs.get("run_yb_releverage")),
        ),
    )

    for event in (
        SimpleNamespace(
            button=1,
            key="shift",
            inaxes=explorer.axes[0],
            xdata=10000.0,
            ydata=10.0,
        ),
        SimpleNamespace(
            button=3,
            key=None,
            inaxes=explorer.axes[0],
            xdata=10000.0,
            ydata=10.0,
        ),
    ):
        explorer._on_click(event)

    assert called == {"metrics": 0, "inspect": 2, "yb_flags": [True, False]}


def test_inspect_yb_releverage_uses_flat_one_percent_fee(monkeypatch, capsys) -> None:
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="best_fee=0.01 apy=4.1%\n")

    monkeypatch.setattr("plot_heatmap_nd_opt.subprocess.run", fake_run)

    NDHeatmapExplorerOpt._run_yb_releverage_simulation(
        SimpleNamespace(python_dir=Path("/tmp")),
        Path("/tmp/detailed-output.json"),
    )

    assert len(commands) == 1
    cmd, kwargs = commands[0]
    assert cmd[:3] == ["uv", "run", "arb_sim/yb_releverage.py"]
    assert "--scan" not in cmd
    assert cmd[cmd.index("--fee") + 1] == str(INSPECT_YB_FEE)
    assert cmd[cmd.index("--path-every") + 1] == "0"
    assert "--quiet" in cmd
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert capsys.readouterr().out.strip() == "YB releverage: best_fee=0.01 apy=4.1%"


def test_shift_inspect_uses_full_npz_and_right_inspect_uses_sparse_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candles_path = tmp_path / "candles.json"
    candles_path.write_text("[]")
    inspect_path = tmp_path / "inspect_pool.json"
    out_path = tmp_path / "inspect_output.json"

    commands = []
    popens = []
    yb_paths = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    def fake_popen(cmd, **_kwargs):
        popens.append(cmd)
        return SimpleNamespace()

    def make_explorer():
        explorer = SimpleNamespace(
            _inspect_running=False,
            candles_file=None,
            inspect_output_path=out_path,
            config_dustswapfreq=3600,
            start_time=None,
            config_start_time=None,
            config_disable_slippage_probes=False,
            _inspect_built_once=False,
            python_dir=tmp_path,
        )
        explorer._resolve_candles_path = lambda: candles_path
        explorer._write_inspect_pool_config = lambda _pool: inspect_path
        explorer._inspect_detailed_interval = (
            lambda run_yb: NDHeatmapExplorerOpt._inspect_detailed_interval(
                explorer, run_yb
            )
        )
        explorer._run_yb_releverage_simulation = lambda path: yb_paths.append(path)
        return explorer

    monkeypatch.setattr("plot_heatmap_nd_opt.subprocess.run", fake_run)
    monkeypatch.setattr("plot_heatmap_nd_opt.subprocess.Popen", fake_popen)

    NDHeatmapExplorerOpt._run_inspect_simulation(
        make_explorer(),
        {},
        run_yb_releverage=True,
    )
    shift_cmd = commands.pop()
    assert "--detailed-npz" in shift_cmd
    assert shift_cmd[shift_cmd.index("--detailed-interval") + 1] == "1"
    assert yb_paths == [tmp_path / "detailed-output.npz"]
    assert popens == []

    NDHeatmapExplorerOpt._run_inspect_simulation(
        make_explorer(),
        {},
        run_yb_releverage=False,
    )
    right_cmd = commands.pop()
    assert "--detailed-npz" not in right_cmd
    assert right_cmd[right_cmd.index("--detailed-interval") + 1] == "1000"
    assert yb_paths == [tmp_path / "detailed-output.npz"]
    assert popens == [
        [
            "uv",
            "run",
            "arb_sim/plot_price_scale.py",
            "--no-save",
            str(tmp_path / "detailed-output.json"),
        ]
    ]
