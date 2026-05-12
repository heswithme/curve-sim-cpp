import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from thread_sweep import best_summary, parse_threads, resolve_start_time, run_one, summarize  # noqa: E402


def test_parse_threads_rejects_empty_and_nonpositive_values() -> None:
    assert parse_threads(" 10,12,14 ") == [10, 12, 14]

    for raw in ("", "0", "1,-2"):
        try:
            parse_threads(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_threads should reject {raw!r}")


def test_resolve_start_time_inherits_pool_config_meta() -> None:
    pool_config = {"meta": {"start_time": "1709638320"}}

    assert resolve_start_time(pool_config, None) == 1_709_638_320
    assert resolve_start_time(pool_config, "01-01-2024") == 1_704_067_200


def test_summarize_keeps_best_exec_and_speedup() -> None:
    rows = summarize(
        [
            {"threads": 1, "exec_ms": 100.0, "wall_ms": 120.0, "total_trades": 7},
            {"threads": 2, "exec_ms": 55.0, "wall_ms": 70.0, "total_trades": 7},
            {"threads": 2, "exec_ms": 50.0, "wall_ms": 65.0, "total_trades": 7},
        ]
    )

    assert rows[0]["threads"] == 1
    assert rows[0]["best_exec_ms"] == 100.0
    assert rows[1]["threads"] == 2
    assert rows[1]["best_exec_ms"] == 50.0
    assert rows[1]["speedup_vs_first"] == 2.0
    assert rows[1]["trades_per_second"] == 140.0
    assert rows[1]["ms_per_million_trades"] == 50.0 * 1_000_000 / 7

    assert best_summary(rows) == {
        "threads": 2,
        "best_exec_ms": 50.0,
        "best_wall_ms": 65.0,
        "total_trades": 7,
        "trades_per_second": 140.0,
        "ms_per_million_trades": 50.0 * 1_000_000 / 7,
    }


def test_run_one_prefers_explicit_loaded_candle_metadata(tmp_path: Path) -> None:
    class FakeRunner:
        def run(self, *_args, **_kwargs):
            return {
                "metadata": {
                    "exec_ms": 10.0,
                    "candles_read_ms": 1.0,
                    "n_pools": 3,
                    "n_candles_loaded": 20,
                    "candles": 999,
                    "events": 40,
                },
                "runs": [
                    {"result": {"trades": 2}},
                    {"result": {"trades": 4}},
                ],
            }

    args = type(
        "Args",
        (),
        {
            "n_candles": None,
            "min_swap": None,
            "max_swap": None,
            "dustswapfreq": 600,
            "userswapfreq": None,
            "userswapsize": None,
            "userswapthresh": None,
            "candle_filter": None,
            "start_ts": None,
            "disable_slippage_probes": True,
            "quiet_harness": True,
            "keep_raw": False,
        },
    )()

    row = run_one(
        runner=FakeRunner(),
        pool_config_path=tmp_path / "pools.json",
        candles_path=tmp_path / "candles.json",
        out_path=tmp_path / "raw.json",
        args=args,
        threads=2,
    )

    assert row["n_candles_loaded"] == 20
    assert row["total_trades"] == 6
