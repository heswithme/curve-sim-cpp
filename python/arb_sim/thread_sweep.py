#!/usr/bin/env python3
"""Run repeatable arb_harness thread-count sweeps.

This is intentionally a thin wrapper around the same C++ harness path used by
arb_sim.py. It exists so local throughput checks can be reproduced without
hand-written shell loops and so each run records the exact pool/candle workload.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arb_sim import ArbHarnessRunner, parse_start_time


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUN_DATA_DIR = SCRIPT_DIR / "run_data"


def parse_threads(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("thread counts must be positive")
        out.append(value)
    if not out:
        raise ValueError("at least one thread count is required")
    return out


def resolve_pool_config(raw: str | None) -> Path:
    path = Path(raw) if raw else RUN_DATA_DIR / "pool_config.json"
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing pool config: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return root


def resolve_candles_path(pool_config: dict[str, Any], raw: str | None) -> Path:
    if raw:
        path = Path(raw)
    else:
        meta = pool_config.get("meta") if isinstance(pool_config, dict) else None
        datafile = meta.get("datafile") if isinstance(meta, dict) else None
        if not datafile:
            raise ValueError("Candles path not provided and meta.datafile missing")
        path = Path(datafile)

    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Candles file not found: {path}")
    return path


def resolve_start_time(pool_config: dict[str, Any], raw: str | None) -> int | None:
    if raw:
        return parse_start_time(raw)
    meta = pool_config.get("meta") if isinstance(pool_config, dict) else None
    config_start = meta.get("start_time") if isinstance(meta, dict) else None
    return parse_start_time(str(config_start)) if config_start is not None else None


def total_trades(raw: dict[str, Any]) -> int:
    total = 0
    for run in raw.get("runs", []):
        if not isinstance(run, dict):
            continue
        result = run.get("result")
        if not isinstance(result, dict):
            continue
        total += int(result.get("trades", 0))
    return total


def run_one(
    *,
    runner: ArbHarnessRunner,
    pool_config_path: Path,
    candles_path: Path,
    out_path: Path,
    args: argparse.Namespace,
    threads: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = runner.run(
        pool_config_path,
        candles_path,
        out_path,
        n_candles=args.n_candles,
        min_swap=args.min_swap,
        max_swap=args.max_swap,
        threads=threads,
        dustswapfreq=args.dustswapfreq,
        userswapfreq=args.userswapfreq,
        userswapsize=args.userswapsize,
        userswapthresh=args.userswapthresh,
        candle_filter=args.candle_filter,
        start_time=args.start_ts,
        disable_slippage_probes=args.disable_slippage_probes,
        quiet_harness=args.quiet_harness,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}

    return {
        "threads": threads,
        "wall_ms": wall_ms,
        "exec_ms": metadata.get("exec_ms"),
        "candles_read_ms": metadata.get("candles_read_ms"),
        "n_pools": metadata.get("n_pools"),
        "n_candles_loaded": metadata.get("n_candles_loaded", metadata.get("candles")),
        "events": metadata.get("events"),
        "total_trades": total_trades(raw),
        "raw_output": str(out_path) if args.keep_raw else None,
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_threads: dict[int, list[dict[str, Any]]] = {}
    for item in results:
        by_threads.setdefault(int(item["threads"]), []).append(item)

    rows: list[dict[str, Any]] = []
    for threads, items in sorted(by_threads.items()):
        exec_values = [float(x["exec_ms"]) for x in items if x.get("exec_ms") is not None]
        wall_values = [float(x["wall_ms"]) for x in items]
        best_exec_ms = min(exec_values) if exec_values else None
        total_trades = items[-1].get("total_trades")
        trades_per_second = None
        ms_per_million_trades = None
        if best_exec_ms and total_trades:
            trades_per_second = float(total_trades) / (best_exec_ms / 1000.0)
            ms_per_million_trades = best_exec_ms * 1_000_000.0 / float(total_trades)
        rows.append(
            {
                "threads": threads,
                "runs": len(items),
                "best_exec_ms": best_exec_ms,
                "mean_exec_ms": sum(exec_values) / len(exec_values) if exec_values else None,
                "best_wall_ms": min(wall_values),
                "mean_wall_ms": sum(wall_values) / len(wall_values),
                "total_trades": total_trades,
                "trades_per_second": trades_per_second,
                "ms_per_million_trades": ms_per_million_trades,
            }
        )
    if rows and rows[0].get("best_exec_ms"):
        baseline = float(rows[0]["best_exec_ms"])
        for row in rows:
            best = row.get("best_exec_ms")
            row["speedup_vs_first"] = baseline / float(best) if best else None
    return rows


def best_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("best_exec_ms") is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: float(row["best_exec_ms"]))
    return {
        "threads": best["threads"],
        "best_exec_ms": best["best_exec_ms"],
        "best_wall_ms": best["best_wall_ms"],
        "total_trades": best.get("total_trades"),
        "trades_per_second": best.get("trades_per_second"),
        "ms_per_million_trades": best.get("ms_per_million_trades"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candles", nargs="?", default=None, help="Candles JSON path")
    p.add_argument("--pool-config", default=None, help="Pool config JSON path")
    p.add_argument(
        "--out",
        type=Path,
        default=RUN_DATA_DIR / "thread_sweep.json",
        help="Summary JSON path",
    )
    p.add_argument(
        "--threads",
        default="1,2,4,8,10,12,14",
        help="Comma-separated thread counts",
    )
    p.add_argument("--repeat", type=int, default=1, help="Runs per thread count")
    p.add_argument("--real", choices=["float", "double", "longdouble"], default="double")
    p.add_argument(
        "--harness-exe",
        default=None,
        help="Use an explicit arb_harness binary instead of cpp_modular/build",
    )
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--n-candles", type=int, default=0)
    p.add_argument("--start-time", type=str, default=None)
    p.add_argument("--min-swap", type=float, default=1e-6)
    p.add_argument("--max-swap", type=float, default=1.0)
    p.add_argument("--dustswapfreq", type=int, default=None)
    p.add_argument("--userswapfreq", type=int, default=0)
    p.add_argument("--userswapsize", type=float, default=0)
    p.add_argument("--userswapthresh", type=float, default=0)
    p.add_argument("--candle-filter", type=float, default=99.0)
    p.add_argument("--disable-slippage-probes", action="store_true")
    p.add_argument(
        "--quiet-harness",
        action="store_true",
        help="Suppress C++ harness progress logs during each timing run",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep per-run raw harness JSON files next to the summary",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    pool_config_path = resolve_pool_config(args.pool_config)
    pool_config = load_json(pool_config_path)
    candles_path = resolve_candles_path(pool_config, args.candles)
    args.start_ts = resolve_start_time(pool_config, args.start_time)
    thread_counts = parse_threads(args.threads)

    runner = ArbHarnessRunner(REPO_ROOT, real=args.real, exe_path=args.harness_exe)
    if args.skip_build:
        if not runner.exe_path.exists():
            runner.build()
    else:
        runner.build()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir_ctx = None
    if args.keep_raw:
        raw_dir = args.out.with_suffix("")
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir_ctx = tempfile.TemporaryDirectory(prefix="arb_thread_sweep_")
        raw_dir = Path(raw_dir_ctx.name)

    results: list[dict[str, Any]] = []
    try:
        for threads in thread_counts:
            for rep in range(args.repeat):
                raw_path = raw_dir / f"threads_{threads}_rep_{rep}.json"
                print(
                    f"\n=== threads={threads} repeat={rep + 1}/{args.repeat} ===",
                    flush=True,
                )
                result = run_one(
                    runner=runner,
                    pool_config_path=pool_config_path,
                    candles_path=candles_path,
                    out_path=raw_path,
                    args=args,
                    threads=threads,
                )
                results.append(result)
    finally:
        if raw_dir_ctx is not None:
            raw_dir_ctx.cleanup()

    summary_rows = summarize(results)
    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pool_config_file": str(pool_config_path),
            "candles_file": str(candles_path),
            "harness_exe": str(runner.exe_path),
            "real": args.real,
            "threads": thread_counts,
            "repeat": args.repeat,
            "n_candles_requested": args.n_candles,
            "start_time": args.start_ts,
            "dustswapfreq": args.dustswapfreq,
            "disable_slippage_probes": args.disable_slippage_probes,
            "quiet_harness": args.quiet_harness,
            "min_swap": args.min_swap,
            "max_swap": args.max_swap,
            "keep_raw": args.keep_raw,
        },
        "best": best_summary(summary_rows),
        "summary": summary_rows,
        "runs": results,
    }

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote thread sweep summary: {args.out}")
    for row in summary["summary"]:
        speedup = row.get("speedup_vs_first")
        speedup_s = f"{speedup:.3f}x" if speedup is not None else "n/a"
        best_exec = row.get("best_exec_ms")
        best_exec_s = f"{best_exec:.3f}" if best_exec is not None else "n/a"
        tps = row.get("trades_per_second")
        tps_s = f"{tps:,.0f}/s" if tps is not None else "n/a"
        ms_per_m = row.get("ms_per_million_trades")
        ms_per_m_s = f"{ms_per_m:.3f}" if ms_per_m is not None else "n/a"
        print(
            f"threads={row['threads']:>2} "
            f"best_exec_ms={best_exec_s} "
            f"best_wall_ms={row['best_wall_ms']:.3f} "
            f"speedup={speedup_s} "
            f"trades_per_second={tps_s} "
            f"ms_per_million_trades={ms_per_m_s}"
        )
    if summary["best"] is not None:
        best = summary["best"]
        print(
            f"best_threads={best['threads']} "
            f"best_exec_ms={best['best_exec_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
