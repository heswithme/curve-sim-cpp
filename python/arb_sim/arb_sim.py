#!/usr/bin/env python3
"""
Arbitrage runner for the C++ arb_harness (multi-pool, threaded in C++).

- Reads pools from python/arb_sim/run_data/pool_config.json (or --pool-config).
- Calls the C++ harness once with all pools; C++ handles internal threading.
- Emits an arb_npz_v1 run directory by default.
- Supports N-dimensional grids via meta.grid.dims (falls back to X/Y).
"""

import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone


def parse_start_time(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    value = str(value)
    if value.isdigit():
        return int(value)
    if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", value):
        raise ValueError("--start-time must be a Unix timestamp or DD-MM-YYYY")
    dt = datetime.strptime(value, "%d-%m-%Y").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class ArbHarnessRunner:
    def __init__(
        self,
        repo_root: Path,
        real: str = "double",
        exe_path: str | Path | None = None,
    ):
        self.repo_root = Path(repo_root)
        self.cpp_dir = self.repo_root / "cpp_old"
        self.cpp_dir = self.repo_root / "cpp_modular"  # modular harness
        self.build_dir = self.cpp_dir / "build"
        self.custom_exe = exe_path is not None
        if self.custom_exe:
            self.exe_path = Path(exe_path).expanduser()
            if not self.exe_path.is_absolute():
                self.exe_path = self.repo_root / self.exe_path
            self.target = self.exe_path.name
            return

        # Resolve binary name based on real type
        real = (real or "double").lower()
        if real in ("float", "f"):
            self.target = "arb_harness_f"
        elif real in ("longdouble", "long_double", "ld", "long"):
            self.target = "arb_harness_ld"
        else:
            self.target = "arb_harness"
        self.exe_path = self.build_dir / self.target

    def configure_build(self):
        self.build_dir.mkdir(parents=True, exist_ok=True)
        print("Configuring C++ build (Release)...")
        r = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=self.build_dir,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("CMake configure failed")

    def build(self):
        if self.custom_exe:
            if not self.exe_path.exists():
                raise FileNotFoundError(f"Missing executable: {self.exe_path}")
            print(f"Using custom C++ harness: {self.exe_path}")
            return

        self.configure_build()
        print(f"Building {self.target}...")
        r = subprocess.run(
            ["cmake", "--build", ".", "--target", self.target],
            cwd=self.build_dir,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("Build failed")
        if not self.exe_path.exists():
            raise FileNotFoundError(f"Missing executable: {self.exe_path}")
        print(f"✓ Built: {self.exe_path}")

    def run(
        self,
        pools_json: Path,
        candles_path: Path,
        out_path: Path,
        n_candles: int = 0,
        save_actions: bool = False,
        min_swap: float = 1e-10,
        max_swap: float = 1.0,
        threads: int = 1,
        dustswapfreq: int | None = 3600,
        userswapfreq: int | None = None,
        userswapsize: float | None = None,
        userswapthresh: float | None = None,
        detailed_log: bool = False,
        detailed_npz: bool = False,
        detailed_interval: int | None = None,
        cowswap_trades: str | None = None,
        cowswap_fee_bps: float | None = None,
        candle_filter: float | None = None,
        start_time: int | None = None,
        disable_slippage_probes: bool = False,
        quiet_harness: bool = False,
    ) -> Dict[str, Any]:
        print("Running arb_harness...", flush=True)
        cmd = [
            str(self.exe_path),
            str(pools_json),
            str(candles_path),
            str(out_path),
        ]
        if n_candles and n_candles > 0:
            cmd += ["--n-candles", str(n_candles)]
        if save_actions:
            cmd += ["--save-actions"]

        if min_swap is not None:
            cmd += ["--min-swap", str(min_swap)]
        if max_swap is not None:
            cmd += ["--max-swap", str(max_swap)]
        # Always pass threads to ensure explicit control (even when 1)
        cmd += ["--threads", str(max(1, threads))]
        if dustswapfreq is not None:
            cmd += ["--dustswapfreq", str(int(dustswapfreq))]
        if userswapfreq is not None:
            cmd += ["--userswapfreq", str(int(userswapfreq))]
        if userswapsize is not None:
            cmd += ["--userswapsize", str(userswapsize)]
        if userswapthresh is not None:
            cmd += ["--userswapthresh", str(userswapthresh)]
        if detailed_log:
            cmd += ["--detailed-log"]
        if detailed_npz:
            cmd += ["--detailed-npz"]
        if detailed_interval is not None:
            cmd += ["--detailed-interval", str(int(detailed_interval))]
        if cowswap_trades:
            cmd += ["--cowswap-trades", str(cowswap_trades)]
        if cowswap_fee_bps is not None:
            cmd += ["--cowswap-fee-bps", str(cowswap_fee_bps)]
        if candle_filter is not None:
            cmd += ["--candle-filter", str(candle_filter)]
        if start_time is not None:
            cmd += ["--start-time", str(start_time)]
        if disable_slippage_probes:
            cmd += ["--disable-slippage-probes"]
        if quiet_harness:
            cmd += ["--quiet"]
        # Stream harness stdout/stderr directly to the console for live progress
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise RuntimeError("arb_harness failed")
        print(f"✓ Results: {out_path}", flush=True)
        manifest_path = out_path / "manifest.json" if out_path.is_dir() else out_path
        with open(manifest_path, "r") as f:
            return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run C++ multi-pool arbitrage harness over candle data"
    )
    parser.add_argument(
        "candles",
        nargs="?",
        default=None,
        type=str,
        help="Path to candles JSON; if omitted, use pool_config meta.datafile",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output run directory (or legacy JSON path ending in .json)",
    )
    parser.add_argument(
        "--pool-config",
        type=str,
        default=None,
        help="Path to pool_config.json (default: run_data/pool_config.json)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip C++ build if binary exists",
    )
    parser.add_argument(
        "--harness-exe",
        type=str,
        default=None,
        help="Use an explicit arb_harness binary instead of cpp_modular/build",
    )
    parser.add_argument(
        "--n-candles",
        type=int,
        default=0,
        help="Limit to first N candles (default: all)",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Skip candles before Unix timestamp or DD-MM-YYYY at 00:00 UTC",
    )
    parser.add_argument(
        "--save-actions",
        action="store_true",
        help="Ask C++ harness to save executed trades/actions",
    )
    parser.add_argument(
        "--min-swap",
        type=float,
        default=1e-6,
        help="Minimum swap fraction of from-side balance (default: 1e-6)",
    )
    parser.add_argument(
        "--max-swap",
        type=float,
        default=1.0,
        help="Maximum swap fraction of from-side balance (default: 1.0)",
    )
    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=1,
        help="Threads in C++ harness (default: 1)",
    )
    parser.add_argument(
        "--real",
        type=str,
        default="double",
        choices=["float", "double", "longdouble"],
        help="Numeric precision for C++ harness",
    )
    parser.add_argument(
        "--dustswapfreq",
        type=int,
        default=3600,
        help="Seconds between dust swaps when no arb trade (default: 3600)",
    )

    parser.add_argument(
        "--candle-filter",
        type=float,
        default=99.0,
        help="Filter candles +/- PCT (default: 99.0)",
    )
    parser.add_argument(
        "--userswapfreq",
        type=int,
        default=0,
        help="Seconds between synthetic user swaps (default: 0)",
    )
    parser.add_argument(
        "--userswapsize",
        type=float,
        default=0,
        help="Fraction of from-side balance per user swap (default: 0)",
    )
    parser.add_argument(
        "--userswapthresh",
        type=float,
        default=0,
        help="Max relative deviation vs CEX to allow user swap",
    )
    parser.add_argument(
        "--detailed-log",
        action="store_true",
        help="Write detailed-output.json next to output file",
    )
    parser.add_argument(
        "--detailed-npz",
        action="store_true",
        help="Write detailed output as detailed-output.npz instead of JSON",
    )
    parser.add_argument(
        "--detailed-interval",
        type=int,
        default=None,
        help="Log every N-th candle in detailed output (default: 1 = all)",
    )
    parser.add_argument(
        "--cow",
        action="store_true",
        help="Enable cowswap organic trade replay (uses cowswap_file from pool_config)",
    )
    parser.add_argument(
        "--disable-slippage-probes",
        action="store_true",
        help="Disable slippage probes in the C++ harness",
    )
    parser.add_argument(
        "--quiet-harness",
        action="store_true",
        help="Suppress C++ harness progress logs",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runner = ArbHarnessRunner(repo_root, real=args.real, exe_path=args.harness_exe)
    if args.skip_build:
        if not runner.exe_path.exists():
            runner.build()
    else:
        runner.build()

    # Load pool_config.json
    pool_config_path = (
        Path(args.pool_config)
        if args.pool_config
        else (repo_root / "python" / "arb_sim" / "run_data" / "pool_config.json")
    )
    if not pool_config_path.is_absolute():
        pool_config_path = repo_root / pool_config_path
    if not pool_config_path.exists():
        raise FileNotFoundError(f"Missing pool config: {pool_config_path}")
    with open(pool_config_path, "r") as f:
        cfg = json.load(f)

    def resolve_candles_path() -> Path:
        if args.candles:
            return Path(args.candles)
        meta = cfg.get("meta") if isinstance(cfg, dict) else None
        candidate = None
        if isinstance(meta, dict):
            candidate = meta.get("datafile")
        if not candidate:
            raise ValueError(
                "Candles path not provided and meta.datafile missing in pool_config.json"
            )
        cand_path = Path(candidate)
        if not cand_path.is_absolute():
            cand_path = repo_root / cand_path
        return cand_path

    candles_path = resolve_candles_path()
    if not candles_path.exists():
        raise FileNotFoundError(f"Candles file not found: {candles_path}")
    meta = cfg.get("meta") if isinstance(cfg, dict) else None
    config_start_time = meta.get("start_time") if isinstance(meta, dict) else None
    start_ts = parse_start_time(args.start_time or config_start_time)

    # Resolve cowswap trades path if --cow enabled
    cowswap_path: str | None = None
    cowswap_fee_bps: float | None = None
    if args.cow:
        cowswap_file = meta.get("cowswap_file") if isinstance(meta, dict) else None
        if cowswap_file:
            cpath = Path(cowswap_file)
            if not cpath.is_absolute():
                cpath = repo_root / cpath
            if not cpath.exists():
                raise FileNotFoundError(f"Cowswap file not found: {cpath}")
            cowswap_path = str(cpath)
        else:
            raise ValueError(
                "--cow specified but meta.cowswap_file missing in pool_config.json"
            )
        # Get fee from config (default 0 if not specified)
        cowswap_fee_bps = (
            meta.get("cowswap_fee_bps", 0.0) if isinstance(meta, dict) else 0.0
        )

    # Invoke the harness once over the entire config
    out_path = (
        Path(args.out)
        if args.out
        else (repo_root / "python" / "arb_sim" / "run_data" / "arb_run_1")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_json_output = out_path.suffix == ".json"
    run_started_at = datetime.now(timezone.utc)
    raw = runner.run(
        pool_config_path,
        candles_path,
        out_path,
        n_candles=args.n_candles,
        save_actions=args.save_actions,
        min_swap=args.min_swap,
        max_swap=args.max_swap,
        threads=max(1, args.threads),
        dustswapfreq=args.dustswapfreq,
        userswapfreq=args.userswapfreq,
        userswapsize=args.userswapsize,
        userswapthresh=args.userswapthresh,
        detailed_log=args.detailed_log,
        detailed_npz=args.detailed_npz,
        detailed_interval=args.detailed_interval,
        cowswap_trades=cowswap_path,
        cowswap_fee_bps=cowswap_fee_bps,
        candle_filter=args.candle_filter,
        start_time=start_ts,
        disable_slippage_probes=args.disable_slippage_probes,
        quiet_harness=args.quiet_harness,
    )
    harness_done_at = datetime.now(timezone.utc)
    harness_wall_s = (harness_done_at - run_started_at).total_seconds()

    runs_raw: List[Dict[str, Any]] = raw.get("runs", [])
    print(f"Time taken: {harness_wall_s} seconds")

    # Derive grid dimension names from pool_config meta (x1..xn format)
    def _grid_dim_names(grid: Dict[str, Any]) -> List[str]:
        if not isinstance(grid, dict):
            return []
        # Parse x1, x2, ..., xN keys
        numbered = []
        for key, val in grid.items():
            if not isinstance(key, str):
                continue
            if key.lower().startswith("x") and key[1:].isdigit():
                name = val.get("name") if isinstance(val, dict) else None
                if name:
                    numbered.append((int(key[1:]), name))
        if numbered:
            return [name for _, name in sorted(numbered)]
        return []

    def _get_dotted(obj: Dict[str, Any], path: str):
        cur: Any = obj
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def get_meta(conf: Dict[str, Any]):
        meta = conf.get("meta", {}) if isinstance(conf, dict) else {}
        grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
        dim_names = _grid_dim_names(grid)
        base_pool = meta.get("base_pool") if isinstance(meta, dict) else None
        base_costs = meta.get("base_costs") if isinstance(meta, dict) else None
        return dim_names, base_pool, base_costs

    dim_names, base_pool_meta, base_costs_meta = get_meta(cfg)
    grid_meta = cfg.get("meta", {}).get("grid", {}) if isinstance(cfg, dict) else {}

    def _grid_axis_values() -> List[List[Any]]:
        values: List[List[Any]] = []
        for idx in range(1, len(dim_names) + 1):
            dim = grid_meta.get(f"x{idx}") if isinstance(grid_meta, dict) else None
            raw = dim.get("values") if isinstance(dim, dict) else None
            if not isinstance(raw, list) or not raw:
                return []
            values.append(raw)
        return values

    axis_values = _grid_axis_values()

    def _coords_from_grid_index(run_idx: int) -> List[Any] | None:
        if not axis_values:
            return None
        coords: List[Any] = []
        stride = 1
        strides: List[int] = []
        for vals in reversed(axis_values[1:]):
            stride *= len(vals)
            strides.append(stride)
        strides = list(reversed(strides)) + [1]

        for vals, axis_stride in zip(axis_values, strides):
            coords.append(vals[(run_idx // axis_stride) % len(vals)])
        return coords

    # Derive base_pool from actual pools if meta missing
    def pools_list():
        if isinstance(cfg, dict) and "pools" in cfg:
            return cfg["pools"]
        elif isinstance(cfg, dict) and "pool" in cfg:
            return [{"pool": cfg["pool"], "costs": cfg.get("costs", {})}]
        return []

    plist = list(pools_list())
    base_pool: Dict[str, Any] = {}
    if not base_pool_meta:

        def to_strish(v):
            if isinstance(v, list):
                return [str(x) for x in v]
            return str(v)

        if plist:
            keys = set(plist[0].get("pool", {}).keys())
            for e in plist[1:]:
                keys &= set(e.get("pool", {}).keys())
            for k in sorted(keys):
                # Exclude all varying dim_names from base_pool
                if k in dim_names:
                    continue
                vals = [to_strish(e.get("pool", {}).get(k)) for e in plist]
                if all(v == vals[0] for v in vals):
                    base_pool[k] = vals[0]
    else:
        base_pool = base_pool_meta

    base_costs = base_costs_meta if isinstance(base_costs_meta, dict) else {}

    if not legacy_json_output:
        raw_metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        postprocess_ms = (
            datetime.now(timezone.utc) - harness_done_at
        ).total_seconds() * 1000.0
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.update(
            {
                "candles_file": str(candles_path),
                "pool_config_file": str(pool_config_path),
                "harness_exe": str(runner.exe_path),
                "real": args.real,
                "run_started_at": run_started_at.isoformat(),
                "threads": max(1, args.threads),
                "quiet_harness": args.quiet_harness,
                "n_candles_requested": args.n_candles,
                "start_time": start_ts,
                "disable_slippage_probes": args.disable_slippage_probes,
                "save_actions": args.save_actions,
                "min_swap": args.min_swap,
                "max_swap": args.max_swap,
                "dustswapfreq": args.dustswapfreq,
                "userswapfreq": args.userswapfreq,
                "userswapsize": args.userswapsize,
                "userswapthresh": args.userswapthresh,
                "candle_filter": args.candle_filter,
                "cowswap_enabled": args.cow,
                "cowswap_file": cowswap_path,
                "cowswap_fee_bps": cowswap_fee_bps,
                "base_pool": base_pool,
                "base_costs": base_costs,
                "grid": grid_meta,
                "fee_equalize": meta.get("fee_equalize") if isinstance(meta, dict) else False,
                "postprocess_ms": postprocess_ms,
                "harness_wall_ms": harness_wall_s * 1000.0,
                "wall_ms": harness_wall_s * 1000.0 + postprocess_ms,
            }
        )
        raw["metadata"] = metadata
        raw["format"] = raw.get("format", "arb_npz_v1")
        manifest_path = out_path / "manifest.json"
        manifest_path.write_text(json.dumps(raw, indent=2))
        print(f"\n✓ Wrote NPZ run: {out_path}")
        return 0

    # Enrich runs with x1_key/x1_val, x2_key/x2_val, ..., xN_key/xN_val
    enriched_runs: List[Dict[str, Any]] = []
    total_trades = 0
    for run_idx, rr in enumerate(runs_raw):
        params_obj = rr.get("params", {}) or {}
        pool_obj = params_obj.get("pool", {}) if isinstance(params_obj, dict) else {}
        try:
            coord_idx = int(rr.get("pool_index", run_idx))
        except Exception:
            coord_idx = run_idx
        grid_coords = _coords_from_grid_index(coord_idx)
        # Accumulate total trades for metadata (no duplicate field in result)
        result_obj = rr.get("result", {}) or {}
        try:
            total_trades += int(result_obj.get("trades", 0))
        except Exception:
            pass

        enriched: Dict[str, Any] = {}
        # Add x1_key/x1_val, x2_key/x2_val, etc. for each dimension
        for idx, name in enumerate(dim_names, start=1):
            val_obj = _get_dotted(pool_obj, name) if name else None
            if val_obj is None and grid_coords is not None:
                val_obj = grid_coords[idx - 1]
            val = str(val_obj) if val_obj is not None else None
            enriched[f"x{idx}_key"] = name
            enriched[f"x{idx}_val"] = val

        enriched["result"] = result_obj
        enriched["final_state"] = rr.get("final_state", {})
        if params_obj:
            # Preserve full per-run config so downstream tools (inspect, drill-down)
            # can reconstruct the exact pool/cost setup for a selected point.
            enriched["params"] = params_obj
        if "actions" in rr:
            enriched["actions"] = rr.get("actions")
        if "states" in rr:
            enriched["states"] = rr.get("states")
        enriched_runs.append(enriched)

    raw_metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    raw_events = raw_metadata.get("events") if isinstance(raw_metadata, dict) else None
    candles_loaded = (
        raw_metadata.get("n_candles_loaded", raw_metadata.get("candles"))
        if isinstance(raw_metadata, dict)
        else None
    )
    if candles_loaded is None and isinstance(raw_events, int):
        candles_loaded = raw_events // 2

    postprocess_ms = (datetime.now(timezone.utc) - harness_done_at).total_seconds() * 1000.0
    agg = {
        "metadata": {
            "candles_file": str(candles_path),
            "pool_config_file": str(pool_config_path),
            "harness_exe": str(runner.exe_path),
            "real": args.real,
            "run_started_at": run_started_at.isoformat(),
            "threads": max(1, args.threads),
            "quiet_harness": args.quiet_harness,
            "n_pools": raw_metadata.get("n_pools", len(runs_raw)),
            "n_candles_requested": args.n_candles,
            "n_candles_loaded": candles_loaded,
            "events": raw_events,
            "start_time": start_ts,
            "disable_slippage_probes": args.disable_slippage_probes,
            "save_actions": args.save_actions,
            "min_swap": args.min_swap,
            "max_swap": args.max_swap,
            "dustswapfreq": args.dustswapfreq,
            "userswapfreq": args.userswapfreq,
            "userswapsize": args.userswapsize,
            "userswapthresh": args.userswapthresh,
            "candle_filter": args.candle_filter,
            "cowswap_enabled": args.cow,
            "cowswap_file": cowswap_path,
            "cowswap_fee_bps": cowswap_fee_bps,
            "base_pool": base_pool,
            "grid": cfg.get("meta", {}).get("grid") if isinstance(cfg, dict) else None,
            "candles_read_ms": raw_metadata.get("candles_read_ms"),
            "exec_ms": raw_metadata.get("exec_ms"),
            "harness_wall_ms": harness_wall_s * 1000.0,
            "postprocess_ms": postprocess_ms,
            "wall_ms": harness_wall_s * 1000.0 + postprocess_ms,
            "total_trades": total_trades,
        },
        "runs": enriched_runs,
    }
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n✓ Wrote aggregated run: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
