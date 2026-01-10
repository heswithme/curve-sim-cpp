#!/usr/bin/env python3
"""
Main orchestration script for cluster sweeps.

Pipeline:
1. BUILD      - Compile once on any blade (shared via NFS)
2. DISTRIBUTE - Upload candles (from pool config) + pool batches once
3. RUN        - Execute on all blades in parallel
4. COLLECT    - Download and merge results

Usage:
    # Full sweep (reads candles path from pool config)
    python orchestrate.py --pools pools.json

    # Skip build (already compiled)
    python orchestrate.py --pools pools.json --skip-build

    # Build only
    python orchestrate.py --build-only

    # Resume/collect from previous job
    python orchestrate.py --resume jobs/20260110_123456/manifest.json
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import (
    DEFAULT_BLADES,
    BLADES_WORKING,
    CORES_PER_BLADE,
    LOCAL_RESULTS_DIR,
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
)
from build import build
from distribute import distribute
from run import run
from collect import collect


def check_connectivity(blades: List[str]) -> List[str]:
    """Return list of reachable blades."""
    print("Checking blade connectivity...")
    reachable = []

    for blade in blades:
        cmd = (
            ["ssh"]
            + SSH_OPTIONS
            + [
                "-i",
                str(SSH_KEY),
                "-o",
                "ConnectTimeout=10",
                f"{SSH_USER}@{blade}",
                "echo OK",
            ]
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if "OK" in result.stdout:
                reachable.append(blade)
                print(f"  {blade}: OK")
            else:
                print(f"  {blade}: UNREACHABLE")
        except Exception as e:
            print(f"  {blade}: ERROR ({e})")

    return reachable


def sweep(
    pools_file: Path,
    blades: List[str],
    skip_build: bool = False,
    candles_file: Optional[Path] = None,
    threads: int = CORES_PER_BLADE,
    dustswap_freq: int = 600,
    apy_period_days: float = 1.0,
    apy_period_cap: int = 20,
    candle_filter: Optional[float] = None,
    output_prefix: str = "cluster_sweep",
    stream_blade: Optional[str] = None,
) -> Optional[Path]:
    """
    Run complete sweep pipeline.

    Returns path to merged results file, or None on failure.
    """
    start = time.time()
    job_id = "latest"

    print(f"\n{'=' * 70}")
    print(f"  CLUSTER SWEEP: {job_id}")
    print(f"{'=' * 70}")
    print(f"  Pools:   {pools_file}")
    print(
        f"  Blades:  {len(blades)} ({', '.join(blades[:3])}{'...' if len(blades) > 3 else ''})"
    )
    print(f"  Threads: {threads} per blade")
    print(f"{'=' * 70}\n")

    # Step 0: Check connectivity
    print("[0/4] Checking connectivity...")
    reachable = check_connectivity(blades)
    if not reachable:
        print("ERROR: No blades reachable!")
        return None
    if len(reachable) < len(blades):
        print(f"WARNING: {len(reachable)}/{len(blades)} blades reachable")
        blades = reachable

    # Step 1: Build
    if not skip_build:
        print("\n[1/4] Building harness...")
        if not build(blades[0]):
            print("ERROR: Build failed!")
            return None
    else:
        print("\n[1/4] Skipping build (--skip-build)")

    # Step 2: Distribute (reads candles from pool config meta.datafile)
    print("\n[2/4] Distributing data...")
    manifest = distribute(
        pools_file=pools_file,
        blades=blades,
        job_id=job_id,
        candles_file=candles_file,
        threads_per_blade=threads,
        dustswap_freq=dustswap_freq,
        apy_period_days=apy_period_days,
        apy_period_cap=apy_period_cap,
        candle_filter=candle_filter,
        output_prefix=output_prefix,
    )

    if not manifest.get("blades"):
        print("ERROR: Distribution failed!")
        return None

    manifest_path = Path(manifest["local_job_dir"]) / "manifest.json"

    # Step 3: Run
    print("\n[3/4] Running jobs...")
    results = run(manifest_path, stream_blade=stream_blade)

    successful = sum(1 for r in results.values() if r.success)
    if successful == 0:
        print("ERROR: All jobs failed!")
        return None

    # Step 4: Collect
    print("\n[4/4] Collecting results...")
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = LOCAL_RESULTS_DIR / f"{output_prefix}_{job_id}.json"

    merged = collect(manifest_path, output_file)
    if not merged:
        print("ERROR: Collection failed!")
        return None

    # Summary
    elapsed = time.time() - start
    n_runs = len(merged.get("runs", []))

    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Job ID:     {job_id}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Pools:      {n_runs}")
    print(f"  Throughput: {n_runs / elapsed:.1f} pools/sec")
    print(f"  Output:     {output_file}")
    print(f"{'=' * 70}\n")

    return output_file


def resume(
    manifest_path: Path,
    collect_only: bool = False,
    stream_blade: Optional[str] = None,
) -> Optional[Path]:
    """Resume a previous job or collect its results."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    job_id = manifest["job_id"]
    print(f"\nResuming job: {job_id}")

    if collect_only or manifest.get("run_status"):
        # Just collect
        cfg = manifest.get("config", {})
        output_prefix = cfg.get("output_prefix", "cluster_sweep")
        LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = LOCAL_RESULTS_DIR / f"{output_prefix}.json"

        merged = collect(manifest_path, output_file)
        return output_file if merged else None

    # Re-run
    results = run(manifest_path, stream_blade=stream_blade)
    if not any(r.success for r in results.values()):
        return None

    cfg = manifest.get("config", {})
    output_prefix = cfg.get("output_prefix", "cluster_sweep")
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = LOCAL_RESULTS_DIR / f"{output_prefix}.json"
    merged = collect(manifest_path, output_file)
    return output_file if merged else None


def main():
    parser = argparse.ArgumentParser(
        description="Cluster sweep orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Job inputs
    parser.add_argument(
        "--pools", type=Path, help="Pools JSON file (candles read from meta.datafile)"
    )
    parser.add_argument("--candles", type=Path, help="Override candles file (optional)")

    # Blade selection
    parser.add_argument(
        "--blades",
        nargs="+",
        default=BLADES_WORKING,
        help="Blades to use (default: a2, b8, b9, b10 with correct permissions)",
    )

    # Harness parameters
    parser.add_argument("--threads", type=int, default=CORES_PER_BLADE)
    parser.add_argument("--dustswap-freq", type=int, default=600)
    parser.add_argument("--apy-period-days", type=float, default=1.0)
    parser.add_argument("--apy-period-cap", type=int, default=20)
    parser.add_argument("--candle-filter", type=float)
    parser.add_argument("--output-prefix", default="cluster_sweep")

    # Workflow control
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume from manifest")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--stream-blade",
        type=str,
        help="Stream stdout/stderr for a single blade (e.g. blade-a2)",
    )

    args = parser.parse_args()

    # Handle modes
    if args.resume:
        if not args.resume.exists():
            print(f"Manifest not found: {args.resume}")
            sys.exit(1)
        result = resume(args.resume, args.collect_only, stream_blade=args.stream_blade)
        sys.exit(0 if result else 1)

    if args.build_only:
        reachable = check_connectivity(args.blades)
        if not reachable:
            sys.exit(1)
        success = build(reachable[0])
        sys.exit(0 if success else 1)

    # Full sweep - require pools
    if not args.pools:
        parser.error("--pools required")

    if not args.pools.exists():
        print(f"Pools not found: {args.pools}")
        sys.exit(1)

    result = sweep(
        pools_file=args.pools,
        blades=args.blades,
        skip_build=args.skip_build,
        candles_file=args.candles,
        threads=args.threads,
        dustswap_freq=args.dustswap_freq,
        apy_period_days=args.apy_period_days,
        apy_period_cap=args.apy_period_cap,
        candle_filter=args.candle_filter,
        output_prefix=args.output_prefix,
        stream_blade=args.stream_blade,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
