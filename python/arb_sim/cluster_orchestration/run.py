#!/usr/bin/env python3
"""
Execute harness jobs on cluster blades in parallel.

Each blade:
- Reads its pool batch from shared NFS
- Reads candles from shared NFS
- Writes results to shared NFS
- Uses all available cores
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import (
    REMOTE_BUILD,
    REMOTE_RESULTS,
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
    HARNESS_BINARY,
    JOB_TIMEOUT,
    CORES_PER_BLADE,
)


@dataclass
class BladeResult:
    """Result from a blade job."""

    blade: str
    success: bool
    remote_output: Optional[str] = None
    elapsed_s: float = 0.0
    n_pools: int = 0
    error: Optional[str] = None


def run_ssh(
    blade: str, command: str, timeout: int = JOB_TIMEOUT
) -> subprocess.CompletedProcess:
    """Run command on blade via SSH."""
    cmd = ["ssh"] + SSH_OPTIONS + ["-i", str(SSH_KEY), f"{SSH_USER}@{blade}", command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_ssh_stream(
    blade: str, command: str, timeout: int = JOB_TIMEOUT
) -> subprocess.CompletedProcess:
    """Run command on blade via SSH, streaming output to console."""
    cmd = ["ssh"] + SSH_OPTIONS + ["-i", str(SSH_KEY), f"{SSH_USER}@{blade}", command]
    proc = subprocess.Popen(cmd)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode)


def run_blade_job(
    blade: str,
    remote_pools: str,
    remote_candles: str,
    job_id: str,
    pool_start: int,
    pool_end: int,
    threads: int,
    dustswap_freq: int,
    apy_period_days: float,
    apy_period_cap: int,
    candle_filter: Optional[float],
    stream_output: bool = False,
    line_buffered: bool = False,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> BladeResult:
    """Run harness on a single blade with retry on connection failures."""
    start = time.time()
    result = BladeResult(blade=blade, success=False)

    for attempt in range(1, retries + 1):
        try:
            # Ensure results directory exists
            run_ssh(blade, f"mkdir -p {REMOTE_RESULTS}", timeout=30)

            # Output file on shared NFS
            output_file = f"{REMOTE_RESULTS}/result_{blade}.json"
            result.remote_output = output_file

            # Build harness command
            harness = f"{REMOTE_BUILD}/{HARNESS_BINARY}"
            cmd_parts = [
                harness,
                remote_pools,
                remote_candles,
                output_file,
                f"--threads",
                str(threads),
                f"--pool-start",
                str(pool_start),
                f"--pool-end",
                str(pool_end),
                f"--dustswapfreq",
                str(dustswap_freq),
                f"--apy-period-days",
                str(apy_period_days),
                f"--apy-period-cap",
                str(apy_period_cap),
            ]
            if candle_filter is not None:
                cmd_parts.extend(["--candle-filter", str(candle_filter)])

            cmd_str = " ".join(cmd_parts)
            if attempt == 1:
                print(f"[{blade}] Starting: pools {pool_start}-{pool_end}...")
            else:
                print(
                    f"[{blade}] Retry {attempt}/{retries}: pools {pool_start}-{pool_end}..."
                )

            if stream_output:
                stream_cmd = cmd_str
                if line_buffered:
                    stream_cmd = f"stdbuf -oL -eL {cmd_str}"
                proc = run_ssh_stream(blade, stream_cmd, timeout=JOB_TIMEOUT)
            else:
                proc = run_ssh(blade, cmd_str, timeout=JOB_TIMEOUT)

            # Check for SSH connection failure (exit 255) - retry
            if proc.returncode == 255:
                stderr = proc.stderr if hasattr(proc, "stderr") and proc.stderr else ""
                result.error = f"Exit 255: {stderr[:200]}"
                print(f"[{blade}] Connection failed (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[{blade}] FAILED: {result.error}")
                    break

            if proc.returncode != 0:
                if stream_output:
                    result.error = f"Exit {proc.returncode}"
                else:
                    result.error = f"Exit {proc.returncode}: {proc.stderr[:200]}"
                print(f"[{blade}] FAILED: {result.error}")
                break  # Non-connection error, don't retry
            else:
                # Verify output exists
                check = run_ssh(
                    blade, f"test -f {output_file} && wc -c < {output_file}", timeout=30
                )
                if check.returncode == 0:
                    result.success = True
                    size = check.stdout.strip()
                    print(f"[{blade}] SUCCESS ({size} bytes)")
                else:
                    result.error = "Output file not created"
                    print(f"[{blade}] FAILED: no output file")
                break  # Success or non-retryable failure

        except subprocess.TimeoutExpired:
            result.error = f"Timeout after {JOB_TIMEOUT}s"
            print(f"[{blade}] TIMEOUT")
            break  # Don't retry timeouts
        except Exception as e:
            result.error = str(e)
            print(f"[{blade}] ERROR (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(retry_delay)
                continue
            print(f"[{blade}] FAILED: {result.error}")

    result.elapsed_s = time.time() - start
    return result


def run_parallel(
    manifest: dict, stream_blade: Optional[str] = None
) -> Dict[str, BladeResult]:
    """Run jobs on all blades in parallel."""
    blades_info = manifest["blades"]
    remote_candles = manifest["remote_candles"]
    remote_pools = manifest["remote_pools"]
    job_id = manifest["job_id"]
    cfg = manifest["config"]

    print(f"\n{'=' * 60}")
    print(f"Running on {len(blades_info)} blades in parallel")
    print(f"{'=' * 60}\n")

    results = {}
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=len(blades_info)) as executor:
        futures = {}

        for blade, info in blades_info.items():
            future = executor.submit(
                run_blade_job,
                blade=blade,
                remote_pools=remote_pools,
                remote_candles=remote_candles,
                job_id=job_id,
                pool_start=info["pool_start"],
                pool_end=info["pool_end"],
                threads=cfg.get("threads_per_blade", CORES_PER_BLADE),
                dustswap_freq=cfg.get("dustswap_freq", 600),
                apy_period_days=cfg.get("apy_period_days", 1.0),
                apy_period_cap=cfg.get("apy_period_cap", 20),
                candle_filter=cfg.get("candle_filter"),
                stream_output=stream_blade == blade,
                line_buffered=False,
            )
            futures[future] = blade

        for future in as_completed(futures):
            blade = futures[future]
            try:
                results[blade] = future.result()
            except Exception as e:
                results[blade] = BladeResult(blade=blade, success=False, error=str(e))

    elapsed = time.time() - total_start
    successful = sum(1 for r in results.values() if r.success)

    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.1f}s - {successful}/{len(results)} successful")
    print(f"{'=' * 60}")

    if successful < len(results):
        print("\nFailed blades:")
        for blade, r in results.items():
            if not r.success:
                print(f"  {blade}: {r.error}")

    return results


def run(
    manifest_path: Path, stream_blade: Optional[str] = None
) -> Dict[str, BladeResult]:
    """Load manifest and run jobs."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = run_parallel(manifest, stream_blade=stream_blade)

    # Update manifest with results
    manifest["run_status"] = {
        blade: {
            "success": r.success,
            "remote_output": r.remote_output,
            "elapsed_s": r.elapsed_s,
            "error": r.error,
        }
        for blade, r in results.items()
    }
    manifest["run_completed_at"] = datetime.utcnow().isoformat()

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run jobs on cluster")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stream-blade",
        type=str,
        help="Stream stdout/stderr for a single blade (e.g. blade-a2)",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    results = run(args.manifest, stream_blade=args.stream_blade)

    all_ok = all(r.success for r in results.values())
    if all_ok:
        print(
            f"\nAll jobs succeeded! Run: python collect.py --manifest {args.manifest}"
        )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
