#!/usr/bin/env python3
"""
Distribute job data to shared NFS.

Since all blades share /home/heswithme, we only upload once:
1. Read candles path from pool config meta.datafile
2. Upload candles file (conditionally - skip if exists)
3. Split pools into per-blade batches
4. Upload all batch files
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from config import (
    REMOTE_DATA,
    REMOTE_JOBS,
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
    DEFAULT_BLADES,
    CORES_PER_BLADE,
    JobConfig,
)


def run_ssh(blade: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run command on blade via SSH."""
    cmd = ["ssh"] + SSH_OPTIONS + ["-i", str(SSH_KEY), f"{SSH_USER}@{blade}", command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def scp_to_cluster(local_path: Path, remote_path: str, blade: str) -> None:
    """Copy file to shared NFS via any blade."""
    cmd = [
        "scp",
        "-i",
        str(SSH_KEY),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(local_path),
        f"{SSH_USER}@{blade}:{remote_path}",
    ]
    subprocess.run(cmd, check=True, timeout=600)


def file_exists_remote(blade: str, remote_path: str) -> bool:
    """Check if file exists on remote."""
    result = run_ssh(blade, f"test -f {remote_path} && echo EXISTS", timeout=30)
    return "EXISTS" in result.stdout


def get_remote_file_size(blade: str, remote_path: str) -> int:
    """Get file size on remote, returns 0 if not exists."""
    result = run_ssh(
        blade, f"stat -c %s {remote_path} 2>/dev/null || echo 0", timeout=30
    )
    try:
        return int(result.stdout.strip())
    except:
        return 0


def load_pools(pools_file: Path) -> Tuple[List[dict], dict]:
    """Load pools from JSON. Returns (pools_list, metadata)."""
    with open(pools_file) as f:
        data = json.load(f)

    pools = data.get("pools", [])
    meta = data.get("meta", data.get("metadata", {}))
    return pools, meta


def split_pools(pools: List[dict], n_blades: int) -> List[List[dict]]:
    """Split pools evenly across blades."""
    if n_blades <= 0 or len(pools) == 0:
        return [[] for _ in range(max(1, n_blades))]

    base_size = len(pools) // n_blades
    remainder = len(pools) % n_blades

    batches = []
    start = 0
    for i in range(n_blades):
        size = base_size + (1 if i < remainder else 0)
        batches.append(pools[start : start + size])
        start += size

    return batches


def distribute(
    pools_file: Path,
    blades: List[str],
    job_id: str = None,
    candles_file: Optional[Path] = None,
    threads_per_blade: int = CORES_PER_BLADE,
    dustswap_freq: int = 600,
    apy_period_days: float = 1.0,
    apy_period_cap: int = 20,
    candle_filter: Optional[float] = None,
    output_prefix: str = "cluster_sweep",
) -> dict:
    """
    Upload job data to shared NFS.

    Reads candles path from pool config meta.datafile if not provided.
    Skips candles upload if file already exists with same size.

    Returns manifest with all paths and configuration.
    """
    if job_id is None:
        job_id = "latest"

    blade = blades[0]  # Use any blade to access shared NFS

    # Load pools and get metadata
    print(f"Loading pools from {pools_file}...")
    pools, meta = load_pools(pools_file)

    # Get candles file from meta.datafile if not provided
    if candles_file is None:
        datafile = meta.get("datafile")
        if not datafile:
            raise ValueError(
                "No candles file provided and meta.datafile not found in pool config"
            )
        candles_file = Path(datafile)

    if not candles_file.exists():
        raise FileNotFoundError(f"Candles file not found: {candles_file}")

    print(f"\n{'=' * 60}")
    print(f"Distributing job: {job_id}")
    print(f"  Pools:   {len(pools)} from {pools_file.name}")
    print(f"  Candles: {candles_file.name}")
    print(f"  Blades:  {len(blades)}")
    print(f"{'=' * 60}\n")

    # Ensure directories exist
    run_ssh(blade, f"mkdir -p {REMOTE_DATA} {REMOTE_JOBS}")

    # Upload candles (conditional - check if exists with same size)
    remote_candles = f"{REMOTE_DATA}/{candles_file.name}"
    local_size = candles_file.stat().st_size
    remote_size = get_remote_file_size(blade, remote_candles)

    if remote_size == local_size:
        print(f"Candles already uploaded: {candles_file.name} ({local_size:,} bytes)")
    else:
        print(f"Uploading candles: {candles_file.name} ({local_size:,} bytes)...")
        scp_to_cluster(candles_file, remote_candles, blade)
        print(f"  Done.")

    # Split pools across blades
    print(f"Splitting {len(pools)} pools across {len(blades)} blades...")
    batches = split_pools(pools, len(blades))

    # Create local job directory
    local_job_dir = Path(__file__).parent / "jobs" / job_id
    local_job_dir.mkdir(parents=True, exist_ok=True)

    # Create and upload batch files
    blade_assignments = {}

    for i, (blade_name, batch) in enumerate(zip(blades, batches)):
        if not batch:
            print(f"  {blade_name}: 0 pools (skipped)")
            continue

        # Create batch file with updated meta
        batch_data = {
            "meta": {
                **meta,
                "datafile": remote_candles,  # Update to remote path
                "blade_index": i,
                "job_id": job_id,
                "batch_size": len(batch),
            },
            "pools": batch,
        }

        local_file = local_job_dir / f"pools_{blade_name}.json"
        with open(local_file, "w") as f:
            json.dump(batch_data, f)

        # Upload to shared NFS
        remote_file = f"{REMOTE_JOBS}/pools_{blade_name}.json"
        scp_to_cluster(local_file, remote_file, blade)

        blade_assignments[blade_name] = {
            "remote_pools": remote_file,
            "n_pools": len(batch),
        }
        print(f"  {blade_name}: {len(batch)} pools")

    # Build manifest
    manifest = {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat(),
        "remote_candles": remote_candles,
        "blades": blade_assignments,
        "config": {
            "threads_per_blade": threads_per_blade,
            "dustswap_freq": dustswap_freq,
            "apy_period_days": apy_period_days,
            "apy_period_cap": apy_period_cap,
            "candle_filter": candle_filter,
            "output_prefix": output_prefix,
        },
        "local_job_dir": str(local_job_dir),
        # Preserve grid metadata for visualization
        "grid": meta.get("grid", {}),
    }

    # Save manifest
    manifest_file = local_job_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    total_pools = sum(b["n_pools"] for b in blade_assignments.values())
    print(f"\nDistributed {total_pools} pools to {len(blade_assignments)} blades")
    print(f"Manifest: {manifest_file}")

    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Distribute jobs to cluster")
    parser.add_argument(
        "--pools",
        type=Path,
        required=True,
        help="Pool config JSON (reads candles from meta.datafile)",
    )
    parser.add_argument("--candles", type=Path, help="Override candles file (optional)")
    parser.add_argument("--blades", nargs="+", default=DEFAULT_BLADES)
    parser.add_argument("--threads", type=int, default=CORES_PER_BLADE)
    parser.add_argument("--job-id", type=str)
    args = parser.parse_args()

    manifest = distribute(
        pools_file=args.pools,
        blades=args.blades,
        job_id=args.job_id,
        candles_file=args.candles,
        threads_per_blade=args.threads,
    )
    print(f"\nNext: python run.py --manifest {manifest['local_job_dir']}/manifest.json")


if __name__ == "__main__":
    main()
