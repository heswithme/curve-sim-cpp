#!/usr/bin/env python3
"""
Distribute job data to shared NFS.

Since all blades share /home/heswithme, we only upload once:
1. Read candles path from pool config meta.datafile
2. Upload candles file (conditionally - skip if exists)
3. Upload single pools file (all blades read same file)
4. Compute pool index ranges per blade
"""

import json
import subprocess
from datetime import datetime, timezone
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
    HARNESS_BINARY,
    JobConfig,
)


def fmt_size(n_bytes: int) -> str:
    """Format bytes as human-readable MB."""
    return f"{n_bytes / 1_000_000:.1f} MB"


def run_ssh(blade: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run command on blade via SSH."""
    cmd = ["ssh"] + SSH_OPTIONS + ["-i", str(SSH_KEY), f"{SSH_USER}@{blade}", command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def rsync_to_cluster(
    local_path: Path,
    remote_path: str,
    blade: str,
    timeout: int = 600,
    quiet: bool = False,
) -> None:
    """Copy file to shared NFS via rsync with compression."""
    ssh_opts = f"ssh -i {SSH_KEY} " + " ".join(SSH_OPTIONS)
    cmd = [
        "rsync",
        "-az" if quiet else "-avz",  # archive, compress; verbose for large files
        "-e",
        ssh_opts,
        str(local_path),
        f"{SSH_USER}@{blade}:{remote_path}",
    ]
    if not quiet:
        cmd.insert(2, "--progress")
    subprocess.run(cmd, check=True, timeout=timeout)


def rsync_many_to_cluster(
    local_paths: List[Path],
    remote_dir: str,
    blade: str,
    timeout: int = 600,
    quiet: bool = False,
) -> None:
    """Copy multiple files to a shared NFS directory via one rsync process."""
    if not local_paths:
        return
    ssh_opts = f"ssh -i {SSH_KEY} " + " ".join(SSH_OPTIONS)
    cmd = [
        "rsync",
        "-az" if quiet else "-avz",
    ]
    if not quiet:
        cmd.append("--progress")
    cmd.extend(
        [
            "-e",
            ssh_opts,
            *[str(path) for path in local_paths],
            f"{SSH_USER}@{blade}:{remote_dir.rstrip('/')}/",
        ]
    )
    subprocess.run(cmd, check=True, timeout=timeout)


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


def load_pools(pools_file: Path) -> Tuple[int, dict]:
    """Load pools from JSON. Returns (pool_count, metadata)."""
    with open(pools_file) as f:
        data = json.load(f)

    meta = data.get("meta", data.get("metadata", {}))
    pools = data.get("pools")
    if isinstance(pools, list):
        return len(pools), meta
    if "pool" in data:
        return 1, meta
    pool_count = meta.get("pool_count") if isinstance(meta, dict) else None
    if pool_count is not None:
        return int(pool_count), meta
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    count = 1
    found_axis = False
    for key, axis in grid.items():
        if not isinstance(key, str) or not key.startswith("x") or not key[1:].isdigit():
            continue
        values = axis.get("values") if isinstance(axis, dict) else None
        if not isinstance(values, list) or not values:
            continue
        found_axis = True
        count *= len(values)
    if found_axis:
        return count, meta
    pools = []
    return len(pools), meta


def compute_ranges(n_pools: int, n_blades: int) -> List[Tuple[int, int]]:
    """Compute (start, end) ranges for each blade."""
    if n_blades <= 0 or n_pools == 0:
        return [(0, 0) for _ in range(max(1, n_blades))]

    base_size = n_pools // n_blades
    remainder = n_pools % n_blades

    ranges = []
    start = 0
    for i in range(n_blades):
        size = base_size + (1 if i < remainder else 0)
        ranges.append((start, start + size))
        start += size

    return ranges


def compute_block_cyclic_ranges(
    n_pools: int, n_blades: int, chunk_size: int
) -> List[List[Tuple[int, int]]]:
    """Compute deterministic block-cyclic half-open ranges per blade."""
    if n_blades <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    assignments: List[List[Tuple[int, int]]] = [[] for _ in range(n_blades)]
    if n_pools <= 0:
        return assignments

    for chunk_idx, start in enumerate(range(0, n_pools, chunk_size)):
        end = min(start + chunk_size, n_pools)
        assignments[chunk_idx % n_blades].append((start, end))

    return assignments


def ranges_pool_count(ranges: List[Tuple[int, int]]) -> int:
    """Return total pools covered by a list of half-open ranges."""
    return sum(end - start for start, end in ranges)


def write_range_file(path: Path, ranges: List[Tuple[int, int]]) -> None:
    """Write half-open ranges as 'start end' lines."""
    with open(path, "w") as f:
        for start, end in ranges:
            f.write(f"{start} {end}\n")


def distribute(
    pools_file: Path,
    blades: List[str],
    job_id: str = None,
    candles_file: Optional[Path] = None,
    threads_per_blade: int = CORES_PER_BLADE,
    dustswap_freq: int = 3600,
    candle_filter: Optional[float] = None,
    disable_slippage_probes: bool = False,
    quiet_harness: bool = False,
    output_prefix: str = "cluster_sweep",
    assignment: str = "block-cyclic",
    chunk_size: int = 2048,
    verbose_assignment: bool = False,
) -> dict:
    """
    Upload job data to shared NFS.

    Reads candles path from pool config meta.datafile if not provided.
    Skips candles upload if file already exists with same size.
    Uploads single pools file. Blades use either --pool-ranges or
    --pool-start/--pool-end depending on assignment.

    Returns manifest with all paths and configuration.
    """
    if job_id is None:
        job_id = "latest"
    if assignment not in {"block-cyclic", "contiguous"}:
        raise ValueError("assignment must be 'block-cyclic' or 'contiguous'")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    blade = blades[0]  # Use any blade to access shared NFS
    local_job_dir = Path(__file__).parent / "jobs" / job_id
    local_job_dir.mkdir(parents=True, exist_ok=True)

    # Load pools and get metadata
    print(f"Loading pools from {pools_file}...")
    n_pools, meta = load_pools(pools_file)

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
    print(f"  Pools:   {n_pools} from {pools_file.name}")
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
        print(f"Candles already uploaded: {candles_file.name} ({fmt_size(local_size)})")
    else:
        print(f"Uploading candles: {candles_file.name} ({fmt_size(local_size)})...")
        rsync_to_cluster(candles_file, remote_candles, blade)
        print(f"  Done.")

    # Upload single pools file. Pool configs are small, and same-size compact
    # grids can encode different axes; never trust size-only freshness here.
    remote_pools = f"{REMOTE_JOBS}/pools.json"
    local_pools_size = pools_file.stat().st_size
    print(f"Uploading pools: {pools_file.name} ({fmt_size(local_pools_size)})...")
    rsync_to_cluster(pools_file, remote_pools, blade)
    print(f"  Done.")

    # Compute pool ranges per blade
    if assignment == "contiguous":
        ranges_by_blade = [
            [range_] if range_[1] > range_[0] else []
            for range_ in compute_ranges(n_pools, len(blades))
        ]
    else:
        ranges_by_blade = compute_block_cyclic_ranges(n_pools, len(blades), chunk_size)
    blade_assignments = {}
    range_files: List[Path] = []

    for blade_name, blade_ranges in zip(blades, ranges_by_blade):
        n_blade_pools = ranges_pool_count(blade_ranges)
        if n_blade_pools == 0:
            if verbose_assignment:
                print(f"  {blade_name}: 0 pools (skipped)")
            continue

        pool_start = min(start for start, _end in blade_ranges)
        pool_end = max(end for _start, end in blade_ranges)
        blade_assignments[blade_name] = {
            "pool_start": pool_start,
            "pool_end": pool_end,
            "n_pools": n_blade_pools,
            "ranges": [[start, end] for start, end in blade_ranges],
        }

        if assignment == "block-cyclic":
            remote_range_file = f"{REMOTE_JOBS}/{job_id}_{blade_name}_ranges.txt"
            range_file = local_job_dir / f"{job_id}_{blade_name}_ranges.txt"
            write_range_file(range_file, blade_ranges)
            range_files.append(range_file)
            blade_assignments[blade_name]["pool_ranges_file"] = str(range_file)
            blade_assignments[blade_name]["pool_ranges_remote"] = remote_range_file
            if verbose_assignment:
                print(
                    f"  {blade_name}: {len(blade_ranges)} chunks, "
                    f"{n_blade_pools} pools (span {pool_start}-{pool_end})"
                )
        else:
            if verbose_assignment:
                print(
                    f"  {blade_name}: pools {pool_start}-{pool_end} "
                    f"({n_blade_pools} pools)"
                )

    if assignment == "block-cyclic" and range_files:
        rsync_many_to_cluster(
            range_files, str(REMOTE_JOBS), blade, timeout=60, quiet=True
        )
        print(f"Uploaded {len(range_files)} range files.")

    if blade_assignments:
        pool_counts = [int(info["n_pools"]) for info in blade_assignments.values()]
        total_chunks = sum(
            len(info.get("ranges", [])) for info in blade_assignments.values()
        )
        if assignment == "block-cyclic":
            print(
                "Assignment: "
                f"{len(blade_assignments)} blades, {total_chunks} chunks, "
                f"per blade {min(pool_counts)}-{max(pool_counts)} pools"
            )
        else:
            print(
                "Assignment: "
                f"{len(blade_assignments)} blades, "
                f"per blade {min(pool_counts)}-{max(pool_counts)} pools"
            )

    # Build manifest
    manifest = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "remote_candles": remote_candles,
        "remote_pools": remote_pools,
        "total_pools": n_pools,
        "assignment_mode": assignment,
        "chunk_size": chunk_size,
        "blades": blade_assignments,
        "config": {
            "threads_per_blade": threads_per_blade,
            "harness_binary": HARNESS_BINARY,
            "dustswap_freq": dustswap_freq,
            "candle_filter": candle_filter,
            "disable_slippage_probes": disable_slippage_probes,
            "quiet_harness": quiet_harness,
            "start_time": meta.get("start_time"),
            "output_prefix": output_prefix,
        },
        "local_job_dir": str(local_job_dir),
        # Preserve grid metadata for visualization
        "grid": meta.get("grid", {}),
        "base_pool": meta.get("base_pool", {}),
        "base_costs": meta.get("base_costs", {}),
        "fee_equalize": meta.get("fee_equalize", False),
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
    parser.add_argument("--dustswap-freq", type=int, default=3600)
    parser.add_argument("--candle-filter", type=float)
    parser.add_argument("--disable-slippage-probes", action="store_true")
    parser.add_argument("--quiet-harness", action="store_true")
    parser.add_argument("--output-prefix", default="cluster_sweep")
    parser.add_argument(
        "--assignment",
        choices=["block-cyclic", "contiguous"],
        default="block-cyclic",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--verbose-assignment",
        action="store_true",
        help="Print per-blade pool assignment details",
    )
    parser.add_argument("--job-id", type=str)
    args = parser.parse_args()

    manifest = distribute(
        pools_file=args.pools,
        blades=args.blades,
        job_id=args.job_id,
        candles_file=args.candles,
        threads_per_blade=args.threads,
        dustswap_freq=args.dustswap_freq,
        candle_filter=args.candle_filter,
        disable_slippage_probes=args.disable_slippage_probes,
        quiet_harness=args.quiet_harness,
        output_prefix=args.output_prefix,
        assignment=args.assignment,
        chunk_size=args.chunk_size,
        verbose_assignment=args.verbose_assignment,
    )
    print(f"\nNext: python run.py --manifest {manifest['local_job_dir']}/manifest.json")


if __name__ == "__main__":
    main()
