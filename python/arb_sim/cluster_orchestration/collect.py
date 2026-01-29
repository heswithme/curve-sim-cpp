#!/usr/bin/env python3
"""
Collect and merge results from shared NFS.

Since all results are on shared NFS, we only need to:
1. Download from any one blade (all see same files)
2. Merge into single output file
3. Preserve grid metadata for visualization compatibility
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
    DEFAULT_BLADES,
    LOCAL_RESULTS_DIR,
    REMOTE_RESULTS,
)

SCRIPT_DIR = Path(__file__).resolve().parents[1]
TRADE_DATA_DIR = SCRIPT_DIR / "trade_data"


def fmt_size(n_bytes: int) -> str:
    """Format bytes as human-readable MB."""
    return f"{n_bytes / 1_000_000:.1f} MB"


def find_local_candles(filename: str) -> Optional[Path]:
    """Search trade_data folder recursively for candles file by name."""
    if not TRADE_DATA_DIR.exists():
        return None
    matches = list(TRADE_DATA_DIR.rglob(filename))
    return matches[0] if matches else None


def rsync_from_cluster(
    remote_path: str,
    local_path: Path,
    blade: str,
    timeout: int = 600,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> bool:
    """Download file from shared NFS via rsync with compression and retry."""
    ssh_opts = f"ssh -i {SSH_KEY} " + " ".join(SSH_OPTIONS)
    cmd = [
        "rsync",
        "-avz",  # archive, verbose, compress
        "--progress",
        "-e",
        ssh_opts,
        f"{SSH_USER}@{blade}:{remote_path}",
        str(local_path),
    ]
    for attempt in range(retries):
        try:
            subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Download failed after {retries} attempts: {e}")
    return False


def load_json(path: Path) -> Optional[dict]:
    """Load JSON file, return None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def extract_grid_values(run: Dict[str, Any], grid: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract grid dimension values from a run's params.

    Grid format: {"x1": {"name": "donation_apy", ...}, "x2": {"name": "A", ...}}
    Returns: {"donation_apy": 0.0, "A": 100000, ...}
    """
    pool_params = run.get("params", {}).get("pool", {})
    values = {}

    for key, dim_info in grid.items():
        if not key.startswith("x") or not key[1:].isdigit():
            continue
        name = dim_info.get("name") if isinstance(dim_info, dict) else None
        if not name:
            continue

        raw_val = pool_params.get(name)
        if raw_val is not None:
            try:
                values[name] = float(raw_val)
            except (ValueError, TypeError):
                pass

    return values


def collect(
    manifest_path: Path, output_file: Path = None, force: bool = False
) -> Optional[dict]:
    """
    Download and merge results from shared NFS.

    Args:
        manifest_path: Path to job manifest
        output_file: Where to save merged results
        force: If True, attempt download even if manifest says jobs failed

    Returns merged results dict.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    job_id = manifest["job_id"]
    run_status = manifest.get("run_status", {})

    print(f"\n{'=' * 60}")
    print(f"Collecting results for job: {job_id}")
    print(f"{'=' * 60}\n")

    # Use any blade to access shared NFS
    blade = DEFAULT_BLADES[0]

    # Download directory
    job_dir = Path(manifest["local_job_dir"])
    downloads_dir = job_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Build list of files to download
    to_download = []
    for blade_name, status in run_status.items():
        if not force and not status.get("success"):
            print(f"[{blade_name}] Skipping (job failed)")
            continue

        remote_path = status.get("remote_output")
        if not remote_path:
            # If force mode, try to construct the path
            if force:
                remote_path = f"{REMOTE_RESULTS}/result_{blade_name}.json"
            else:
                continue

        local_path = downloads_dir / f"result_{blade_name}.json"
        to_download.append((blade_name, remote_path, local_path))

    # Download in parallel
    def download_one(args: Tuple[str, str, Path]) -> Tuple[str, Path, bool]:
        blade_name, remote_path, local_path = args
        success = rsync_from_cluster(remote_path, local_path, blade)
        return blade_name, local_path, success

    downloaded = {}
    if not to_download:
        print("No files to download!")
        print("Hint: use --force to attempt download even if manifest says jobs failed")
        return None

    print(f"Downloading {len(to_download)} result files in parallel...")

    with ThreadPoolExecutor(max_workers=min(32, len(to_download))) as executor:
        futures = {executor.submit(download_one, args): args[0] for args in to_download}
        for future in as_completed(futures):
            blade_name, local_path, success = future.result()
            if success:
                downloaded[blade_name] = local_path
                print(f"[{blade_name}] OK ({fmt_size(local_path.stat().st_size)})")
            else:
                print(f"[{blade_name}] Failed")

    if not downloaded:
        print("No results downloaded!")
        return None

    # Get grid metadata from manifest
    grid = manifest.get("grid", {})

    # Merge results
    print(f"\nMerging {len(downloaded)} result files...")

    all_runs = []
    blade_stats = {}

    for blade_name, path in downloaded.items():
        data = load_json(path)
        if not data:
            continue

        runs = data.get("runs", [])
        errors = sum(1 for r in runs if "error" in r)

        blade_stats[blade_name] = {"pools": len(runs), "errors": errors}

        for run in runs:
            run["_blade"] = blade_name
            # Extract grid dimension values and add as "pool" dict
            # This makes the output compatible with plot_heatmap_nd.py
            if grid:
                run["pool"] = extract_grid_values(run, grid)
            all_runs.append(run)

    # Build merged output
    cfg = manifest.get("config", {})
    merged = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "total_pools": len(all_runs),
            "errors": sum(s["errors"] for s in blade_stats.values()),
            "blades_used": len(downloaded),
            "blade_stats": blade_stats,
            "harness_args": {
                "dustswapfreq": cfg.get("dustswap_freq"),
                "candle_filter": cfg.get("candle_filter"),
            },
            # Include grid metadata for visualization compatibility
            "grid": grid,
        },
        "runs": all_runs,
    }

    remote_candles = manifest.get("remote_candles")
    if remote_candles:
        candles_name = Path(remote_candles).name
        local_candles = find_local_candles(candles_name)
        if local_candles:
            merged["metadata"]["candles_file"] = str(local_candles)
        else:
            merged["metadata"]["candles_file"] = (
                candles_name  # fallback to filename only
            )

    # Compute summary stats
    vps = [
        r.get("result", {}).get("vp")
        for r in all_runs
        if r.get("result", {}).get("vp") is not None
    ]
    apys = [
        r.get("result", {}).get("apy")
        for r in all_runs
        if r.get("result", {}).get("apy") is not None
        and r.get("result", {}).get("apy") >= 0
    ]

    merged["summary"] = {
        "total_runs": len(all_runs),
        "successful": len([r for r in all_runs if "error" not in r]),
    }
    if vps:
        merged["summary"]["vp"] = {
            "min": min(vps),
            "max": max(vps),
            "avg": sum(vps) / len(vps),
        }
    if apys:
        merged["summary"]["apy"] = {
            "min": min(apys),
            "max": max(apys),
            "avg": sum(apys) / len(apys),
        }

    # Save output
    if output_file is None:
        LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = cfg.get("output_prefix", "cluster_sweep")
        output_file = LOCAL_RESULTS_DIR / f"{prefix}.json"

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Results Summary")
    print(f"{'=' * 60}")
    print(f"  Total pools: {merged['summary']['total_runs']}")
    print(f"  Successful: {merged['summary']['successful']}")
    if "vp" in merged["summary"]:
        vp = merged["summary"]["vp"]
        print(f"  VP: {vp['min']:.4f} - {vp['max']:.4f} (avg {vp['avg']:.4f})")
    if "apy" in merged["summary"]:
        apy = merged["summary"]["apy"]
        print(f"  APY: {apy['min']:.2%} - {apy['max']:.2%} (avg {apy['avg']:.2%})")
    print(f"\nSaved to: {output_file}")

    return merged


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect results from cluster")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Attempt download even if manifest says jobs failed",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    merged = collect(args.manifest, args.output, force=args.force)
    sys.exit(0 if merged else 1)


if __name__ == "__main__":
    main()
