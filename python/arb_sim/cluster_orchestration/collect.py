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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
    DEFAULT_BLADES,
    LOCAL_RESULTS_DIR,
    REMOTE_RESULTS,
)


def scp_from_cluster(remote_path: str, local_path: Path, blade: str) -> bool:
    """Download file from shared NFS via any blade."""
    cmd = [
        "scp",
        "-i",
        str(SSH_KEY),
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{SSH_USER}@{blade}:{remote_path}",
        str(local_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
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


def collect(manifest_path: Path, output_file: Path = None) -> Optional[dict]:
    """
    Download and merge results from shared NFS.

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

    # Download each result file
    downloaded = {}
    for blade_name, status in run_status.items():
        if not status.get("success"):
            print(f"[{blade_name}] Skipping (job failed)")
            continue

        remote_path = status.get("remote_output")
        if not remote_path:
            continue

        local_path = downloads_dir / f"result_{blade_name}.json"
        print(f"[{blade_name}] Downloading...")

        if scp_from_cluster(remote_path, local_path, blade):
            downloaded[blade_name] = local_path
            print(f"[{blade_name}] OK ({local_path.stat().st_size:,} bytes)")
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
            "created_utc": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "total_pools": len(all_runs),
            "errors": sum(s["errors"] for s in blade_stats.values()),
            "blades_used": len(downloaded),
            "blade_stats": blade_stats,
            "harness_args": {
                "dustswapfreq": cfg.get("dustswap_freq"),
                "apy_period_days": cfg.get("apy_period_days"),
                "apy_period_cap": cfg.get("apy_period_cap"),
                "candle_filter": cfg.get("candle_filter"),
            },
            # Include grid metadata for visualization compatibility
            "grid": grid,
        },
        "runs": all_runs,
    }

    # Compute summary stats
    vps = [
        r.get("result", {}).get("vp")
        for r in all_runs
        if r.get("result", {}).get("vp") is not None
    ]
    apys = [
        r.get("result", {}).get("tw_capped_apy")
        for r in all_runs
        if r.get("result", {}).get("tw_capped_apy") is not None
        and r.get("result", {}).get("tw_capped_apy") >= 0
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
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    merged = collect(args.manifest, args.output)
    sys.exit(0 if merged else 1)


if __name__ == "__main__":
    main()
