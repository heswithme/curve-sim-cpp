#!/usr/bin/env python3
"""
Collect and merge results from shared NFS.

Since all results are on shared NFS, we only need to:
1. Download from any one blade (all see same files)
2. Merge into single output file
3. Preserve grid metadata for visualization compatibility
"""

import json
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    remote_is_dir: bool = False,
) -> bool:
    """Download file from shared NFS via rsync with compression and retry."""
    ssh_opts = f"ssh -i {SSH_KEY} " + " ".join(SSH_OPTIONS)
    source = f"{SSH_USER}@{blade}:{remote_path}"
    dest = str(local_path)
    if remote_is_dir:
        local_path.mkdir(parents=True, exist_ok=True)
        source = source.rstrip("/") + "/"
        dest = str(local_path) + "/"
    cmd = [
        "rsync",
        "-avz",  # archive, verbose, compress
        "--progress",
        "-e",
        ssh_opts,
        source,
        dest,
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


def _validate_index_coverage(
    indices_by_shard: List[Tuple[str, np.ndarray]],
    total_pools: int,
) -> None:
    """Validate complete one-to-one coverage of [0, total_pools)."""
    seen = np.zeros(total_pools, dtype=bool)
    for blade_name, raw_indices in indices_by_shard:
        indices = np.asarray(raw_indices)
        if indices.ndim != 1:
            raise ValueError(f"{blade_name}: pool_index must be a 1-D array")
        if np.issubdtype(indices.dtype, np.signedinteger) and np.any(indices < 0):
            raise ValueError(f"{blade_name}: pool_index contains negative values")
        if np.any(indices >= total_pools):
            raise ValueError(f"{blade_name}: pool_index contains out-of-range values")

        idx = indices.astype(np.intp, copy=False)
        if len(np.unique(idx)) != len(idx) or np.any(seen[idx]):
            raise ValueError(f"{blade_name}: duplicate pool_index values detected")
        seen[idx] = True

    missing = total_pools - int(seen.sum())
    if missing:
        raise ValueError(f"missing {missing} pool indices in merged shards")


def _validate_pool_index_shards(shard_infos: List[Dict[str, Any]], total_pools: int) -> None:
    """Validate NPZ shards that carry explicit pool_index arrays."""
    indices_by_shard: List[Tuple[str, np.ndarray]] = []
    for info in shard_infos:
        with np.load(info["metrics_path"]) as arrays:
            if "pool_index" not in arrays.files:
                raise ValueError(f"{info['blade']}: missing pool_index.npy")
            indices = np.asarray(arrays["pool_index"])
            if len(indices) != int(info["n_pools"]):
                raise ValueError(
                    f"{info['blade']}: pool_index length {len(indices)} "
                    f"does not match n_pools {info['n_pools']}"
                )
            indices_by_shard.append((str(info["blade"]), indices.copy()))

    _validate_index_coverage(indices_by_shard, total_pools)


def _validate_contiguous_shards(shard_infos: List[Dict[str, Any]], total_pools: int) -> None:
    """Validate old contiguous shards before slice-based merge fallback."""
    indices_by_shard: List[Tuple[str, np.ndarray]] = []
    for info in shard_infos:
        start = int(info["pool_start"])
        n_pools = int(info["n_pools"])
        end = start + n_pools
        if start < 0 or end > total_pools:
            raise ValueError(f"{info['blade']}: contiguous shard range is out of bounds")
        indices_by_shard.append((str(info["blade"]), np.arange(start, end, dtype=np.intp)))

    _validate_index_coverage(indices_by_shard, total_pools)


def extract_grid_values(run: Dict[str, Any], grid: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract grid dimension values from a run's params.

    Grid format: {"x1": {"name": "donation_apy", ...}, "x2": {"name": "A", ...}}
    Returns: {"donation_apy": 0.0, "A": 100000, ...}
    """
    pool_params = run.get("params", {}).get("pool", {})
    values = {}

    def pool_value(name: str) -> Any:
        current: Any = pool_params
        for part in name.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None
        return current

    for key, dim_info in grid.items():
        if not key.startswith("x") or not key[1:].isdigit():
            continue
        name = dim_info.get("name") if isinstance(dim_info, dict) else None
        if not name:
            continue

        raw_val = pool_value(name)
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
                remote_path = f"{REMOTE_RESULTS}/result_{blade_name}"
            else:
                continue

        remote_is_dir = not str(remote_path).endswith(".json")
        local_path = downloads_dir / (
            f"result_{blade_name}" if remote_is_dir else f"result_{blade_name}.json"
        )
        to_download.append((blade_name, remote_path, local_path, remote_is_dir))

    # Download in parallel
    def download_one(args: Tuple[str, str, Path, bool]) -> Tuple[str, Path, bool]:
        blade_name, remote_path, local_path, remote_is_dir = args
        success = rsync_from_cluster(remote_path, local_path, blade, remote_is_dir=remote_is_dir)
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

    if all(path.is_dir() for path in downloaded.values()):
        cfg = manifest.get("config", {})
        if output_file is None:
            LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            prefix = cfg.get("output_prefix", "cluster_sweep")
            output_file = LOCAL_RESULTS_DIR / prefix
        if output_file.suffix == ".json":
            output_file = output_file.with_suffix("")

        if output_file.exists():
            if output_file.is_dir():
                shutil.rmtree(output_file)
            else:
                output_file.unlink()
        output_file.mkdir(parents=True, exist_ok=True)

        shard_infos = []
        blade_stats = {}
        total_runs = 0
        total_errors = 0
        vp_values: list[float] = []
        apy_values: list[float] = []
        metrics_schema = None
        merged_errors: dict[str, Any] = {}
        real_type: str | None = None

        for blade_name, path in downloaded.items():
            manifest_path = path / "manifest.json"
            data = load_json(manifest_path)
            if not data:
                continue
            if metrics_schema is None:
                metrics_schema = data.get("metrics_schema")
            meta = data.get("metadata", {})
            shard_real = meta.get("real") if isinstance(meta, dict) else None
            if isinstance(shard_real, str):
                if real_type is None:
                    real_type = shard_real
                elif real_type != shard_real:
                    real_type = "mixed"
            n_pools = int(data.get("n_pools", meta.get("n_pools", 0)))
            pool_start = int(data.get("pool_start", meta.get("pool_start", 0)))
            pool_end_raw = data.get("pool_end", meta.get("pool_end"))
            pool_end = int(pool_end_raw) if pool_end_raw is not None else pool_start + n_pools
            errors_path = path / "errors.json"
            shard_errors = load_json(errors_path) if errors_path.exists() else {}
            if isinstance(shard_errors, dict):
                merged_errors.update(shard_errors)
            else:
                shard_errors = {}
            errors = len(shard_errors)
            blade_stats[blade_name] = {"pools": n_pools, "errors": errors}
            total_runs += n_pools
            total_errors += errors
            shard_infos.append(
                {
                    "blade": blade_name,
                    "path": path,
                    "metrics_path": path / "metrics.npz",
                    "pool_start": pool_start,
                    "pool_end": pool_end,
                    "n_pools": n_pools,
                    "has_pool_index": False,
                }
            )
            try:
                with np.load(path / "metrics.npz") as arrays:
                    shard_infos[-1]["has_pool_index"] = "pool_index" in arrays.files
                    if "vp" in arrays.files:
                        vp = arrays["vp"]
                        vp_values.extend(float(x) for x in vp[np.isfinite(vp)])
                    if "apy" in arrays.files:
                        apy = arrays["apy"]
                        apy_values.extend(float(x) for x in apy[np.isfinite(apy) & (apy >= 0)])
            except Exception:
                pass

        total_pools = int(manifest.get("total_pools", total_runs))
        metric_names: list[str] = []
        if isinstance(metrics_schema, list):
            metric_names = [
                item["name"]
                for item in metrics_schema
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
        if not metric_names and shard_infos:
            with np.load(shard_infos[0]["metrics_path"]) as arrays:
                metric_names = list(arrays.files)
                metrics_schema = [
                    {"name": name, "dtype": str(arrays[name].dtype)}
                    for name in metric_names
                ]

        has_pool_index = [bool(info.get("has_pool_index")) for info in shard_infos]
        merge_by_pool_index = bool(has_pool_index) and all(has_pool_index)
        if any(has_pool_index) and not merge_by_pool_index:
            raise ValueError("mixed NPZ shards with and without pool_index.npy")
        if merge_by_pool_index:
            _validate_pool_index_shards(shard_infos, total_pools)
        else:
            _validate_contiguous_shards(shard_infos, total_pools)

        tmp_arrays = output_file / ".merge_tmp"
        tmp_arrays.mkdir()
        try:
            with zipfile.ZipFile(output_file / "metrics.npz", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for metric in metric_names:
                    dtype = None
                    for info in shard_infos:
                        with np.load(info["metrics_path"]) as arrays:
                            if metric in arrays.files:
                                dtype = arrays[metric].dtype
                                break
                    if dtype is None:
                        continue

                    tmp_npy = tmp_arrays / f"{metric}.npy"
                    merged_array = np.lib.format.open_memmap(
                        tmp_npy,
                        mode="w+",
                        dtype=dtype,
                        shape=(total_pools,),
                    )
                    if np.issubdtype(dtype, np.floating):
                        merged_array[:] = np.nan
                    else:
                        merged_array[:] = 0

                    for info in shard_infos:
                        with np.load(info["metrics_path"]) as arrays:
                            if metric not in arrays.files:
                                continue
                            arr = arrays[metric]
                            if merge_by_pool_index:
                                pool_index = arrays["pool_index"].astype(np.intp, copy=False)
                                if len(pool_index) != len(arr):
                                    raise ValueError(
                                        f"{info['blade']}: {metric} length {len(arr)} "
                                        f"does not match pool_index length {len(pool_index)}"
                                    )
                                merged_array[pool_index] = arr
                            else:
                                start_idx = int(info["pool_start"])
                                merged_array[start_idx:start_idx + len(arr)] = arr
                    merged_array.flush()
                    del merged_array
                    zf.write(tmp_npy, arcname=f"{metric}.npy")
                    tmp_npy.unlink()
        finally:
            shutil.rmtree(tmp_arrays, ignore_errors=True)

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "total_pools": total_runs,
            "n_pools": total_pools,
            "errors": total_errors,
            "blades_used": len(downloaded),
            "blade_stats": blade_stats,
            "harness_args": {
                "harness_binary": cfg.get("harness_binary"),
                "dustswapfreq": cfg.get("dustswap_freq"),
                "candle_filter": cfg.get("candle_filter"),
                "start_time": cfg.get("start_time"),
                "disable_slippage_probes": cfg.get("disable_slippage_probes"),
                "quiet_harness": cfg.get("quiet_harness"),
            },
            "grid": grid,
            "base_pool": manifest.get("base_pool", {}),
            "base_costs": manifest.get("base_costs", {}),
            "fee_equalize": manifest.get("fee_equalize", False),
        }
        if real_type is not None:
            metadata["real"] = real_type
        if cfg.get("start_time") is not None:
            metadata["start_time"] = cfg.get("start_time")
        remote_candles = manifest.get("remote_candles")
        if remote_candles:
            candles_name = Path(remote_candles).name
            local_candles = find_local_candles(candles_name)
            metadata["candles_file"] = str(local_candles) if local_candles else candles_name

        merged = {
            "format": "arb_npz_v1",
            "metrics_file": "metrics.npz",
            "metadata": metadata,
            "n_pools": total_pools,
            "pool_start": 0,
            "pool_end": total_pools,
            "metrics_schema": metrics_schema or [],
            "summary": {
                "total_runs": total_runs,
                "successful": total_runs - total_errors,
            },
        }
        if vp_values:
            merged["summary"]["vp"] = {
                "min": min(vp_values),
                "max": max(vp_values),
                "avg": sum(vp_values) / len(vp_values),
            }
        if apy_values:
            merged["summary"]["apy"] = {
                "min": min(apy_values),
                "max": max(apy_values),
                "avg": sum(apy_values) / len(apy_values),
            }

        with open(output_file / "manifest.json", "w") as f:
            json.dump(merged, f, indent=2)
        if merged_errors:
            with open(output_file / "errors.json", "w") as f:
                json.dump(merged_errors, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Results Summary")
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

    if all("pool_index" in run for run in all_runs):
        all_runs.sort(key=lambda run: int(run["pool_index"]))

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
                "start_time": cfg.get("start_time"),
                "disable_slippage_probes": cfg.get("disable_slippage_probes"),
                "quiet_harness": cfg.get("quiet_harness"),
            },
            # Include grid metadata for visualization compatibility
            "grid": grid,
            "base_pool": manifest.get("base_pool", {}),
            "base_costs": manifest.get("base_costs", {}),
            "fee_equalize": manifest.get("fee_equalize", False),
        },
        "runs": all_runs,
    }
    if cfg.get("start_time") is not None:
        merged["metadata"]["start_time"] = cfg.get("start_time")

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
