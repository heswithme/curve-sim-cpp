#!/usr/bin/env python3
"""
Fast local-max finder for large arb_run JSON files (full load).

This version loads the entire JSON into RAM using orjson for speed.
Use the streamed script if you need lower memory usage.

Examples:
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --arb arb_sim/run_data/arb_run_1.json
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric apy_mask_5 --local --top 10 --enumerate
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric apy_masked --pricethr 100 --local
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric tw_real_slippage_5pct --min --local
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import orjson
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "orjson is required. Run with: uv run --with orjson python arb_sim/find_local_maxima_orjson.py"
    ) from exc


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "run_data"


def _latest_arb_run() -> Path:
    files = list(RUN_DIR.glob("arb_run_*.json"))
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {RUN_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load(path: Path) -> Dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _ordered_grid(
    metadata: Dict[str, Any],
) -> Tuple[List[str], List[List[float]], List[str]]:
    grid = metadata.get("grid", {})
    keys = sorted(grid.keys(), key=lambda k: int(k[1:]))
    names = [grid[k]["name"] for k in keys]
    values = [grid[k]["values"] for k in keys]
    return names, values, keys


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _add_env(env: Dict[str, float], src: Any) -> None:
    if not isinstance(src, dict):
        return
    for key, val in src.items():
        num = _to_float(val)
        if num is not None and key not in env:
            env[key] = num


def _build_env(run: Dict[str, Any]) -> Dict[str, float]:
    env: Dict[str, float] = {}
    for src in (run.get("result", {}), run.get("final_state", {})):
        _add_env(env, src)
    _add_env(env, run.get("pool"))
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    _add_env(env, params.get("pool"))
    _add_env(env, params.get("costs"))
    i = 1
    while True:
        kkey = f"x{i}_key"
        kval = f"x{i}_val"
        if kkey not in run:
            break
        name = run.get(kkey)
        num = _to_float(run.get(kval))
        if isinstance(name, str) and num is not None and name not in env:
            env[name] = num
        i += 1
    return env


def _index_for_value(values: np.ndarray, raw: Any) -> int:
    val = float(raw)
    idx = int(np.argmin(np.abs(values - val)))
    return idx


def _run_coord_values(
    run: Dict[str, Any],
    x_keys: List[str],
    names: List[str],
) -> List[float]:
    x_val_keys = [f"{k}_val" for k in x_keys]
    if all(k in run for k in x_val_keys):
        values: List[float] = []
        for k in x_val_keys:
            num = _to_float(run.get(k))
            if num is None:
                raise ValueError(f"Non-numeric {k} in run")
            values.append(num)
        return values

    pool = run.get("pool")
    if not isinstance(pool, dict):
        params = run.get("params")
        if isinstance(params, dict):
            pool = params.get("pool")

    if isinstance(pool, dict):
        values = []
        for name in names:
            num = _to_float(pool.get(name))
            if num is None:
                raise ValueError(f"Non-numeric pool value for {name}")
            values.append(num)
        return values

    raise ValueError("Run does not contain grid coordinates")


def _build_metric_grid(
    runs: Iterable[Dict[str, Any]],
    names: List[str],
    x_keys: List[str],
    grid_values: List[List[float]],
    metric_key: str,
    price_thr_bps: float,
) -> np.ndarray:
    arrays = [np.asarray(v, dtype=float) for v in grid_values]
    shape = [len(a) for a in arrays]
    metric = np.full(shape, np.nan, dtype=float)

    for run in runs:
        if run.get("success") is False:
            continue
        if not isinstance(run.get("result"), dict):
            continue
        coord_vals = _run_coord_values(run, x_keys, names)
        idx = []
        for i, arr in enumerate(arrays):
            idx.append(_index_for_value(arr, coord_vals[i]))
        env = _build_env(run)
        if metric_key == "apy_masked":
            apy_net = env.get("apy_net")
            avg_rel = env.get("avg_rel_price_diff")
            if apy_net is None or avg_rel is None:
                raise KeyError(
                    "Metric 'apy_masked' requires apy_net and avg_rel_price_diff"
                )
            thr = price_thr_bps / 10000.0
            metric[tuple(idx)] = apy_net if avg_rel <= thr else np.nan
        else:
            if metric_key not in env:
                raise KeyError(f"Metric '{metric_key}' not found in run data")
            metric[tuple(idx)] = env[metric_key]

    return metric


def _local_maxima(metric: np.ndarray, connectivity: str) -> List[Tuple[int, ...]]:
    conn = 1 if connectivity == "axis" else metric.ndim
    foot = ndi.generate_binary_structure(metric.ndim, conn)
    max_filt = ndi.maximum_filter(metric, footprint=foot, mode="nearest")
    max_mask = np.isfinite(metric) & (metric == max_filt)
    labels, nlab = ndi.label(max_mask, structure=foot)
    coords: List[Tuple[int, ...]] = []
    for lab in range(1, nlab + 1):
        pts = np.argwhere(labels == lab)
        vals = metric[tuple(pts.T)]
        best = pts[int(np.argmax(vals))]
        coords.append(tuple(int(x) for x in best))
    return coords


def _coord_dict(
    coord: Tuple[int, ...],
    names: List[str],
    grid_values: List[List[float]],
) -> Dict[str, float | str]:
    formatted: Dict[str, float | str] = {}
    for i, name in enumerate(names):
        val = float(grid_values[i][coord[i]])
        if name == "A":
            disp = val / 10_000
            formatted[name] = str(f"{disp:4.2f}")
        elif name in {"mid_fee", "out_fee"}:
            disp = int(val / 10**10 * 10_000)
            formatted[name] = str(f"{disp:d}")
        elif name == "donation_apy":
            formatted[name] = str(f"{val:4.3f}")
        elif name == "fee_gamma":
            formatted[name] = str(f"{val / 1e18:0.6f}")
        else:
            formatted[name] = val
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arb", type=Path, default=None, help="Path to arb_run JSON")
    parser.add_argument("--metric", type=str, default="apy_mask_3", help="Metric key")
    parser.add_argument(
        "--pricethr",
        type=float,
        default=100.0,
        help="Max avg_rel_price_diff in bps for apy_masked",
    )
    parser.add_argument(
        "--local", action="store_true", help="Report local maxima on grid"
    )
    parser.add_argument(
        "--min", action="store_true", help="Find minima instead of maxima"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of local maxima to print (<=0 for all)",
    )
    parser.add_argument(
        "--connectivity",
        choices=["axis", "full"],
        default="full",
        help="Neighbor definition for local maxima",
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Prefix local maxima with rank",
    )
    args = parser.parse_args()

    arb_path = args.arb or _latest_arb_run()
    print(f"Loading {arb_path}")
    data = _load(arb_path)

    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs found in JSON")

    names, grid_values, x_keys = _ordered_grid(data.get("metadata", {}))
    metric = _build_metric_grid(
        runs, names, x_keys, grid_values, args.metric, args.pricethr
    )
    finite_metric = np.where(np.isfinite(metric), metric, np.nan)
    if not np.isfinite(finite_metric).any():
        raise SystemExit("No finite metric values found")

    if args.min:
        search_metric = np.where(np.isfinite(metric), -metric, -np.inf)
        min_idx = np.unravel_index(int(np.argmax(search_metric)), search_metric.shape)
        min_val = float(metric[min_idx])
        min_coord = _coord_dict(min_idx, names, grid_values)
        print(f"global_min {args.metric}={min_val:.6f} coords={min_coord}")

        if args.local:
            coords = _local_maxima(search_metric, args.connectivity)
            coords_sorted = sorted(coords, key=lambda c: metric[c])
            print(f"local_minima_count {len(coords_sorted)}")
            limit = len(coords_sorted) if args.top <= 0 else args.top
            for rank, coord in enumerate(coords_sorted[:limit], start=1):
                val = float(metric[coord])
                coord_dict = _coord_dict(coord, names, grid_values)
                if args.enumerate:
                    print(
                        f"local_min #{rank} {args.metric}={val:.6f} coords={coord_dict}"
                    )
                else:
                    print(f"local_min {args.metric}={val:.6f} coords={coord_dict}")
    else:
        metric = np.where(np.isfinite(metric), metric, -np.inf)
        max_idx = np.unravel_index(int(np.argmax(metric)), metric.shape)
        max_val = float(metric[max_idx])
        max_coord = _coord_dict(max_idx, names, grid_values)
        print(f"global_max {args.metric}={max_val:.6f} coords={max_coord}")

        if args.local:
            coords = _local_maxima(metric, args.connectivity)
            coords_sorted = sorted(coords, key=lambda c: metric[c], reverse=True)
            print(f"local_maxima_count {len(coords_sorted)}")
            limit = len(coords_sorted) if args.top <= 0 else args.top
            for rank, coord in enumerate(coords_sorted[:limit], start=1):
                val = float(metric[coord])
                coord_dict = _coord_dict(coord, names, grid_values)
                if args.enumerate:
                    print(
                        f"local_max #{rank} {args.metric}={val:.6f} coords={coord_dict}"
                    )
                else:
                    print(f"local_max {args.metric}={val:.6f} coords={coord_dict}")


if __name__ == "__main__":
    main()
