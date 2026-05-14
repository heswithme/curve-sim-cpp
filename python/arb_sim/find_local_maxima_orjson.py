#!/usr/bin/env python3
"""
Fast local-max finder for large arb_run JSON files (full load).

This version loads the entire JSON into RAM using orjson for speed.
Use the streamed script if you need lower memory usage.

Examples:
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --arb arb_sim/run_data/arb_run_1.json
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric apy_mask_5 --local --top 10 --enumerate
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric apy_masked --pricethr 100 --local
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric apy_masked --pricethr 100 --max-rel-pdiff --local
  uv run --with orjson python arb_sim/find_local_maxima_orjson.py --metric tw_real_slippage_5pct --min --local
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from grid_axes import infer_grid_from_runs, pool_for_run, pool_value
from arb_run_npz import NpzRun, load_json_or_npz, ordered_grid

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
    files = [
        p
        for p in RUN_DIR.glob("arb_run_*")
        if p.is_dir() or p.suffix == ".json"
    ]
    if not files:
        raise SystemExit(f"No arb_run_* result found under {RUN_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load(path: Path) -> Dict[str, Any]:
    return load_json_or_npz(path)


def _ordered_grid(
    metadata: Dict[str, Any],
) -> Tuple[List[str], List[List[float]], List[str]]:
    grid = metadata.get("grid", {})
    if not isinstance(grid, dict):
        return [], [], []
    keys = sorted(
        (k for k in grid.keys() if isinstance(k, str) and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )
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


def _add_env(env: Dict[str, float], src: Any, prefix: str = "") -> None:
    if not isinstance(src, dict):
        return
    for key, val in src.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            _add_env(env, val, name)
            continue
        num = _to_float(val)
        if num is not None and name not in env:
            env[name] = num


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
    if x_keys and all(k in run for k in x_val_keys):
        values: List[float] = []
        for k in x_val_keys:
            num = _to_float(run.get(k))
            if num is None:
                raise ValueError(f"Non-numeric {k} in run")
            values.append(num)
        return values

    pool = pool_for_run(run)

    if pool:
        values = []
        for name in names:
            num = _to_float(pool_value(pool, name))
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
    price_metric_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(v, dtype=float) for v in grid_values]
    shape = [len(a) for a in arrays]
    metric = np.full(shape, np.nan, dtype=float)
    pdiff_bps = np.full(shape, np.nan, dtype=float)

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
        pdiff = env.get(price_metric_key)
        if pdiff is not None:
            pdiff_bps[tuple(idx)] = pdiff * 10000.0
        if metric_key == "apy_masked":
            apy_net = env.get("apy_net")
            rel = env.get(price_metric_key)
            if apy_net is None or rel is None:
                raise KeyError(
                    f"Metric 'apy_masked' requires apy_net and {price_metric_key}"
                )
            thr = price_thr_bps / 10000.0
            metric[tuple(idx)] = apy_net if rel <= thr else np.nan
        else:
            if metric_key not in env:
                raise KeyError(f"Metric '{metric_key}' not found in run data")
            metric[tuple(idx)] = env[metric_key]

    return metric, pdiff_bps


def _build_metric_grid_npz(
    npz_run: NpzRun,
    metadata: Dict[str, Any],
    metric_key: str,
    price_thr_bps: float,
    price_metric_key: str,
) -> Tuple[List[str], List[List[float]], List[str], np.ndarray, np.ndarray]:
    names, grid_values, x_keys = ordered_grid(metadata)
    if not names:
        raise SystemExit("NPZ arb run requires metadata.grid")
    shape = tuple(len(v) for v in grid_values)

    def load(name: str) -> np.ndarray:
        return npz_run.load_array(name).astype(float, copy=False).reshape(shape)

    try:
        pdiff_raw = load(price_metric_key)
    except KeyError as exc:
        raise KeyError(f"Metric '{price_metric_key}' not found in NPZ run") from exc
    pdiff_bps = pdiff_raw * 10000.0

    if metric_key == "apy_masked":
        try:
            metric = load("apy_net")
        except KeyError as exc:
            raise KeyError("Metric 'apy_masked' requires apy_net") from exc
        metric = np.where(pdiff_raw <= price_thr_bps / 10000.0, metric, np.nan)
    else:
        try:
            metric = load(metric_key)
        except KeyError as exc:
            raise KeyError(f"Metric '{metric_key}' not found in NPZ run") from exc

    try:
        success = npz_run.load_array("success").astype(bool).reshape(shape)
        metric = np.where(success, metric, np.nan)
        pdiff_bps = np.where(success, pdiff_bps, np.nan)
    except KeyError:
        pass

    return names, grid_values, x_keys, metric, pdiff_bps


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
        elif "fee_bps" in name:
            formatted[name] = str(f"{val:4.2f}")
        elif name in {"mid_fee", "out_fee"}:
            disp = int(val / 10**10 * 10_000)
            formatted[name] = str(f"{disp:d}")
        elif name == "donation_apy":
            formatted[name] = str(f"{val:4.3f}")
        elif name in {"reserved_profit_fraction", "admin_fee"}:
            formatted[name] = str(f"{val / 1e10:0.4f}")
        elif name.endswith("_wad"):
            formatted[name] = str(f"{val / 1e18:0.6f}")
        elif name == "fee_gamma":
            formatted[name] = str(f"{val / 1e18:0.6f}")
        else:
            formatted[name] = val
    return formatted


def _pdiff_label(price_metric_key: str) -> str:
    if price_metric_key == "max_rel_price_diff":
        return "max_pdiff_bps"
    if price_metric_key == "max_7d_rel_price_diff":
        return "max_7d_pdiff_bps"
    return "avg_pdiff_bps"


def _pdiff_text(pdiff_grid: np.ndarray, coord: Tuple[int, ...], label: str) -> str:
    pdiff = float(pdiff_grid[coord])
    if not np.isfinite(pdiff):
        return ""
    return f" {label}={pdiff:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arb", type=Path, default=None, help="Path to arb_run JSON or NPZ directory")
    parser.add_argument("--metric", type=str, default="apy_mask_3", help="Metric key")
    parser.add_argument(
        "--pricethr",
        type=float,
        default=100.0,
        help="Max price-diff metric in bps for apy_masked",
    )
    parser.add_argument(
        "--price-metric",
        choices=[
            "avg_rel_price_diff",
            "max_rel_price_diff",
            "max_7d_rel_price_diff",
        ],
        default="avg_rel_price_diff",
        help="Price-diff metric used by apy_masked",
    )
    parser.add_argument(
        "--max-rel-pdiff",
        dest="price_metric",
        action="store_const",
        const="max_rel_price_diff",
        help="Alias for --price-metric max_rel_price_diff",
    )
    parser.add_argument(
        "--max-7d-pdiff",
        dest="price_metric",
        action="store_const",
        const="max_7d_rel_price_diff",
        help="Alias for --price-metric max_7d_rel_price_diff",
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

    npz_run = data.get("_npz_run")
    if isinstance(npz_run, NpzRun):
        names, grid_values, x_keys, metric, pdiff_bps = _build_metric_grid_npz(
            npz_run,
            data.get("metadata", {}),
            args.metric,
            args.pricethr,
            args.price_metric,
        )
    else:
        runs = data.get("runs", [])
        if not runs:
            raise SystemExit("No runs found in JSON")

        names, grid_values, x_keys = _ordered_grid(data.get("metadata", {}))
        if not names:
            names, grid_values, x_keys = infer_grid_from_runs(runs)
        if not names:
            raise SystemExit("No grid dimensions found in metadata or params.pool")
        metric, pdiff_bps = _build_metric_grid(
            runs,
            names,
            x_keys,
            grid_values,
            args.metric,
            args.pricethr,
            args.price_metric,
        )
    pdiff_label = _pdiff_label(args.price_metric)
    finite_metric = np.where(np.isfinite(metric), metric, np.nan)
    if not np.isfinite(finite_metric).any():
        raise SystemExit("No finite metric values found")

    if args.min:
        search_metric = np.where(np.isfinite(metric), -metric, -np.inf)
        min_idx = np.unravel_index(int(np.argmax(search_metric)), search_metric.shape)
        min_val = float(metric[min_idx])
        min_coord = _coord_dict(min_idx, names, grid_values)
        pdiff = _pdiff_text(pdiff_bps, min_idx, pdiff_label)
        print(f"global_min {args.metric}={min_val:.6f}{pdiff} coords={min_coord}")

        if args.local:
            coords = _local_maxima(search_metric, args.connectivity)
            coords_sorted = sorted(coords, key=lambda c: metric[c])
            print(f"local_minima_count {len(coords_sorted)}")
            limit = len(coords_sorted) if args.top <= 0 else args.top
            for rank, coord in enumerate(coords_sorted[:limit], start=1):
                val = float(metric[coord])
                coord_dict = _coord_dict(coord, names, grid_values)
                pdiff = _pdiff_text(pdiff_bps, coord, pdiff_label)
                if args.enumerate:
                    print(
                        f"local_min #{rank} {args.metric}={val:.6f}{pdiff} coords={coord_dict}"
                    )
                else:
                    print(f"local_min {args.metric}={val:.6f}{pdiff} coords={coord_dict}")
    else:
        metric = np.where(np.isfinite(metric), metric, -np.inf)
        max_idx = np.unravel_index(int(np.argmax(metric)), metric.shape)
        max_val = float(metric[max_idx])
        max_coord = _coord_dict(max_idx, names, grid_values)
        pdiff = _pdiff_text(pdiff_bps, max_idx, pdiff_label)
        print(f"global_max {args.metric}={max_val:.6f}{pdiff} coords={max_coord}")

        if args.local:
            coords = _local_maxima(metric, args.connectivity)
            coords_sorted = sorted(coords, key=lambda c: metric[c], reverse=True)
            print(f"local_maxima_count {len(coords_sorted)}")
            limit = len(coords_sorted) if args.top <= 0 else args.top
            for rank, coord in enumerate(coords_sorted[:limit], start=1):
                val = float(metric[coord])
                coord_dict = _coord_dict(coord, names, grid_values)
                pdiff = _pdiff_text(pdiff_bps, coord, pdiff_label)
                if args.enumerate:
                    print(
                        f"local_max #{rank} {args.metric}={val:.6f}{pdiff} coords={coord_dict}"
                    )
                else:
                    print(f"local_max {args.metric}={val:.6f}{pdiff} coords={coord_dict}")


if __name__ == "__main__":
    main()
