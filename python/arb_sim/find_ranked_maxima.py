#!/usr/bin/env python3
"""
Rank grid runs by aggregating per-metric ranks.

Examples:
  uv run --with orjson python arb_sim/find_ranked_maxima.py \
    --arb arb_sim/cluster_orchestration/results/cluster_sweep_latest.json \
    --desc-metrics apy_net --asc-metrics avg_rel_price_diff,tw_slippage_5pct
  uv run --with orjson python arb_sim/find_ranked_maxima.py \
    --desc-metrics apy_net --asc-metrics avg_rel_price_diff,tw_slippage_5pct \
    --weights apy_net=2
  uv run --with orjson python arb_sim/find_ranked_maxima.py \
    --desc-metrics apy_net --asc-metrics avg_rel_price_diff,tw_slippage_5pct \
    --asc-thr avg_rel_price_diff=0.001,tw_slippage_5pct=0.002
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import orjson
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "orjson is required. Run with: uv run --with orjson python arb_sim/find_ranked_maxima.py"
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


def _ordered_grid(metadata: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    grid = metadata.get("grid", {})
    if not isinstance(grid, dict):
        return [], []
    keys = sorted(
        (k for k in grid.keys() if isinstance(k, str) and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )
    names = []
    for k in keys:
        entry = grid.get(k)
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
        else:
            names.append(k)
    return names, keys


def _run_coord_values(
    run: Dict[str, Any],
    x_keys: List[str],
    names: List[str],
) -> List[float] | None:
    if x_keys:
        x_val_keys = [f"{k}_val" for k in x_keys]
        if all(k in run for k in x_val_keys):
            values: List[float] = []
            for k in x_val_keys:
                num = _to_float(run.get(k))
                if num is None:
                    return None
                values.append(num)
            return values

    pool = run.get("pool")
    if not isinstance(pool, dict):
        params = run.get("params")
        if isinstance(params, dict):
            pool = params.get("pool")

    if isinstance(pool, dict) and names:
        values = []
        for name in names:
            num = _to_float(pool.get(name))
            if num is None:
                return None
            values.append(num)
        return values

    return None


def _format_coord(name: str, val: float) -> str | float:
    if name == "A":
        return f"{val / 10_000:4.2f}"
    if name in {"mid_fee", "out_fee"}:
        return f"{int(val / 10**10 * 10_000)}"
    if name == "donation_apy":
        return f"{val:4.3f}"
    if name == "fee_gamma":
        return f"{val / 1e18:0.6f}"
    return val


def _rank_values(
    values: List[float],
    descending: bool,
    good_threshold: float | None,
    good_when_low: bool,
) -> List[int]:
    ranks: List[int] = [0] * len(values)
    good_flags = [False] * len(values)

    if good_threshold is not None:
        for i, v in enumerate(values):
            if not math.isfinite(v):
                continue
            if good_when_low and v <= good_threshold:
                good_flags[i] = True
            if not good_when_low and v >= good_threshold:
                good_flags[i] = True

    valid: List[Tuple[int, float]] = [
        (i, v) for i, v in enumerate(values) if math.isfinite(v) and not good_flags[i]
    ]
    valid.sort(key=lambda item: item[1], reverse=descending)

    current_rank = 1
    for idx, _ in valid:
        ranks[idx] = current_rank
        current_rank += 1

    worst_rank = current_rank if current_rank > 1 else 1
    for i, v in enumerate(values):
        if not math.isfinite(v):
            ranks[i] = worst_rank
    return ranks


def _parse_metric_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _parse_weights(raw: str | None) -> Dict[str, float]:
    if not raw:
        return {}
    weights: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"Invalid weight '{part}'. Use metric=weight")
        name, val = part.split("=", 1)
        name = name.strip()
        val = val.strip()
        if not name:
            raise SystemExit(f"Invalid weight '{part}'. Use metric=weight")
        try:
            weight = float(val)
        except ValueError:
            raise SystemExit(f"Invalid weight '{part}'. Use metric=weight")
        if weight <= 0:
            raise SystemExit(f"Weight for '{name}' must be > 0")
        weights[name] = weight
    return weights


def _parse_thresholds(raw: str | None) -> Dict[str, float]:
    if not raw:
        return {}
    thresholds: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"Invalid threshold '{part}'. Use metric=value")
        name, val = part.split("=", 1)
        name = name.strip()
        val = val.strip()
        if not name:
            raise SystemExit(f"Invalid threshold '{part}'. Use metric=value")
        try:
            thr = float(val)
        except ValueError:
            raise SystemExit(f"Invalid threshold '{part}'. Use metric=value")
        thresholds[name] = thr
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arb", type=Path, default=None, help="Path to arb_run JSON")
    parser.add_argument(
        "--asc-metrics",
        "--asc_metrics",
        type=str,
        default="",
        help="Comma-separated metrics to minimize",
    )
    parser.add_argument(
        "--desc-metrics",
        "--desc_metrics",
        type=str,
        default="",
        help="Comma-separated metrics to maximize",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated metric=weight (default weight 1.0)",
    )
    parser.add_argument(
        "--asc-thr",
        "--asc_thr",
        type=str,
        default="",
        help="Comma-separated metric=threshold (good if <= threshold)",
    )
    parser.add_argument(
        "--desc-thr",
        "--desc_thr",
        type=str,
        default="",
        help="Comma-separated metric=threshold (good if >= threshold)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Limit output to top N ranks (<=0 for all)",
    )
    args = parser.parse_args()

    asc_metrics = _parse_metric_list(args.asc_metrics)
    desc_metrics = _parse_metric_list(args.desc_metrics)
    if not asc_metrics and not desc_metrics:
        raise SystemExit(
            "Provide at least one metric via --asc-metrics or --desc-metrics"
        )

    overlap = set(asc_metrics) & set(desc_metrics)
    if overlap:
        raise SystemExit(f"Metrics cannot be in both asc and desc: {sorted(overlap)}")

    weights = _parse_weights(args.weights)
    asc_thresholds = _parse_thresholds(args.asc_thr)
    desc_thresholds = _parse_thresholds(args.desc_thr)

    arb_path = args.arb or _latest_arb_run()
    print(f"Loading {arb_path}")
    data = _load(arb_path)

    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs found in JSON")

    names, x_keys = _ordered_grid(data.get("metadata", {}))
    if not names:
        first_pool = runs[0].get("pool") or runs[0].get("params", {}).get("pool")
        if isinstance(first_pool, dict):
            names = sorted(first_pool.keys())

    entries = []
    for run in runs:
        if run.get("success") is False:
            continue
        env = _build_env(run)
        coord_vals = _run_coord_values(run, x_keys, names)
        if coord_vals is None:
            continue
        coords = {
            name: _format_coord(name, coord_vals[i]) for i, name in enumerate(names)
        }
        entries.append(
            {
                "coords": coords,
                "env": env,
            }
        )

    if not entries:
        raise SystemExit("No valid runs found after filtering")

    metric_names = asc_metrics + desc_metrics
    unknown_weights = set(weights) - set(metric_names)
    if unknown_weights:
        raise SystemExit(
            f"Weights provided for unknown metrics: {sorted(unknown_weights)}"
        )
    unknown_asc_thr = set(asc_thresholds) - set(asc_metrics)
    if unknown_asc_thr:
        raise SystemExit(
            f"Thresholds provided for unknown asc metrics: {sorted(unknown_asc_thr)}"
        )
    unknown_desc_thr = set(desc_thresholds) - set(desc_metrics)
    if unknown_desc_thr:
        raise SystemExit(
            f"Thresholds provided for unknown desc metrics: {sorted(unknown_desc_thr)}"
        )
    metric_weights = {m: weights.get(m, 1.0) for m in metric_names}
    values_by_metric: Dict[str, List[float]] = {m: [] for m in metric_names}

    for entry in entries:
        env = entry["env"]
        for m in metric_names:
            v = env.get(m)
            values_by_metric[m].append(v if v is not None else float("nan"))

    for m, vals in values_by_metric.items():
        if not any(math.isfinite(v) for v in vals):
            raise SystemExit(f"Metric '{m}' has no finite values")

    ranks_by_metric: Dict[str, List[int]] = {}
    for m in asc_metrics:
        ranks_by_metric[m] = _rank_values(
            values_by_metric[m],
            descending=False,
            good_threshold=asc_thresholds.get(m),
            good_when_low=True,
        )
    for m in desc_metrics:
        ranks_by_metric[m] = _rank_values(
            values_by_metric[m],
            descending=True,
            good_threshold=desc_thresholds.get(m),
            good_when_low=False,
        )

    ranked = []
    for i, entry in enumerate(entries):
        total_rank = sum(
            ranks_by_metric[m][i] * metric_weights[m] for m in metric_names
        )
        metrics = {m: values_by_metric[m][i] for m in metric_names}
        ranked.append(
            {
                "rank_score": total_rank,
                "metrics": metrics,
                "coords": entry["coords"],
            }
        )

    ranked.sort(key=lambda r: r["rank_score"])

    print(f"asc_metrics={asc_metrics}")
    print(f"desc_metrics={desc_metrics}")
    print(f"weights={metric_weights}")
    print(f"asc_thresholds={asc_thresholds}")
    print(f"desc_thresholds={desc_thresholds}")
    print(f"runs={len(ranked)}")

    limit = len(ranked) if args.top <= 0 else min(args.top, len(ranked))
    for idx, row in enumerate(ranked[:limit], start=1):
        metrics_str = ", ".join(
            f"{k}={row['metrics'][k]:.6g}"
            if math.isfinite(row["metrics"][k])
            else f"{k}=nan"
            for k in metric_names
        )
        print(
            f"rank #{idx} score={row['rank_score']} metrics={{ {metrics_str} }} coords={row['coords']}"
        )


if __name__ == "__main__":
    main()
