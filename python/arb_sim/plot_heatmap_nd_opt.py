#!/usr/bin/env python3
"""
Optimized N-dimensional heatmap explorer.

Key improvements vs plot_heatmap_nd.py:
- Precompute dense N-D metric arrays
- Use index-based slicing (O(1) per slider update)
- Avoid per-point scanning on updates
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import orjson
except Exception:
    orjson = None

# Force interactive backend
import matplotlib

for backend in ["macosx", "TkAgg", "Qt5Agg"]:
    try:
        matplotlib.use(backend)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.ticker import FormatStrFormatter

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "run_data"

LN2 = math.log(2)
SECONDS_PER_HOUR = 3600.0
MA_TIME_TO_HOURS = LN2 / SECONDS_PER_HOUR

# ==================== LAYOUT CONSTANTS ====================
CTRL_FIG_WIDTH = 3.2
CTRL_MIN_HEIGHT = 6.0
CTRL_HEIGHT_MULT = 11.0
CTRL_HEIGHT_PAD = 1.0

CTRL_TITLE_Y = 0.98
CTRL_TITLE_FONTSIZE = 10

RADIO_ITEM_HEIGHT = 0.03
RADIO_BOX_PADDING = 0.01
RADIO_FONTSIZE = 8
RADIO_X_LABEL_Y = 0.95
RADIO_LABEL_GAP = 0.01
RADIO_GROUP_GAP = 0.02

SLIDER_HEIGHT = 0.05
SLIDER_TOP_GAP = 0.02
SLIDER_LABEL_OFFSET = 0.015
SLIDER_BOX_LEFT = 0.20
SLIDER_BOX_WIDTH = 0.50
SLIDER_BOX_HEIGHT = 0.03
SLIDER_BOX_Y_OFFSET = 0.02
SLIDER_VALUE_X = 0.73
SLIDER_FONTSIZE = 8
# ==========================================================

DEFAULT_METRICS = [
    "apy_net",
    "apy_corr",
    "tw_real_slippage",
    "rel_price_diff_geom_mean",
    "virtual_price",
    "xcp_profit",
]

DEFAULT_COSTS = {
    "arb_fee_bps": 10.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1,
}

INSPECT_POOL_FILENAME = "inspect_pool.json"
INSPECT_OUTPUT_FILENAME = "inspect_output.json"
INSPECT_REAL = "double"
INSPECT_DUSTSWAPFREQ = 600
INSPECT_APY_PERIOD_DAYS = 1
INSPECT_APY_PERIOD_CAP = 20
INSPECT_THREADS = 10
INSPECT_DETAILED_INTERVAL = 1000


def _latest_arb_run() -> Path:
    files = list(RUN_DIR.glob("arb_run_*.json"))
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {RUN_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load(path: Path) -> Dict[str, Any]:
    if orjson is not None:
        return orjson.loads(path.read_bytes())
    with path.open("r") as f:
        return json.load(f)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_grid_dims(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Parse grid dimensions from metadata.
    Returns list of (dim_key, dim_name) tuples sorted by dim_key (x1, x2, ...).
    """
    grid = data.get("metadata", {}).get("grid", {})
    dims = []
    for key, val in grid.items():
        if isinstance(key, str) and key.lower().startswith("x") and key[1:].isdigit():
            idx = int(key[1:])
            name = val.get("name") if isinstance(val, dict) else None
            if name:
                dims.append((idx, key, name))
    dims.sort(key=lambda t: t[0])
    return [(key, name) for _, key, name in dims]


def _axis_normalization(name: str) -> Tuple[float, str]:
    key = (name or "").lower()
    if name == "A" or key == "a":
        return 1e4, " (÷1e4)"
    if "fee" in key and "gamma" not in key:
        return 0.0, " (bps)"
    if "ma_time" in key:
        return 1.0, " (hrs)" if "hrs" not in key else ""
    if "gamma" in key:
        return 1e18, " (/1e18)"
    if "liquidity" in key or "balance" in key:
        return 1e18, " (/1e18)"
    return 1.0, ""


def _format_axis_labels(name: str, values: List[float]) -> Tuple[List[str], str]:
    scale, suffix = _axis_normalization(name)
    key = (name or "").lower()
    display_name = name or ""
    if suffix and suffix not in display_name:
        display_name = f"{display_name}{suffix}"

    if scale == 0.0 and "fee" in key and "gamma" not in key:
        labels = [f"{(v / 1e10 * 1e4):.2f}" for v in values]
        return labels, f"{name} (bps)"
    if "ma_time" in key:
        labels = [f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}" for v in values]
        return labels, display_name
    if "gamma" in key:
        labels = [f"{(v / 1e18):.5f}" for v in values]
        return labels, name
    if scale != 1.0:
        labels = [f"{(v / scale):.2f}" for v in values]
        return labels, display_name
    labels = [f"{v:.4g}" for v in values]
    return labels, display_name


def _format_slider_value(name: str, value: float) -> str:
    scale, _ = _axis_normalization(name)
    key = (name or "").lower()
    if scale == 0.0 and "fee" in key and "gamma" not in key:
        return f"{(value / 1e10 * 1e4):.1f} bps"
    if name == "A" or key == "a":
        return f"{value / 1e4:.2f}"
    if "gamma" in key:
        return f"{value / 1e18:.6f}"
    if "apy" in key or "ratio" in key:
        return f"{value:.4f}"
    return f"{value:.4g}"


def _stringify_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_stringify_value(v) for v in value]
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _stringify_pool(pool: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _stringify_value(val) for key, val in pool.items()}


def _nearest_index(values: List[float], coord: float) -> int:
    if not values or not math.isfinite(coord):
        return 0
    arr = np.asarray(values, dtype=float)
    idx = int(np.clip(np.searchsorted(arr, coord), 0, len(arr) - 1))
    if idx > 0 and abs(arr[idx - 1] - coord) < abs(arr[idx] - coord):
        idx -= 1
    return idx


def _edges_from_centers(centers: List[float]) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    if c.size == 0:
        return np.array([0, 1])
    if c.size == 1:
        x0 = float(c[0])
        d = abs(x0) * 0.5 if x0 != 0 else 0.5
        return np.array([x0 - d, x0 + d])
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _select_ticks(values: List[float], max_ticks: int) -> List[int]:
    n = len(values)
    if n == 0:
        return []
    if max_ticks <= 0 or n <= max_ticks:
        return list(range(n))
    idxs = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    uniq = sorted(set(int(i) for i in idxs))
    if uniq[-1] != n - 1:
        uniq[-1] = n - 1
    return uniq


def _auto_font_size(nx: int, ny: int) -> int:
    grid = max(1, nx, ny)
    if grid <= 4:
        return 6
    if grid <= 8:
        return 7
    if grid <= 16:
        return 8
    if grid >= 32:
        return 10
    span = 32 - 16
    t = (grid - 16) / span
    size = 8 + t * (10 - 8)
    return int(round(size))


def _metric_scale_flags(metric: str) -> Tuple[bool, bool]:
    mlow = (metric or "").lower()
    scale_1e18 = metric in {
        "virtual_price",
        "xcp_profit",
        "price_scale",
        "D",
        "totalSupply",
    }
    scale_percent = (
        mlow in {"vpminusone", "apy"}
        or "apy" in mlow
        or "tw_real_slippage" in mlow
        or "geom_mean" in mlow
        or "rel_price_diff" in mlow
        or "tw_avg_pool_fee" in mlow
    )
    return scale_1e18, scale_percent


def _metric_scale_info(metric: str) -> Tuple[float, str]:
    scale_1e18, scale_percent = _metric_scale_flags(metric)
    factor = 1.0
    if scale_1e18:
        factor /= 1e18
    if scale_percent:
        factor *= 100.0
    suffix = " (%)" if scale_percent else ""
    return factor, suffix


def _extract_coord(run: Dict[str, Any], dim_names: List[str]) -> List[float] | None:
    pool = run.get("pool", {})
    coords: List[float] = []
    for i, name in enumerate(dim_names):
        raw = run.get(f"x{i + 1}_val")
        if raw is None:
            raw = pool.get(name)
        v = _to_float(raw) if raw is not None else float("nan")
        if not math.isfinite(v):
            return None
        coords.append(v)
    return coords


def _extract_nd_arrays(
    data: Dict[str, Any],
    metrics: List[str],
) -> Tuple[
    List[str],
    Dict[str, List[float]],
    Dict[str, np.ndarray],
    Dict[str, float],
    Dict[Tuple[int, ...], Dict[str, Any]],
]:
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] found in arb_run JSON")

    dims = _parse_grid_dims(data)
    if dims:
        dim_names = [name for _, name in dims]
    else:
        pool = runs[0].get("pool", {})
        if not pool:
            raise SystemExit("No grid dimensions found in metadata or pool")
        dim_names = sorted(pool.keys())

    dim_values: Dict[str, set] = {name: set() for name in dim_names}
    for r in runs:
        coords = _extract_coord(r, dim_names)
        if coords is None:
            continue
        for name, val in zip(dim_names, coords):
            dim_values[name].add(val)

    dim_values_sorted: Dict[str, List[float]] = {
        name: sorted(vals) for name, vals in dim_values.items()
    }
    dim_index: Dict[str, Dict[float, int]] = {
        name: {v: i for i, v in enumerate(vals)}
        for name, vals in dim_values_sorted.items()
    }

    shape = tuple(len(dim_values_sorted[name]) for name in dim_names)
    metric_arrays: Dict[str, np.ndarray] = {
        m: np.full(shape, np.nan, dtype=float) for m in metrics
    }
    metric_scale: Dict[str, float] = {m: _metric_scale_info(m)[0] for m in metrics}
    pool_configs: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    metric_items = [(m, metric_arrays[m], metric_scale[m]) for m in metrics]

    for r in runs:
        coords = _extract_coord(r, dim_names)
        if coords is None:
            continue
        idxs = []
        valid = True
        for name, val in zip(dim_names, coords):
            idx = dim_index[name].get(val)
            if idx is None:
                valid = False
                break
            idxs.append(idx)
        if not valid:
            continue
        idx_tuple = tuple(idxs)

        fs = r.get("final_state", {})
        res = r.get("result", {})
        metrics_dict = {**res, **fs}
        for metric, arr, factor in metric_items:
            val = metrics_dict.get(metric)
            if val is None:
                continue
            v = _to_float(val)
            if math.isfinite(v):
                arr[idx_tuple] = v * factor

        params = r.get("params", {})
        pool = params.get("pool") or r.get("pool")
        costs = params.get("costs")
        if pool:
            if costs is not None:
                pool_configs[idx_tuple] = {"pool": pool, "costs": costs}
            else:
                pool_configs[idx_tuple] = {"pool": pool}

    return dim_names, dim_values_sorted, metric_arrays, metric_scale, pool_configs


def _compute_global_clims(
    metric_arrays: Dict[str, np.ndarray],
) -> Dict[str, Tuple[float, float]]:
    clims: Dict[str, Tuple[float, float]] = {}
    for metric, arr in metric_arrays.items():
        if arr.size == 0:
            clims[metric] = (0.0, 1.0)
            continue
        try:
            zmin = float(np.nanmin(arr))
            zmax = float(np.nanmax(arr))
        except ValueError:
            clims[metric] = (0.0, 1.0)
            continue
        if not math.isfinite(zmin) or not math.isfinite(zmax):
            clims[metric] = (0.0, 1.0)
            continue
        if zmin == zmax:
            eps = 1e-12 if zmax == 0 else abs(zmax) * 1e-12
            zmin, zmax = zmin - eps, zmax + eps
        clims[metric] = (zmin, zmax)
    return clims


class NDHeatmapExplorerOpt:
    """Optimized N-dimensional heatmap explorer with fast slicing."""

    def __init__(
        self,
        data: Dict[str, Any],
        metrics: List[str],
        ncol: int,
        cmap: str,
        max_ticks: int,
    ):
        self.data = data
        self.metrics = metrics
        self.ncol = ncol
        self.cmap = cmap
        self.max_ticks = max_ticks

        (
            self.dim_names,
            self.dim_values,
            self.metric_arrays,
            self.metric_scale,
            self.pool_configs,
        ) = _extract_nd_arrays(data, metrics)
        self.n_dims = len(self.dim_names)

        if self.n_dims < 2:
            raise SystemExit("Need at least 2 dimensions for a heatmap")

        self.x_name = self.dim_names[0]
        self.y_name = self.dim_names[1]

        self.slider_indices: Dict[str, int] = {}
        self._init_slider_indices()

        self.global_clim = _compute_global_clims(self.metric_arrays)

        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        self.base_pool = meta.get("base_pool") if isinstance(meta, dict) else None
        if not isinstance(self.base_pool, dict):
            self.base_pool = {}
        self.candles_file = None
        if isinstance(meta, dict):
            self.candles_file = (
                meta.get("candles_file")
                or meta.get("datafile")
                or meta.get("remote_candles")
            )

        self.repo_root = Path(__file__).resolve().parents[2]
        self.python_dir = HERE.parent
        self.inspect_pool_path = RUN_DIR / INSPECT_POOL_FILENAME
        self.inspect_output_path = RUN_DIR / INSPECT_OUTPUT_FILENAME
        self._inspect_running = False
        self._inspect_built_once = False  # Track if we've built binary this session

        self.fig_main = None
        self.fig_controls = None
        self.axes = []
        self.meshes = []
        self.colorbars = []
        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []
        self.x_radio = None
        self.y_radio = None
        self._updating_radios = False

        self._setup_figures()

    def _init_slider_indices(self):
        self.slider_indices = {}
        for name in self.dim_names:
            if name not in (self.x_name, self.y_name):
                self.slider_indices[name] = 0

    def _get_slider_dims(self) -> List[Tuple[int, str]]:
        return [
            (i, name)
            for i, name in enumerate(self.dim_names)
            if name not in (self.x_name, self.y_name)
        ]

    def _slice_metric(self, metric: str) -> np.ndarray:
        arr = self.metric_arrays[metric]
        x_idx = self.dim_names.index(self.x_name)
        y_idx = self.dim_names.index(self.y_name)

        slicer: List[Any] = []
        for name in self.dim_names:
            if name == self.x_name or name == self.y_name:
                slicer.append(slice(None))
            else:
                slicer.append(self.slider_indices.get(name, 0))

        slice_arr = arr[tuple(slicer)]

        if x_idx < y_idx:
            slice_arr = slice_arr.T

        return slice_arr

    def _attach_format_coord(self, ax, xs, ys, mesh):
        """Attach a format_coord function to ax that shows x, y, z on hover."""
        xs_arr = np.array(xs)
        ys_arr = np.array(ys)

        def format_coord(x, y):
            if len(xs_arr) == 0 or len(ys_arr) == 0:
                return ""
            j = int(np.clip(np.searchsorted(xs_arr, x) - 0.5, 0, len(xs_arr) - 1))
            i = int(np.clip(np.searchsorted(ys_arr, y) - 0.5, 0, len(ys_arr) - 1))
            if j < len(xs_arr) - 1 and abs(x - xs_arr[j + 1]) < abs(x - xs_arr[j]):
                j += 1
            if i < len(ys_arr) - 1 and abs(y - ys_arr[i + 1]) < abs(y - ys_arr[i]):
                i += 1
            Z_arr = mesh.get_array()
            if Z_arr is not None:
                Z_2d = Z_arr.reshape(len(ys_arr), len(xs_arr))
                z_val = (
                    Z_2d[i, j]
                    if 0 <= i < Z_2d.shape[0] and 0 <= j < Z_2d.shape[1]
                    else float("nan")
                )
            else:
                z_val = float("nan")
            return f"x={xs_arr[j]:.4g}, y={ys_arr[i]:.4g}, z={z_val:.4g}"

        ax.format_coord = format_coord

    def _setup_figures(self):
        n = len(self.metrics)
        cols = min(self.ncol, n)
        rows = int(np.ceil(n / cols)) if n > 0 else 1

        max_fig_w = 22.0
        max_fig_h = 12.0

        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]

        cell_aspect = len(ys) / max(1, len(xs))
        cell_w = max_fig_w / cols
        cell_h = cell_w * cell_aspect
        fig_h = cell_h * rows

        if fig_h > max_fig_h:
            fig_h = max_fig_h
            cell_h = fig_h / rows
            cell_w = cell_h / cell_aspect
            fig_w = cell_w * cols
        else:
            fig_w = max_fig_w

        fig_w = max(10.0, min(max_fig_w, fig_w))
        fig_h = max(6.0, min(max_fig_h, fig_h))

        self.fig_main, axes_grid = plt.subplots(
            rows, cols, figsize=(fig_w, fig_h), constrained_layout=True, num="Heatmaps"
        )
        axes_grid = np.atleast_1d(axes_grid).reshape(rows, cols)

        base_font = _auto_font_size(len(xs), len(ys))
        tick_font = base_font
        label_font = max(8, base_font + 2)
        title_font = max(label_font, base_font + 4)
        colorbar_font = base_font

        xticks = _select_ticks(xs, self.max_ticks)
        yticks = _select_ticks(ys, self.max_ticks)
        xlab_full, xlabel = _format_axis_labels(self.x_name, xs)
        ylab_full, ylabel = _format_axis_labels(self.y_name, ys)
        xlabels = [xlab_full[i] for i in xticks]
        ylabels = [ylab_full[i] for i in yticks]

        Xedges = _edges_from_centers(xs)
        Yedges = _edges_from_centers(ys)

        self.axes = []
        self.meshes = []
        self.colorbars = []

        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes_grid[r, c]
                if idx >= n:
                    ax.axis("off")
                    continue

                metric = self.metrics[idx]
                Z = self._slice_metric(metric)

                mesh = ax.pcolormesh(Xedges, Yedges, Z, cmap=self.cmap, shading="auto")

                ny, nx = Z.shape
                try:
                    ax.set_box_aspect(ny / nx)
                except Exception:
                    ax.set_aspect("equal", adjustable="box")

                ax.set_xticks([xs[i] for i in xticks])
                ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=tick_font)
                if c == 0:
                    ax.set_yticks([ys[i] for i in yticks])
                    ax.set_yticklabels(ylabels, fontsize=tick_font)
                    ax.set_ylabel(ylabel, fontsize=label_font)
                else:
                    ax.set_yticks([ys[i] for i in yticks])
                    ax.set_yticklabels([])
                ax.set_xlabel(xlabel, fontsize=label_font)

                _, title_suffix = _metric_scale_info(metric)
                ax.set_title(f"{metric}{title_suffix}", fontsize=title_font)

                if metric in self.global_clim:
                    zmin, zmax = self.global_clim[metric]
                    mesh.set_clim(zmin, zmax)

                cb = self.fig_main.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(metric + title_suffix, fontsize=colorbar_font)
                cb.ax.tick_params(labelsize=tick_font)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

                self._attach_format_coord(ax, xs, ys, mesh)

                self.axes.append(ax)
                self.meshes.append(mesh)
                self.colorbars.append(cb)
                idx += 1

        self._setup_controls()
        self.fig_main.canvas.mpl_connect("button_press_event", self._on_click)

    def _setup_controls(self):
        slider_dims = self._get_slider_dims()
        n_sliders = len(slider_dims)
        n_dims = len(self.dim_names)

        radio_box_height = n_dims * RADIO_ITEM_HEIGHT + RADIO_BOX_PADDING

        total_content = (
            0.03
            + 2 * (RADIO_LABEL_GAP + radio_box_height + RADIO_GROUP_GAP)
            + n_sliders * SLIDER_HEIGHT
            + SLIDER_TOP_GAP
        )
        height_mult = CTRL_HEIGHT_MULT + max(0, n_dims - 5) * 0.3
        fig_height = max(CTRL_MIN_HEIGHT, total_content * height_mult + CTRL_HEIGHT_PAD)

        self.fig_controls = plt.figure(
            figsize=(CTRL_FIG_WIDTH, fig_height), num="Controls"
        )

        self.fig_controls.text(
            0.5,
            CTRL_TITLE_Y,
            "Dimension Controls",
            ha="center",
            va="top",
            fontsize=CTRL_TITLE_FONTSIZE,
            fontweight="bold",
        )

        x_label_y = RADIO_X_LABEL_Y
        x_box_top = x_label_y - RADIO_LABEL_GAP
        x_box_height = radio_box_height

        self.fig_controls.text(
            0.05, x_label_y, "X axis:", ha="left", va="top", fontsize=RADIO_FONTSIZE + 1
        )
        x_ax = self.fig_controls.add_axes(
            [SLIDER_BOX_LEFT, x_box_top - x_box_height, 0.75, x_box_height]
        )
        x_ax.set_frame_on(False)
        self.x_radio = RadioButtons(
            x_ax, self.dim_names, active=self.dim_names.index(self.x_name)
        )
        for label in self.x_radio.labels:
            label.set_fontsize(RADIO_FONTSIZE)
        self.x_radio.on_clicked(self._on_x_changed)

        y_label_y = x_box_top - x_box_height - RADIO_GROUP_GAP
        y_box_top = y_label_y - RADIO_LABEL_GAP
        y_box_height = radio_box_height

        self.fig_controls.text(
            0.05, y_label_y, "Y axis:", ha="left", va="top", fontsize=RADIO_FONTSIZE + 1
        )
        y_ax = self.fig_controls.add_axes(
            [SLIDER_BOX_LEFT, y_box_top - y_box_height, 0.75, y_box_height]
        )
        y_ax.set_frame_on(False)
        self.y_radio = RadioButtons(
            y_ax, self.dim_names, active=self.dim_names.index(self.y_name)
        )
        for label in self.y_radio.labels:
            label.set_fontsize(RADIO_FONTSIZE)
        self.y_radio.on_clicked(self._on_y_changed)

        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []

        slider_start_y = y_box_top - y_box_height - SLIDER_TOP_GAP

        for i, (_, dim_name) in enumerate(slider_dims):
            vals = self.dim_values[dim_name]
            slider_y = slider_start_y - i * SLIDER_HEIGHT

            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                f"{dim_name}:",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                len(vals) - 1,
                valinit=self.slider_indices.get(dim_name, 0),
                valstep=1,
            )
            slider.valtext.set_visible(False)

            current_idx = int(slider.val)
            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y,
                _format_slider_value(dim_name, vals[current_idx]),
                ha="left",
                va="center",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_value_texts.append(val_text)

            def make_update(name, vals_list, val_txt):
                def update(idx):
                    idx = int(idx)
                    self.slider_indices[name] = idx
                    val_txt.set_text(_format_slider_value(name, vals_list[idx]))
                    self._refresh_heatmaps()

                return update

            slider.on_changed(make_update(dim_name, vals, val_text))
            self.sliders.append((dim_name, slider))

        self.fig_controls.canvas.draw_idle()

    def _on_x_changed(self, label: str):
        if self._updating_radios:
            return
        if label == self.x_name:
            return

        if label == self.y_name:
            self._updating_radios = True
            old_x = self.x_name
            self.x_name = label
            self.y_name = old_x
            y_idx = self.dim_names.index(self.y_name)
            self.y_radio.set_active(y_idx)
            self._updating_radios = False
        else:
            self.x_name = label

        self._rebuild_sliders()
        self._rebuild_heatmaps()

    def _on_y_changed(self, label: str):
        if self._updating_radios:
            return
        if label == self.y_name:
            return

        if label == self.x_name:
            self._updating_radios = True
            old_y = self.y_name
            self.y_name = label
            self.x_name = old_y
            x_idx = self.dim_names.index(self.x_name)
            self.x_radio.set_active(x_idx)
            self._updating_radios = False
        else:
            self.y_name = label

        self._rebuild_sliders()
        self._rebuild_heatmaps()

    def _rebuild_sliders(self):
        for slider_ax in self.slider_axes:
            slider_ax.remove()
        for lbl in self.slider_labels:
            lbl.remove()
        for val_txt in self.slider_value_texts:
            val_txt.remove()
        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []

        new_slider_indices = {}
        for name in self.dim_names:
            if name not in (self.x_name, self.y_name):
                new_slider_indices[name] = min(
                    self.slider_indices.get(name, 0),
                    len(self.dim_values[name]) - 1,
                )
        self.slider_indices = new_slider_indices

        slider_dims = self._get_slider_dims()
        n_dims = len(self.dim_names)
        radio_box_height = n_dims * RADIO_ITEM_HEIGHT + RADIO_BOX_PADDING

        x_label_y = RADIO_X_LABEL_Y
        x_box_top = x_label_y - RADIO_LABEL_GAP
        y_label_y = x_box_top - radio_box_height - RADIO_GROUP_GAP
        y_box_top = y_label_y - RADIO_LABEL_GAP
        slider_start_y = y_box_top - radio_box_height - SLIDER_TOP_GAP

        for i, (_, dim_name) in enumerate(slider_dims):
            vals = self.dim_values[dim_name]
            slider_y = slider_start_y - i * SLIDER_HEIGHT
            current_idx = self.slider_indices.get(dim_name, 0)

            lbl = self.fig_controls.text(
                0.05,
                slider_y + SLIDER_LABEL_OFFSET,
                f"{dim_name}:",
                ha="left",
                va="bottom",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_labels.append(lbl)

            slider_ax = self.fig_controls.add_axes(
                [
                    SLIDER_BOX_LEFT,
                    slider_y - SLIDER_BOX_Y_OFFSET,
                    SLIDER_BOX_WIDTH,
                    SLIDER_BOX_HEIGHT,
                ]
            )
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                len(vals) - 1,
                valinit=current_idx,
                valstep=1,
            )
            slider.valtext.set_visible(False)

            val_text = self.fig_controls.text(
                SLIDER_VALUE_X,
                slider_y,
                _format_slider_value(dim_name, vals[current_idx]),
                ha="left",
                va="center",
                fontsize=SLIDER_FONTSIZE,
            )
            self.slider_value_texts.append(val_text)

            def make_update(name, vals_list, val_txt):
                def update(idx):
                    idx = int(idx)
                    self.slider_indices[name] = idx
                    val_txt.set_text(_format_slider_value(name, vals_list[idx]))
                    self._refresh_heatmaps()

                return update

            slider.on_changed(make_update(dim_name, vals, val_text))
            self.sliders.append((dim_name, slider))

        self.fig_controls.canvas.draw_idle()

    def _rebuild_heatmaps(self):
        for cb in self.colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self.colorbars = []
        self.meshes = []

        for ax in self.axes:
            ax.clear()

        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]

        base_font = _auto_font_size(len(xs), len(ys))
        tick_font = base_font
        label_font = max(8, base_font + 2)
        title_font = max(label_font, base_font + 4)
        colorbar_font = base_font

        xticks = _select_ticks(xs, self.max_ticks)
        yticks = _select_ticks(ys, self.max_ticks)
        xlab_full, xlabel = _format_axis_labels(self.x_name, xs)
        ylab_full, ylabel = _format_axis_labels(self.y_name, ys)
        xlabels = [xlab_full[i] for i in xticks]
        ylabels = [ylab_full[i] for i in yticks]

        Xedges = _edges_from_centers(xs)
        Yedges = _edges_from_centers(ys)

        n = len(self.metrics)
        cols = min(self.ncol, n)

        for idx, ax in enumerate(self.axes):
            if idx >= n:
                ax.axis("off")
                continue

            metric = self.metrics[idx]
            Z = self._slice_metric(metric)

            mesh = ax.pcolormesh(Xedges, Yedges, Z, cmap=self.cmap, shading="auto")

            ny, nx = Z.shape
            try:
                ax.set_box_aspect(ny / nx)
            except Exception:
                ax.set_aspect("equal", adjustable="box")

            ax.set_xticks([xs[i] for i in xticks])
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=tick_font)
            c = idx % cols
            if c == 0:
                ax.set_yticks([ys[i] for i in yticks])
                ax.set_yticklabels(ylabels, fontsize=tick_font)
                ax.set_ylabel(ylabel, fontsize=label_font)
            else:
                ax.set_yticks([ys[i] for i in yticks])
                ax.set_yticklabels([])
            ax.set_xlabel(xlabel, fontsize=label_font)

            _, title_suffix = _metric_scale_info(metric)
            ax.set_title(f"{metric}{title_suffix}", fontsize=title_font)

            if metric in self.global_clim:
                zmin, zmax = self.global_clim[metric]
                mesh.set_clim(zmin, zmax)

            cb = self.fig_main.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(metric + title_suffix, fontsize=colorbar_font)
            cb.ax.tick_params(labelsize=tick_font)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

            self._attach_format_coord(ax, xs, ys, mesh)

            self.meshes.append(mesh)
            self.colorbars.append(cb)

        self.fig_main.canvas.draw_idle()

    def _refresh_heatmaps(self):
        for idx, mesh in enumerate(self.meshes):
            if idx >= len(self.metrics):
                continue
            metric = self.metrics[idx]
            Z = self._slice_metric(metric)
            mesh.set_array(Z.ravel())
        self.fig_main.canvas.draw_idle()

    def _build_inspect_pool_config(
        self, coords: Dict[str, float], idx_tuple: Tuple[int, ...]
    ) -> Dict[str, Any] | None:
        config = self.pool_configs.get(idx_tuple)
        if config and "pool" in config:
            pool = _stringify_pool(config.get("pool", {}))
            costs = config.get("costs") or dict(DEFAULT_COSTS)
            return {"tag": "inspect", "pool": pool, "costs": costs}

        if not self.base_pool:
            return None

        pool = dict(self.base_pool)
        for name, val in coords.items():
            pool[name] = val
        return {
            "tag": "inspect",
            "pool": _stringify_pool(pool),
            "costs": dict(DEFAULT_COSTS),
        }

    def _resolve_candles_path(self) -> Path | None:
        if not self.candles_file:
            return None
        candles_path = Path(str(self.candles_file))
        if candles_path.is_absolute():
            return candles_path

        candidate = self.repo_root / candles_path
        if candidate.exists():
            return candidate

        candidate = HERE / candles_path
        if candidate.exists():
            return candidate

        name = candles_path.name
        trade_root = HERE / "trade_data"
        if trade_root.exists():
            matches = list(trade_root.rglob(name))
            if matches:
                return matches[0]

        return candidate

    def _write_inspect_pool_config(self, pool_config: Dict[str, Any]) -> Path:
        meta: Dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        candles_path = self._resolve_candles_path()
        if candles_path:
            meta["datafile"] = str(candles_path)
        if self.base_pool:
            meta["base_pool"] = self.base_pool
        payload = {"meta": meta, "pools": [pool_config]}
        self.inspect_pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.inspect_pool_path.write_text(json.dumps(payload, indent=2))
        return self.inspect_pool_path

    def _run_inspect_simulation(self, pool_config: Dict[str, Any]):
        if self._inspect_running:
            print("Inspect run already in progress; ignoring click.")
            return

        candles_path = self._resolve_candles_path()
        if not candles_path or not candles_path.exists():
            missing = self.candles_file or "(missing)"
            print(f"Candles file not found for inspect: {missing}")
            return

        self._inspect_running = True
        try:
            inspect_path = self._write_inspect_pool_config(pool_config)
            out_path = self.inspect_output_path

            cmd = [
                "uv",
                "run",
                "arb_sim/arb_sim.py",
                "--real",
                INSPECT_REAL,
                "--dustswapfreq",
                str(INSPECT_DUSTSWAPFREQ),
                "--apy-period-days",
                str(INSPECT_APY_PERIOD_DAYS),
                "--apy-period-cap",
                str(INSPECT_APY_PERIOD_CAP),
                "-n",
                str(INSPECT_THREADS),
                "--detailed-log",
                "--detailed-interval",
                str(INSPECT_DETAILED_INTERVAL),
                "--out",
                str(out_path),
                "--pool-config",
                str(inspect_path),
            ]
            # First click rebuilds, subsequent clicks skip build
            if self._inspect_built_once:
                cmd.append("--skip-build")
            cmd.append(str(candles_path))

            print("\nRunning inspect simulation...")
            subprocess.run(cmd, cwd=self.python_dir, check=True)
            self._inspect_built_once = True

            detailed_path = out_path.parent / "detailed-output.json"
            plot_cmd = [
                "uv",
                "run",
                "arb_sim/plot_price_scale.py",
                "--no-save",
                str(detailed_path),
            ]
            subprocess.Popen(plot_cmd, cwd=self.python_dir, start_new_session=True)
        except subprocess.CalledProcessError as exc:
            print(f"Inspect run failed: {exc}")
        finally:
            self._inspect_running = False

    def _on_click(self, event):
        is_shift_click = event.button == 1 and event.key == "shift"
        is_right_click = event.button == 3
        if not (is_shift_click or is_right_click):
            return
        if event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]
        x_idx = _nearest_index(xs, event.xdata)
        y_idx = _nearest_index(ys, event.ydata)

        indices: List[int] = []
        coords: Dict[str, float] = {}
        for name in self.dim_names:
            if name == self.x_name:
                idx = x_idx
                val = xs[x_idx]
            elif name == self.y_name:
                idx = y_idx
                val = ys[y_idx]
            else:
                idx = self.slider_indices.get(name, 0)
                val = self.dim_values[name][idx]
            indices.append(idx)
            coords[name] = val

        idx_tuple = tuple(indices)
        pool_config = self._build_inspect_pool_config(coords, idx_tuple)

        print("\nSelected point:")
        for name in self.dim_names:
            print(f"  {name}: {coords[name]}")

        if pool_config:
            print("Pool config:")
            print(json.dumps(pool_config, indent=2))
            self._run_inspect_simulation(pool_config)
        else:
            print("No pool config found for this point.")

    def show(self):
        print(f"\nGrid dimensions: {self.dim_names}")
        for name in self.dim_names:
            vals = self.dim_values[name]
            print(f"  {name}: {len(vals)} values ({vals[0]:.4g} .. {vals[-1]:.4g})")
        print(f"\nMetrics: {self.metrics}")
        print(f"X axis: {self.x_name}, Y axis: {self.y_name}")
        slider_dims = self._get_slider_dims()
        if slider_dims:
            print(f"Sliders: {[name for _, name in slider_dims]}")
        print("\nShift+click or right-click to run inspect sim.")
        print("Close both windows to exit.")
        plt.show()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Optimized N-dim heatmap explorer")
    ap.add_argument("--arb", type=str, default=None, help="Path to arb_run_*.json")
    ap.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=f"Comma-separated metrics (default: {','.join(DEFAULT_METRICS)})",
    )
    ap.add_argument("--cmap", type=str, default="turbo")
    ap.add_argument("--max-ticks", type=int, default=12)
    ap.add_argument("--ncol", type=int, default=3)
    args = ap.parse_args()

    arb_path = Path(args.arb) if args.arb else _latest_arb_run()
    print(f"Loading {arb_path}")
    data = _load(arb_path)

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = DEFAULT_METRICS

    explorer = NDHeatmapExplorerOpt(
        data=data,
        metrics=metrics,
        ncol=args.ncol,
        cmap=args.cmap,
        max_ticks=args.max_ticks,
    )
    explorer.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
