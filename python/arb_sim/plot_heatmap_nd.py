#!/usr/bin/env python3
"""
Interactive N-dimensional heatmap explorer.

- Loads aggregated arb_run JSON with x1..xN grid dimensions.
- Plots a grid of 2D heatmaps (one per metric), mirroring plot_heatmap.py layout.
- Separate controls window with dropdowns for X/Y axis selection and sliders
  for remaining dimensions.

Usage:
  uv run python arb_sim/plot_heatmap_nd.py
  uv run python arb_sim/plot_heatmap_nd.py --metrics apy_net,apy_corr,tw_real_slippage
  uv run python arb_sim/plot_heatmap_nd.py --arb path/to/arb_run.json --ncol 4
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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

# Default metrics to display
DEFAULT_METRICS = [
    "apy_net",
    "apy_corr",
    "tw_real_slippage",
    "rel_price_diff_geom_mean",
    "virtual_price",
    "xcp_profit",
]


def _latest_arb_run() -> Path:
    files = list(RUN_DIR.glob("arb_run_*.json"))
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {RUN_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _axis_transformer(name: str):
    """Return a transformer for axis values based on the parameter name."""
    key = (name or "").lower()
    if "ma_time" in key:
        return lambda value: value * MA_TIME_TO_HOURS
    return lambda value: value


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


def _extract_nd_grid(
    data: Dict[str, Any],
) -> Tuple[
    List[str], Dict[str, List[float]], Dict[Tuple[float, ...], Dict[str, float]]
]:
    """
    Extract N-dimensional grid data for all metrics at once.

    Returns:
        dim_names: list of dimension names in order (x1, x2, ...)
        dim_values: dict of dim_name -> sorted unique values
        points: dict of (v1, v2, ..., vN) -> {metric_name: z_value}
    """
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] found in arb_run JSON")

    # Get dimension info
    dims = _parse_grid_dims(data)
    if not dims:
        raise SystemExit("No grid dimensions found in metadata")

    dim_keys = [key for key, _ in dims]
    dim_names = [name for _, name in dims]
    n_dims = len(dim_names)

    # Collect values per dimension and all metric values
    dim_values: Dict[str, set] = {name: set() for name in dim_names}
    points: Dict[Tuple[float, ...], Dict[str, Any]] = {}

    for r in runs:
        coords = []
        valid = True
        for i, name in enumerate(dim_names):
            key = f"x{i + 1}_val"
            raw = r.get(key)
            v = _to_float(raw) if raw is not None else float("nan")
            if not math.isfinite(v):
                valid = False
                break
            coords.append(v)
            dim_values[name].add(v)

        if not valid:
            continue

        # Store all metrics from final_state and result
        coord_tuple = tuple(coords)
        fs = r.get("final_state", {})
        res = r.get("result", {})
        merged = {**res, **fs}  # final_state takes precedence
        points[coord_tuple] = merged

    if not points:
        raise SystemExit("No valid data points found")

    # Sort values per dimension
    dim_values_sorted = {name: sorted(vals) for name, vals in dim_values.items()}

    return dim_names, dim_values_sorted, points


def _metric_scale_flags(m: str) -> Tuple[bool, bool]:
    """Return (scale_1e18, scale_percent) for a metric."""
    mlow = (m or "").lower()
    scale_1e18 = m in {
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
    )
    return scale_1e18, scale_percent


def _axis_normalization(name: str) -> Tuple[float, str]:
    """Return (scale_factor, unit_suffix) for axis values."""
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
    """Format a value for slider display."""
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


def _edges_from_centers(centers: List[float]) -> np.ndarray:
    """Compute bin edges from bin centers for pcolormesh."""
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
    """Choose a tick font size based on grid resolution. Smaller for interactive use."""
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


class NDHeatmapExplorer:
    """Interactive N-dimensional heatmap explorer with separate controls window."""

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

        # Extract grid data
        self.dim_names, self.dim_values, self.points = _extract_nd_grid(data)
        self.n_dims = len(self.dim_names)

        if self.n_dims < 2:
            raise SystemExit("Need at least 2 dimensions for a heatmap")

        # Current X/Y axes (default to first two dimensions)
        self.x_name = self.dim_names[0]
        self.y_name = self.dim_names[1]

        # Slider values for non-X/Y dimensions
        self.slider_values: Dict[str, float] = {}
        self._init_slider_values()

        # Compute global min/max per metric (across ALL slices)
        self.global_clim: Dict[str, Tuple[float, float]] = {}
        self._compute_global_clims()

        # Figure handles
        self.fig_main = None
        self.fig_controls = None
        self.axes = []
        self.meshes = []
        self.colorbars = []
        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []  # Track slider label text objects
        self.slider_value_texts = []  # Track slider value text objects
        self.x_radio = None
        self.y_radio = None
        self._updating_radios = False  # Prevent recursive radio updates

        self._setup_figures()

    def _init_slider_values(self):
        """Initialize slider values for all dimensions except current X/Y."""
        self.slider_values = {}
        for name in self.dim_names:
            if name not in (self.x_name, self.y_name):
                self.slider_values[name] = self.dim_values[name][0]

    def _compute_global_clims(self):
        """Compute global min/max for each metric across ALL data points."""
        self.global_clim = {}
        for metric in self.metrics:
            scale_1e18, scale_percent = _metric_scale_flags(metric)
            all_z = []
            for coord, metrics_dict in self.points.items():
                z = _to_float(metrics_dict.get(metric, float("nan")))
                if scale_1e18 and math.isfinite(z):
                    z = z / 1e18
                if scale_percent and math.isfinite(z):
                    z = z * 100.0
                if math.isfinite(z):
                    all_z.append(z)
            if all_z:
                zmin, zmax = min(all_z), max(all_z)
                if zmin == zmax:
                    eps = 1e-12 if zmax == 0 else abs(zmax) * 1e-12
                    zmin, zmax = zmin - eps, zmax + eps
                self.global_clim[metric] = (zmin, zmax)
            else:
                self.global_clim[metric] = (0.0, 1.0)

    def _get_slider_dims(self) -> List[Tuple[int, str]]:
        """Get dimensions that need sliders (all except X and Y)."""
        return [
            (i, name)
            for i, name in enumerate(self.dim_names)
            if name not in (self.x_name, self.y_name)
        ]

    def _build_slice(self, metric: str) -> np.ndarray:
        """Build 2D Z array for current X/Y and slider positions."""
        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]
        x_idx = self.dim_names.index(self.x_name)
        y_idx = self.dim_names.index(self.y_name)
        slider_dims = self._get_slider_dims()

        Z = np.full((len(ys), len(xs)), float("nan"))

        scale_1e18, scale_percent = _metric_scale_flags(metric)

        for coord, metrics_dict in self.points.items():
            # Check if this point matches slider values
            match = True
            for dim_idx, dim_name in slider_dims:
                expected = self.slider_values.get(dim_name)
                if expected is None:
                    continue
                actual = coord[dim_idx]
                if actual != expected:
                    match = False
                    break
            if not match:
                continue

            # Get z value
            z = _to_float(metrics_dict.get(metric, float("nan")))
            if scale_1e18 and math.isfinite(z):
                z = z / 1e18
            if scale_percent and math.isfinite(z):
                z = z * 100.0

            if not math.isfinite(z):
                continue

            # Get x and y indices
            x_val = coord[x_idx]
            y_val = coord[y_idx]
            try:
                xi = xs.index(x_val)
                yi = ys.index(y_val)
                Z[yi, xi] = z
            except ValueError:
                pass

        return Z

    def _setup_figures(self):
        """Create main heatmap figure and controls figure."""
        # Main figure with heatmap grid
        n = len(self.metrics)
        cols = min(self.ncol, n)
        rows = int(np.ceil(n / cols)) if n > 0 else 1

        # Target 27" screen full size: ~24x13 inches usable at 100 DPI
        # Leave room for window chrome and controls window
        max_fig_w = 22.0
        max_fig_h = 12.0

        # Calculate ideal size based on grid
        xs = self.dim_values[self.x_name]
        ys = self.dim_values[self.y_name]

        # Each subplot wants roughly square aspect based on data
        cell_aspect = len(ys) / max(1, len(xs))  # height/width ratio of data

        # Start with max width, compute height
        cell_w = max_fig_w / cols
        cell_h = cell_w * cell_aspect
        fig_h = cell_h * rows

        # If too tall, constrain by height instead
        if fig_h > max_fig_h:
            fig_h = max_fig_h
            cell_h = fig_h / rows
            cell_w = cell_h / cell_aspect
            fig_w = cell_w * cols
        else:
            fig_w = max_fig_w

        # Ensure minimum size
        fig_w = max(10.0, min(max_fig_w, fig_w))
        fig_h = max(6.0, min(max_fig_h, fig_h))

        self.fig_main, axes_grid = plt.subplots(
            rows, cols, figsize=(fig_w, fig_h), constrained_layout=True, num="Heatmaps"
        )
        axes_grid = np.atleast_1d(axes_grid).reshape(rows, cols)

        # Font sizes
        base_font = _auto_font_size(len(xs), len(ys))
        tick_font = base_font
        label_font = max(8, base_font + 2)
        title_font = max(label_font, base_font + 4)
        colorbar_font = base_font

        # Precompute tick indices and labels
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

                m = self.metrics[idx]
                Z = self._build_slice(m)

                # Plot
                mesh = ax.pcolormesh(Xedges, Yedges, Z, cmap=self.cmap, shading="auto")

                # Aspect ratio
                ny, nx = Z.shape
                try:
                    ax.set_box_aspect(ny / nx)
                except Exception:
                    ax.set_aspect("equal", adjustable="box")

                # Ticks
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

                # Title
                _, scale_percent = _metric_scale_flags(m)
                _, scale_1e18 = _metric_scale_flags(m)
                title_scale = " (%)" if scale_percent else ""
                ax.set_title(f"{m}{title_scale}", fontsize=title_font)

                # Color limits - use global min/max
                if m in self.global_clim:
                    zmin, zmax = self.global_clim[m]
                    mesh.set_clim(zmin, zmax)

                # Colorbar
                cb = self.fig_main.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(
                    m + (" (%)" if scale_percent else ""), fontsize=colorbar_font
                )
                cb.ax.tick_params(labelsize=tick_font)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

                self.axes.append(ax)
                self.meshes.append(mesh)
                self.colorbars.append(cb)
                idx += 1

        # Controls figure
        self._setup_controls()

    def _setup_controls(self):
        """Create separate controls window with dropdowns and sliders."""
        slider_dims = self._get_slider_dims()
        n_sliders = len(slider_dims)
        n_dims = len(self.dim_names)

        # Calculate heights based on content
        radio_item_height = 0.06  # per dimension option
        radio_box_height = n_dims * radio_item_height + 0.03
        slider_height_each = 0.08

        # Total height calculation
        total_content = (
            0.1
            + 2 * (0.05 + radio_box_height + 0.04)
            + n_sliders * slider_height_each
            + 0.08
        )
        fig_height = max(3.5, total_content * 6)

        self.fig_controls = plt.figure(figsize=(5, fig_height), num="Controls")

        # Title
        self.fig_controls.text(
            0.5,
            0.97,
            "Dimension Controls",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

        # X axis radio buttons
        x_label_y = 0.91
        x_box_top = x_label_y - 0.03
        x_box_height = radio_box_height

        self.fig_controls.text(
            0.05, x_label_y, "X axis:", ha="left", va="top", fontsize=10
        )
        x_ax = self.fig_controls.add_axes(
            [0.20, x_box_top - x_box_height, 0.75, x_box_height]
        )
        x_ax.set_frame_on(False)
        self.x_radio = RadioButtons(
            x_ax, self.dim_names, active=self.dim_names.index(self.x_name)
        )
        for label in self.x_radio.labels:
            label.set_fontsize(9)
        self.x_radio.on_clicked(self._on_x_changed)

        # Y axis radio buttons
        y_label_y = x_box_top - x_box_height - 0.05
        y_box_top = y_label_y - 0.03
        y_box_height = radio_box_height

        self.fig_controls.text(
            0.05, y_label_y, "Y axis:", ha="left", va="top", fontsize=10
        )
        y_ax = self.fig_controls.add_axes(
            [0.20, y_box_top - y_box_height, 0.75, y_box_height]
        )
        y_ax.set_frame_on(False)
        self.y_radio = RadioButtons(
            y_ax, self.dim_names, active=self.dim_names.index(self.y_name)
        )
        for label in self.y_radio.labels:
            label.set_fontsize(9)
        self.y_radio.on_clicked(self._on_y_changed)

        # Sliders for remaining dimensions
        self.sliders = []
        self.slider_axes = []
        self.slider_labels = []
        self.slider_value_texts = []  # Separate text elements for values

        slider_start_y = y_box_top - y_box_height - 0.07

        for i, (dim_idx, dim_name) in enumerate(slider_dims):
            vals = self.dim_values[dim_name]
            slider_y = slider_start_y - i * slider_height_each

            # Label
            lbl = self.fig_controls.text(
                0.05,
                slider_y + 0.02,
                f"{dim_name}:",
                ha="left",
                va="bottom",
                fontsize=9,
            )
            self.slider_labels.append(lbl)

            # Slider - shorter to leave room for value text
            slider_ax = self.fig_controls.add_axes([0.20, slider_y - 0.025, 0.55, 0.04])
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                len(vals) - 1,
                valinit=0,
                valstep=1,
            )
            # Hide built-in value text
            slider.valtext.set_visible(False)

            # Separate value text element
            val_text = self.fig_controls.text(
                0.78,
                slider_y,
                _format_slider_value(dim_name, vals[0]),
                ha="left",
                va="center",
                fontsize=9,
            )
            self.slider_value_texts.append(val_text)

            # Update callback
            def make_update(name, vals_list, val_txt):
                def update(idx):
                    idx = int(idx)
                    val = vals_list[idx]
                    self.slider_values[name] = val
                    formatted = _format_slider_value(name, val)
                    val_txt.set_text(formatted)
                    self._refresh_heatmaps()

                return update

            slider.on_changed(make_update(dim_name, vals, val_text))
            self.sliders.append((dim_name, slider))

        self.fig_controls.canvas.draw_idle()

    def _on_x_changed(self, label: str):
        """Handle X axis selection change."""
        if self._updating_radios:
            return
        if label == self.x_name:
            return  # No change

        if label == self.y_name:
            # Swap X and Y
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
        """Handle Y axis selection change."""
        if self._updating_radios:
            return
        if label == self.y_name:
            return  # No change

        if label == self.x_name:
            # Swap X and Y
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
        """Rebuild sliders after X/Y change."""
        # Clear old sliders, labels, and value texts
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

        # Update slider values dict
        new_slider_values = {}
        for name in self.dim_names:
            if name not in (self.x_name, self.y_name):
                if name in self.slider_values:
                    new_slider_values[name] = self.slider_values[name]
                else:
                    new_slider_values[name] = self.dim_values[name][0]
        self.slider_values = new_slider_values

        # Recreate sliders
        slider_dims = self._get_slider_dims()
        n_dims = len(self.dim_names)

        # Calculate positions matching _setup_controls
        radio_item_height = 0.06
        radio_box_height = n_dims * radio_item_height + 0.03
        slider_height_each = 0.08

        x_label_y = 0.91
        x_box_top = x_label_y - 0.03
        y_label_y = x_box_top - radio_box_height - 0.05
        y_box_top = y_label_y - 0.03
        slider_start_y = y_box_top - radio_box_height - 0.07

        for i, (dim_idx, dim_name) in enumerate(slider_dims):
            vals = self.dim_values[dim_name]
            slider_y = slider_start_y - i * slider_height_each

            # Find current index
            current_val = self.slider_values.get(dim_name, vals[0])
            try:
                current_idx = vals.index(current_val)
            except ValueError:
                current_idx = 0
                self.slider_values[dim_name] = vals[0]

            # Label
            lbl = self.fig_controls.text(
                0.05,
                slider_y + 0.02,
                f"{dim_name}:",
                ha="left",
                va="bottom",
                fontsize=9,
            )
            self.slider_labels.append(lbl)

            # Slider - shorter to leave room for value text
            slider_ax = self.fig_controls.add_axes([0.20, slider_y - 0.025, 0.55, 0.04])
            self.slider_axes.append(slider_ax)

            slider = Slider(
                slider_ax,
                "",
                0,
                len(vals) - 1,
                valinit=current_idx,
                valstep=1,
            )
            # Hide built-in value text
            slider.valtext.set_visible(False)

            # Separate value text element
            val_text = self.fig_controls.text(
                0.78,
                slider_y,
                _format_slider_value(dim_name, vals[current_idx]),
                ha="left",
                va="center",
                fontsize=9,
            )
            self.slider_value_texts.append(val_text)

            def make_update(name, vals_list, val_txt):
                def update(idx):
                    idx = int(idx)
                    val = vals_list[idx]
                    self.slider_values[name] = val
                    formatted = _format_slider_value(name, val)
                    val_txt.set_text(formatted)
                    self._refresh_heatmaps()

                return update

            slider.on_changed(make_update(dim_name, vals, val_text))
            self.sliders.append((dim_name, slider))

        self.fig_controls.canvas.draw_idle()

    def _rebuild_heatmaps(self):
        """Completely rebuild heatmaps after X/Y axis change."""
        # Clear and rebuild main figure
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

        # Remove old colorbars
        for cb in self.colorbars:
            cb.remove()
        self.colorbars = []
        self.meshes = []

        n = len(self.metrics)
        cols = min(self.ncol, n)

        for idx, ax in enumerate(self.axes):
            if idx >= n:
                ax.axis("off")
                continue

            m = self.metrics[idx]
            Z = self._build_slice(m)

            # Plot
            mesh = ax.pcolormesh(Xedges, Yedges, Z, cmap=self.cmap, shading="auto")

            # Aspect ratio
            ny, nx = Z.shape
            try:
                ax.set_box_aspect(ny / nx)
            except Exception:
                ax.set_aspect("equal", adjustable="box")

            # Ticks
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

            # Title
            _, scale_percent = _metric_scale_flags(m)
            title_scale = " (%)" if scale_percent else ""
            ax.set_title(f"{m}{title_scale}", fontsize=title_font)

            # Color limits - use global min/max
            if m in self.global_clim:
                zmin, zmax = self.global_clim[m]
                mesh.set_clim(zmin, zmax)

            # Colorbar
            cb = self.fig_main.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(m + (" (%)" if scale_percent else ""), fontsize=colorbar_font)
            cb.ax.tick_params(labelsize=tick_font)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

            self.meshes.append(mesh)
            self.colorbars.append(cb)

        self.fig_main.canvas.draw_idle()

    def _refresh_heatmaps(self):
        """Update heatmap data without rebuilding axes (for slider changes)."""
        for idx, (ax, mesh) in enumerate(zip(self.axes, self.meshes)):
            if idx >= len(self.metrics):
                continue

            m = self.metrics[idx]
            Z = self._build_slice(m)

            mesh.set_array(Z.ravel())
            # Color limits stay fixed (global clim already set)

        self.fig_main.canvas.draw_idle()

    def show(self):
        """Display the interactive explorer."""
        print(f"\nGrid dimensions: {self.dim_names}")
        for name in self.dim_names:
            vals = self.dim_values[name]
            print(f"  {name}: {len(vals)} values ({vals[0]:.4g} .. {vals[-1]:.4g})")
        print(f"\nMetrics: {self.metrics}")
        print(f"X axis: {self.x_name}, Y axis: {self.y_name}")
        slider_dims = self._get_slider_dims()
        if slider_dims:
            print(f"Sliders: {[name for _, name in slider_dims]}")
        print("\nClose both windows to exit.")
        plt.show()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Interactive N-dim heatmap explorer")
    ap.add_argument("--arb", type=str, default=None, help="Path to arb_run_*.json")
    ap.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=f"Comma-separated metrics (default: {','.join(DEFAULT_METRICS)})",
    )
    ap.add_argument(
        "--cmap", type=str, default="turbo", help="Colormap (default: turbo)"
    )
    ap.add_argument(
        "--max-ticks", type=int, default=12, help="Max ticks per axis (default: 12)"
    )
    ap.add_argument(
        "--ncol", type=int, default=3, help="Number of columns (default: 3)"
    )
    args = ap.parse_args()

    # Load data
    arb_path = Path(args.arb) if args.arb else _latest_arb_run()
    print(f"Loading {arb_path}")
    data = _load(arb_path)

    # Parse metrics
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = DEFAULT_METRICS

    # Create and show explorer
    explorer = NDHeatmapExplorer(
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
