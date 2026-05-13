#!/usr/bin/env python3
"""YieldBasis releverage post-pass for twocrypto detailed traces.

The input is the C++ harness detailed-output.json stream. The script can run
one AMM fee or scan a fee grid, then optionally plot fee APY and deposit growth.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np


HERE = Path(__file__).resolve().parent
SECONDS_PER_YEAR = 365.0 * 86400.0
MAX_ANNUALIZED_LOG = 700.0

ProfitField = Literal["profit", "vp", "vp_boosted", "xcp"]
FeeGrid = Literal["log", "linear"]


@dataclass(frozen=True)
class SimulationConfig:
    fee: float
    leverage: float
    ext_fee: float
    donation_apy: float
    path_every: int


@dataclass(frozen=True)
class Trace:
    t: np.ndarray
    token0: np.ndarray
    token1: np.ndarray
    high: np.ndarray
    low: np.ndarray
    peg: np.ndarray
    profit: np.ndarray
    donation_apy: float | None
    peg_to: str
    profit_field: ProfitField

    @property
    def duration(self) -> float:
        if len(self.t) < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])


class ReleverageAMM:
    def __init__(
        self,
        collateral: float,
        leverage: float,
        fee: float,
        oracle: float,
    ) -> None:
        self.collateral = collateral
        self.leverage = leverage
        self.fee = fee
        self.p_oracle = oracle
        self.debt = collateral * oracle * (leverage - 1.0) / leverage

    def set_p_oracle(self, p: float) -> None:
        self.p_oracle = p

    def get_x0(self) -> float:
        lev_ratio = (self.leverage / (2.0 * self.leverage - 1.0)) ** 2
        disc = (
            self.p_oracle * self.p_oracle * self.collateral * self.collateral
            - 4.0 * self.p_oracle * self.collateral * self.debt * lev_ratio
        )
        if disc < 0.0:
            if disc > -1e-14:
                disc = 0.0
            else:
                raise ValueError(f"negative AMM discriminant: {disc}")
        return (self.p_oracle * self.collateral + math.sqrt(disc)) / (
            2.0 * lev_ratio
        )

    def get_p(self) -> float:
        x0 = self.get_x0()
        return (x0 - self.debt) / self.collateral

    def trade_to_price(self, p: float) -> bool:
        if p <= 0.0:
            return False

        x0 = self.get_x0()
        initial_price = (x0 - self.debt) / self.collateral

        if p > initial_price:
            p *= 1.0 - self.fee
            if p <= initial_price:
                return False
        elif p < initial_price:
            p /= 1.0 - self.fee
            if p >= initial_price:
                return False
        else:
            return False

        inv = (x0 - self.debt) * self.collateral
        x_after = math.sqrt(inv * p)
        y_after = x_after / p

        if p > initial_price:
            x_after += (x_after - (x0 - self.debt)) * self.fee
        else:
            y_after += (y_after - self.collateral) * self.fee

        self.collateral = y_after
        self.debt = x0 - x_after
        return True

    def get_value(self) -> float:
        x0 = self.get_x0()
        ip = (x0 - self.debt) * self.collateral * self.p_oracle
        return 2.0 * math.sqrt(max(ip, 0.0)) - x0


def _json_load(path: Path) -> Any:
    if path.suffix in {".xz", ".lzma"}:
        with lzma.open(path, "rt") as f:
            return json.load(f)
    with path.open("rb") as f:
        try:
            import orjson

            return orjson.loads(f.read())
        except ImportError:
            f.seek(0)
            return json.load(f)


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field {field!r} is not numeric: {value!r}") from exc


def _profit_value(entry: dict[str, Any], field: ProfitField) -> float:
    value = _as_float(entry[field], field)
    if field == "profit":
        return value
    return value - 1.0


def load_trace(path: Path, peg_to: str, profit_field: ProfitField) -> Trace:
    data = _json_load(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    n = len(data)
    t = np.empty(n, dtype=np.float64)
    token0 = np.empty(n, dtype=np.float64)
    token1 = np.empty(n, dtype=np.float64)
    high = np.empty(n, dtype=np.float64)
    low = np.empty(n, dtype=np.float64)
    peg = np.empty(n, dtype=np.float64)
    profit = np.empty(n, dtype=np.float64)
    donation_apy = None
    if isinstance(data[0], dict) and "donation_apy" in data[0]:
        donation_apy = _as_float(data[0]["donation_apy"], "donation_apy")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"trace row {i} is not an object")
        if donation_apy is not None:
            if "donation_apy" not in row:
                raise ValueError(f"trace row {i} is missing donation_apy")
            row_donation_apy = _as_float(row["donation_apy"], "donation_apy")
            if not math.isclose(
                row_donation_apy, donation_apy, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(
                    "detailed trace has changing donation_apy: "
                    f"row 0={donation_apy}, row {i}={row_donation_apy}"
                )
        t[i] = _as_float(row["t"], "t")
        token0[i] = _as_float(row["token0"], "token0")
        token1[i] = _as_float(row["token1"], "token1")
        high[i] = _as_float(row["high"], "high")
        low[i] = _as_float(row["low"], "low")
        peg[i] = _as_float(row[peg_to], peg_to)
        profit[i] = _profit_value(row, profit_field)

    return Trace(
        t=t,
        token0=token0,
        token1=token1,
        high=high,
        low=low,
        peg=peg,
        profit=profit,
        donation_apy=donation_apy,
        peg_to=peg_to,
        profit_field=profit_field,
    )


def run_releverage(trace: Trace, config: SimulationConfig) -> dict[str, Any]:
    if len(trace.t) < 2:
        raise ValueError("need at least two detailed rows")
    if config.fee < 0.0 or config.fee >= 1.0:
        raise ValueError("--fee must be in [0, 1)")
    if config.leverage <= 1.0:
        raise ValueError("--leverage must be greater than 1")
    if config.ext_fee < 0.0 or config.ext_fee >= 1.0:
        raise ValueError("--ext-fee must be in [0, 1)")
    if config.donation_apy < 0.0:
        raise ValueError("donation_apy must be >= 0")

    amm = ReleverageAMM(
        collateral=config.leverage,
        leverage=config.leverage,
        fee=config.fee,
        oracle=1.0,
    )
    initial_value = amm.get_value()

    ema0 = float(trace.peg[0])
    v0 = float(trace.token0[0] + trace.token1[0] * trace.low[0])
    if ema0 <= 0.0 or v0 <= 0.0:
        raise ValueError("initial peg and pool value must be positive")

    donation_rate = float(config.donation_apy) / SECONDS_PER_YEAR

    t_prev = float(trace.t[0])
    final_growth = 1.0
    n_trades = 0
    path: list[dict[str, float]] = []

    t_arr = trace.t
    token0 = trace.token0
    token1 = trace.token1
    high_arr = trace.high
    low_arr = trace.low
    peg_arr = trace.peg
    profit_arr = trace.profit

    for i in range(len(t_arr)):
        t = float(t_arr[i])
        high = float(high_arr[i])
        low = float(low_arr[i])
        if high <= 0.0 or low <= 0.0 or high < low:
            raise ValueError(f"invalid OHLC range at row {i}: low={low}, high={high}")

        pool_profit = 1.0 + float(profit_arr[i])
        ema = math.sqrt(float(peg_arr[i]) / ema0)
        amm.set_p_oracle(ema * pool_profit)

        r = math.sqrt(high / low)
        low_bound = float(token0[i] + token1[i] * low) / v0
        high_bound = low_bound * r
        high_bound *= 1.0 - config.ext_fee
        low_bound *= 1.0 + config.ext_fee

        if high_bound > amm.get_p() and amm.trade_to_price(high_bound):
            n_trades += 1
        if low_bound < amm.get_p() and amm.trade_to_price(low_bound):
            n_trades += 1

        dt = max(0.0, t - t_prev)
        if donation_rate:
            amm.debt *= 1.0 + 2.0 * donation_rate * dt
        t_prev = t

        current_value = amm.get_value() / (ema**config.leverage)
        final_growth = current_value / initial_value
        if config.path_every > 0 and (
            i == 0 or i == len(t_arr) - 1 or i % config.path_every == 0
        ):
            path.append(
                {
                    "t": t,
                    "growth": final_growth,
                    "amm_price": amm.get_p(),
                    "oracle": amm.p_oracle,
                    "low_bound": low_bound,
                    "high_bound": high_bound,
                    "collateral": amm.collateral,
                    "debt": amm.debt,
                }
            )

    apy = math.nan
    if trace.duration > 0.0 and final_growth > 0.0:
        annualized_log = math.log(final_growth) * SECONDS_PER_YEAR / trace.duration
        if annualized_log > MAX_ANNUALIZED_LOG:
            apy = math.exp(MAX_ANNUALIZED_LOG)
        else:
            apy = math.expm1(annualized_log)

    return {
        "fee": config.fee,
        "apy": apy,
        "final_growth": final_growth,
        "final_value": final_growth * initial_value,
        "initial_value": initial_value,
        "n_releverage_trades": n_trades,
        "final_collateral": amm.collateral,
        "final_debt": amm.debt,
        "final_amm_price": amm.get_p(),
        "path": path,
    }


def make_fee_grid(
    fee: float | None,
    fee_min: float,
    fee_max: float,
    fee_count: int,
    grid: FeeGrid,
    scan: bool,
) -> list[float]:
    if fee is not None and not scan:
        return [fee]
    if fee_count <= 0:
        raise ValueError("--fee-count must be positive")
    if fee_min <= 0.0 and grid == "log":
        raise ValueError("--fee-min must be positive for log grids")
    if fee_max < fee_min:
        raise ValueError("--fee-max must be >= --fee-min")
    if grid == "log":
        return [
            float(x)
            for x in np.logspace(math.log10(fee_min), math.log10(fee_max), fee_count)
        ]
    return [float(x) for x in np.linspace(fee_min, fee_max, fee_count)]


def build_payload(
    trace_path: Path,
    trace: Trace,
    runs: list[dict[str, Any]],
    config: argparse.Namespace,
    donation_apy: float,
) -> dict[str, Any]:
    best_run = max(runs, key=lambda row: row["apy"])
    path = best_run.get("path", [])
    compact_runs = [{k: v for k, v in row.items() if k != "path"} for row in runs]
    best = {k: v for k, v in best_run.items() if k != "path"}
    best["path"] = path
    return {
        "metadata": {
            "trace": str(trace_path),
            "rows": int(len(trace.t)),
            "start_timestamp": float(trace.t[0]),
            "end_timestamp": float(trace.t[-1]),
            "duration_days": trace.duration / 86400.0,
            "peg_to": trace.peg_to,
            "profit_field": trace.profit_field,
            "leverage": config.leverage,
            "ext_fee": config.ext_fee,
            "donation_apy": donation_apy,
            "fee_grid": {
                "mode": "single" if len(runs) == 1 else config.fee_grid,
                "fee": config.fee,
                "fee_min": config.fee_min,
                "fee_max": config.fee_max,
                "fee_count": len(runs),
            },
        },
        "best": best,
        "runs": compact_runs,
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _downsample_path(path: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    if max_points <= 0 or len(path) <= max_points:
        return path
    idx = np.linspace(0, len(path) - 1, max_points, dtype=int)
    return [path[int(i)] for i in idx]


def plot_payload(payload: dict[str, Any], out: Path | None, max_points: int) -> None:
    import matplotlib

    if out is not None:
        matplotlib.use("Agg")
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt

    runs = payload["runs"]
    path = _downsample_path(payload["best"].get("path", []), max_points)
    has_scan = len(runs) > 1
    nrows = 2 if has_scan and path else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4.5 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    if has_scan:
        fees = np.array([row["fee"] for row in runs], dtype=np.float64)
        apys = np.array([row["apy"] for row in runs], dtype=np.float64) * 100.0
        ax.plot(fees, apys, marker="o", linewidth=1.4)
        if payload["metadata"]["fee_grid"]["mode"] == "log":
            ax.set_xscale("log")
        ax.axvline(payload["best"]["fee"], color="red", linestyle="--", linewidth=1)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{100 * x:g}%"))
        ax.set_xlabel("YB AMM fee")
        ax.set_ylabel("Final annualized APY (%)")
        ax.grid(True, alpha=0.3)
        ax.set_title("YB releverage fee scan: fee vs final annualized APY")
    else:
        _plot_growth_axis(ax, path)

    if has_scan and path:
        _plot_growth_axis(axes[1], path)

    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    else:
        plt.show()


def _plot_growth_axis(ax: Any, path: list[dict[str, float]]) -> None:
    import matplotlib.dates as mdates

    if not path:
        ax.text(0.5, 0.5, "No path samples", ha="center", va="center")
        ax.set_axis_off()
        return

    dates = [datetime.fromtimestamp(row["t"], timezone.utc) for row in path]
    growth = np.array([row["growth"] for row in path], dtype=np.float64)
    amm_price = np.array([row["amm_price"] for row in path], dtype=np.float64)
    oracle = np.array([row["oracle"] for row in path], dtype=np.float64)

    ax.plot(dates, (growth - 1.0) * 100.0, label="Deposit growth", linewidth=1.2)
    ax.set_ylabel("Deposit growth (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(dates, amm_price, label="AMM price", alpha=0.55, linewidth=0.9)
    ax2.plot(dates, oracle, label="Oracle", alpha=0.55, linewidth=0.9)
    ax2.set_ylabel("Normalized price")
    ax2.legend(loc="upper right")

    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YieldBasis releverage simulation on detailed-output.json"
    )
    parser.add_argument(
        "--detailed",
        type=Path,
        default=HERE / "run_data" / "detailed-output.json",
        help="Path to detailed-output.json or detailed-output.json.xz",
    )
    parser.add_argument(
        "--donation-apy",
        type=float,
        default=None,
        help="Annual donation rate as a fraction. Overrides detailed-output donation_apy.",
    )
    parser.add_argument("--fee", type=float, default=None, help="Single AMM fee fraction")
    parser.add_argument(
        "--scan", action="store_true", help="Scan the fee grid even if --fee is set"
    )
    parser.add_argument("--fee-min", type=float, default=0.002)
    parser.add_argument("--fee-max", type=float, default=0.05)
    parser.add_argument("--fee-count", type=int, default=20)
    parser.add_argument("--fee-grid", choices=["log", "linear"], default="log")
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--ext-fee", type=float, default=0.0)
    parser.add_argument("--peg-to", default="price_scale")
    parser.add_argument(
        "--profit-field",
        choices=["profit", "vp", "vp_boosted", "xcp"],
        default="profit",
    )
    parser.add_argument(
        "--path-every",
        type=int,
        default=1,
        help="Sample every N rows for the output path. Use 0 to disable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to yb_releverage.json next to the trace.",
    )
    parser.add_argument("--plot", action="store_true", help="Show an interactive plot")
    parser.add_argument("--plot-out", type=Path, default=None, help="Write plot PNG")
    parser.add_argument("--max-plot-points", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace_path = args.detailed.expanduser().resolve()
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}", file=sys.stderr)
        return 2

    trace = load_trace(trace_path, args.peg_to, args.profit_field)
    donation_apy = args.donation_apy
    if donation_apy is None:
        donation_apy = trace.donation_apy
    if donation_apy is None:
        print(
            "Trace does not contain donation_apy. Regenerate detailed output with "
            "the current harness or pass --donation-apy explicitly.",
            file=sys.stderr,
        )
        return 2

    fees = make_fee_grid(
        args.fee,
        args.fee_min,
        args.fee_max,
        args.fee_count,
        args.fee_grid,
        args.scan,
    )

    runs: list[dict[str, Any]] = []
    for fee in fees:
        config = SimulationConfig(
            fee=fee,
            leverage=args.leverage,
            ext_fee=args.ext_fee,
            donation_apy=donation_apy,
            path_every=args.path_every,
        )
        runs.append(run_releverage(trace, config))

    out_path = args.out or (trace_path.parent / "yb_releverage.json")
    payload = build_payload(trace_path, trace, runs, args, donation_apy)
    write_payload(out_path, payload)

    best = payload["best"]
    print(
        f"Loaded {len(trace.t)} rows over {trace.duration / 86400.0:.2f} days "
        f"from {trace_path}"
    )
    source = "CLI override" if args.donation_apy is not None else "detailed output"
    print(f"Donation APY: {donation_apy:.6g} ({source})")
    print(
        f"Best fee={best['fee']:.8g}, apy={best['apy'] * 100.0:.6g}%, "
        f"final_growth={best['final_growth']:.8g}, trades={best['n_releverage_trades']}"
    )
    print(f"Saved JSON to {out_path}")

    if args.plot or args.plot_out is not None:
        plot_payload(payload, args.plot_out, args.max_plot_points)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
