#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import math
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

try:
    from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool
except ModuleNotFoundError:
    from python.arb_sim.pool_helpers import (
        _first_candle_ts,
        _initial_price_from_file,
        strify_pool,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY_PATH = REPO_ROOT / "cpp_modular" / "build" / "arb_eval_server"
CANDLES_PATH = (
    REPO_ROOT / "python" / "arb_sim" / "trade_data" / "btcusd" / "btcusd-2023-2026.json"
)
RESULT_PATH = REPO_ROOT / "comparison-results" / "nevergrad_fee_result.json"

FORCE_REBUILD_BINARY = False
BUILD_BINARY_IF_MISSING = True
FAIL_PENALTY = 1_000_000.0
OUT_FEE_BPS_MAX = 600.0


def scalar(
    lower: float,
    upper: float,
    *,
    request_key: str | None = None,
    step: float | None = None,
    scale: float = 1.0,
) -> dict[str, Any]:
    spec: dict[str, Any] = {"kind": "scalar", "lower": lower, "upper": upper}
    if request_key is not None:
        spec["request_key"] = request_key
    if step is not None:
        spec["step"] = step
    if scale != 1.0:
        spec["scale"] = scale
    return spec


def log_scale(
    lower: float, upper: float, *, request_key: str | None = None
) -> dict[str, Any]:
    spec: dict[str, Any] = {"kind": "log", "lower": lower, "upper": upper}
    if request_key is not None:
        spec["request_key"] = request_key
    return spec


SERVER_CONFIG = {
    "pool_index": 0,
    "n_candles": 0,
    "candle_filter": 99.0,
    "dustswapfreq": 600,
    "min_swap": 1e-6,
    "max_swap": 1.0,
    "disable_slippage_probes": True,
}

TEMPLATE_POOL = {
    "A": 6 * 10_000,
    "gamma": 0.0001,
    "lp_profit_fraction": 0.5,
    "allowed_extra_profit": 1e-10,
    "adjustment_step": 0.005,
    "ma_time": 866.0,
    "donation_apy": 0.036,
    "donation_frequency": 86400.0,
    "donation_coins_ratio": 0.5,
    "initial_liq_coin0": 10_000_000.0,
    "mid_fee_bps": 1.0,
    "out_fee_bps": 1.0,
    "fee_gamma": 0.003,
}

TEMPLATE_COSTS = {
    "arb_fee_bps": 3.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1.0,
}

# Add more request fields here as needed, for example:
# "A": scalar(2 * 10_000, 20 * 10_000)
# "donation_rate": scalar(0.0, 0.10, request_key="donation_apy")
OPTIMIZABLE_VARS = {
    "mid_fee_bps": scalar(1.0, 100.0, step=1),
    "spread_bps": scalar(0.0, 200.0, step=1),
    "fee_gamma": log_scale(1e-6, 0.05),
    "A_units": scalar(2.0, 40.0, request_key="A", step=0.1, scale=10_000.0),
    "donation_apy": scalar(0.0, 0.05, step=0.001),
    # "lp_profit_fraction": scalar(0.1, 1.0, step=0.025),
}

LOSS_CONFIG = {
    "avg_rel_target": 500.0 / 10_000.0,
    "avg_rel_lambda": 0.1,
}

ROBUST_EVAL = {
    "enabled": True,
    "metric": "apy_net",
    "mode": "maximize",
    "tolerance_frac": 0.20,
    "penalty_weight": 0.1,
    "metric_floor": 0.01,
    "A_shift_frac": 0.05,
    "fee_shift_frac": 0.0,
    "aggregate": "mean",
}

OPTIMIZATION = {
    "optimizer": "TwoPointsDE",
    "budget": 5000,
    "seed": 0,
    "verbose_every": 10,
    "workers": 16,
}


def scale_1e18(value: float) -> int:
    return int(round(value * 1e18))


def fee_bps_to_1e10(value_bps: float) -> int:
    return int(round((value_bps / 10_000.0) * 1e10))


def build_template_config(candles_path: Path) -> dict[str, Any]:
    start_ts = _first_candle_ts(str(candles_path))
    initial_price = _initial_price_from_file(str(candles_path))

    total_liq_coin0 = float(TEMPLATE_POOL["initial_liq_coin0"])
    initial_liquidity = [
        int((total_liq_coin0 * 1e18) // 2),
        int((total_liq_coin0 * 1e18) // 2 / initial_price),
    ]

    pool = {
        "initial_liquidity": initial_liquidity,
        "A": float(TEMPLATE_POOL["A"]),
        "gamma": float(TEMPLATE_POOL["gamma"]),
        "lp_profit_fraction": float(TEMPLATE_POOL["lp_profit_fraction"]),
        "mid_fee": fee_bps_to_1e10(float(TEMPLATE_POOL["mid_fee_bps"])),
        "out_fee": fee_bps_to_1e10(float(TEMPLATE_POOL["out_fee_bps"])),
        "fee_gamma": scale_1e18(float(TEMPLATE_POOL["fee_gamma"])),
        "allowed_extra_profit": scale_1e18(
            float(TEMPLATE_POOL["allowed_extra_profit"])
        ),
        "adjustment_step": scale_1e18(float(TEMPLATE_POOL["adjustment_step"])),
        "ma_time": float(TEMPLATE_POOL["ma_time"]),
        "initial_price": scale_1e18(initial_price),
        "start_timestamp": int(start_ts),
        "donation_apy": float(TEMPLATE_POOL["donation_apy"]),
        "donation_frequency": float(TEMPLATE_POOL["donation_frequency"]),
        "donation_coins_ratio": float(TEMPLATE_POOL["donation_coins_ratio"]),
    }

    return {
        "meta": {
            "created_by": "nevergrad_fee_runner.py",
            "datafile": str(candles_path),
        },
        "pools": [
            {
                "tag": "embedded_template",
                "pool": strify_pool(pool),
                "costs": dict(TEMPLATE_COSTS),
            }
        ],
    }


def build_pool_config_entry(
    request_params: dict[str, Any], candles_path: Path
) -> dict[str, Any]:
    template = build_template_config(candles_path)
    entry = dict(template["pools"][0])
    pool = dict(entry["pool"])

    if "A" in request_params:
        pool["A"] = str(float(request_params["A"]))
    if "gamma" in request_params:
        pool["gamma"] = str(float(request_params["gamma"]))
    if "lp_profit_fraction" in request_params:
        pool["lp_profit_fraction"] = str(float(request_params["lp_profit_fraction"]))
    if "mid_fee_bps" in request_params:
        pool["mid_fee"] = str(fee_bps_to_1e10(float(request_params["mid_fee_bps"])))
    if "out_fee_bps" in request_params:
        pool["out_fee"] = str(fee_bps_to_1e10(float(request_params["out_fee_bps"])))
    if "fee_gamma" in request_params:
        pool["fee_gamma"] = str(scale_1e18(float(request_params["fee_gamma"])))
    if "allowed_extra_profit" in request_params:
        pool["allowed_extra_profit"] = str(
            scale_1e18(float(request_params["allowed_extra_profit"]))
        )
    if "adjustment_step" in request_params:
        pool["adjustment_step"] = str(
            scale_1e18(float(request_params["adjustment_step"]))
        )
    if "ma_time" in request_params:
        pool["ma_time"] = str(float(request_params["ma_time"]))
    if "donation_apy" in request_params:
        pool["donation_apy"] = str(float(request_params["donation_apy"]))
    if "donation_frequency" in request_params:
        pool["donation_frequency"] = str(float(request_params["donation_frequency"]))
    if "donation_coins_ratio" in request_params:
        pool["donation_coins_ratio"] = str(
            float(request_params["donation_coins_ratio"])
        )

    costs = dict(entry["costs"])
    if "arb_fee_bps" in request_params:
        costs["arb_fee_bps"] = float(request_params["arb_fee_bps"])
    if "gas_coin0" in request_params:
        costs["gas_coin0"] = float(request_params["gas_coin0"])

    entry["pool"] = pool
    entry["costs"] = costs
    return entry


def build_saved_pool_payload(
    which: str,
    row: dict[str, Any] | None,
    candles_path: Path,
) -> dict[str, Any] | None:
    if row is None:
        return None

    request_params = dict(row.get("request", {}))
    request_params.pop("id", None)
    return {
        "which": which,
        "candles_path": str(candles_path),
        "server_config": dict(SERVER_CONFIG),
        "template_pool": dict(TEMPLATE_POOL),
        "template_costs": dict(TEMPLATE_COSTS),
        "params_display": dict(row.get("params", {})),
        "request_params": request_params,
        "pool_config_entry": build_pool_config_entry(request_params, candles_path),
    }


def write_template_file(candles_path: Path, directory: Path) -> Path:
    template_path = directory / "embedded_pool_config.json"
    template = build_template_config(candles_path)
    template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template_path


def build_eval_server(repo_root: Path) -> None:
    build_dir = repo_root / "cpp_modular" / "build"
    subprocess.run(
        ["cmake", "-S", str(repo_root / "cpp_modular"), "-B", str(build_dir)],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "arb_eval_server", "-j8"],
        check=True,
    )


def ensure_binary() -> Path:
    binary_path = BINARY_PATH.resolve()
    if FORCE_REBUILD_BINARY or (BUILD_BINARY_IF_MISSING and not binary_path.exists()):
        build_eval_server(REPO_ROOT)
    if not binary_path.exists():
        raise SystemExit(f"evaluator binary not found: {binary_path}")
    return binary_path


def require_nevergrad() -> Any:
    try:
        import nevergrad as ng
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "nevergrad is not installed. Install it with: uv pip install nevergrad"
        ) from exc
    return ng


class EvalServerClient:
    def __init__(
        self,
        binary: Path,
        template_path: Path,
        candles_path: Path,
        server_config: dict[str, Any],
    ) -> None:
        self.binary = binary
        self.template_path = template_path
        self.candles_path = candles_path
        self.server_config = server_config
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "EvalServerClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def build_command(self) -> list[str]:
        command = [
            str(self.binary),
            str(self.template_path),
            str(self.candles_path),
            "--pool-index",
            str(int(self.server_config["pool_index"])),
            "--n-candles",
            str(int(self.server_config["n_candles"])),
            "--candle-filter",
            str(float(self.server_config["candle_filter"])),
            "--dustswapfreq",
            str(int(self.server_config["dustswapfreq"])),
            "--min-swap",
            str(float(self.server_config["min_swap"])),
            "--max-swap",
            str(float(self.server_config["max_swap"])),
        ]
        if self.server_config.get("disable_slippage_probes", False):
            command.append("--disable-slippage-probes")
        return command

    def start(self) -> None:
        if self.proc is not None:
            return
        self.proc = subprocess.Popen(
            self.build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.write("quit\n")
                    self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        self.proc = None

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None:
            self.start()
        assert self.proc is not None
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Evaluator process missing stdio pipes")

        self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("Evaluator process terminated before sending a response")

        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("Evaluator returned non-object JSON")
        return response


def as_float(data: dict[str, Any], key: str, fallback: float) -> float:
    value = data.get(key, fallback)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def build_parametrization(ng: Any) -> Any:
    instruments: dict[str, Any] = {}
    for name, spec in OPTIMIZABLE_VARS.items():
        kind = str(spec["kind"])
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        if kind == "scalar":
            instruments[name] = ng.p.Scalar(lower=lower, upper=upper)
        elif kind == "log":
            instruments[name] = ng.p.Log(lower=lower, upper=upper)
        else:
            raise ValueError(f"unsupported variable kind for {name}: {kind}")
    return ng.p.Instrumentation(**instruments)


def compute_loss(response: dict[str, Any]) -> float:
    apy_net = as_float(response, "apy_net", -1.0)
    avg_rel = abs(as_float(response, "avg_rel_price_diff", 1.0))
    target = float(LOSS_CONFIG["avg_rel_target"])
    lam = float(LOSS_CONFIG["avg_rel_lambda"])

    excess = max(0.0, avg_rel / target - 1.0)
    return -apy_net + lam * (excess**2)


def clamp_request_value(request_key: str, value: float) -> float:
    for name, spec in OPTIMIZABLE_VARS.items():
        if str(spec.get("request_key", name)) != request_key:
            continue
        scale = float(spec.get("scale", 1.0))
        lower = float(spec["lower"]) * scale
        upper = float(spec["upper"]) * scale
        return min(max(value, lower), upper)

    if request_key == "out_fee_bps":
        return min(max(value, 0.0), OUT_FEE_BPS_MAX)
    return value


def quantize_value(value: float, step: float | None) -> float:
    if step is None or step <= 0.0:
        return value
    return round(value / step) * step


def candidate_value_to_request_value(
    spec: dict[str, Any], candidate_value: float
) -> float:
    quantized = quantize_value(candidate_value, spec.get("step"))
    scaled = quantized * float(spec.get("scale", 1.0))
    return scaled


def request_value_to_display_value(spec: dict[str, Any], request_value: float) -> float:
    scale = float(spec.get("scale", 1.0))
    value = request_value / scale
    return quantize_value(value, spec.get("step"))


def shifted_request(
    request: dict[str, Any], key: str, shift_frac: float
) -> dict[str, Any]:
    shifted = dict(request)
    shifted[key] = clamp_request_value(key, float(request[key]) * (1.0 + shift_frac))
    for name, spec in OPTIMIZABLE_VARS.items():
        if str(spec.get("request_key", name)) != key:
            continue
        step = spec.get("step")
        scale = float(spec.get("scale", 1.0))
        if step is not None and step > 0.0:
            shifted[key] = clamp_request_value(
                key,
                quantize_value(float(shifted[key]) / scale, step) * scale,
            )
        break
    return shifted


def dedupe_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for request in requests:
        marker = json.dumps(request, sort_keys=True, separators=(",", ":"))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(request)
    return unique


def build_robust_neighbor_requests(request: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(ROBUST_EVAL.get("enabled", False)):
        return []

    neighbors: list[dict[str, Any]] = []

    a_shift_frac = float(ROBUST_EVAL.get("A_shift_frac", 0.0))
    if a_shift_frac > 0.0 and "A" in request:
        neighbors.append(shifted_request(request, "A", +a_shift_frac))
        neighbors.append(shifted_request(request, "A", -a_shift_frac))

    fee_shift_frac = float(ROBUST_EVAL.get("fee_shift_frac", 0.0))
    if fee_shift_frac > 0.0 and "mid_fee_bps" in request and "out_fee_bps" in request:
        for sign in (+1.0, -1.0):
            shifted = dict(request)
            shifted["mid_fee_bps"] = clamp_request_value(
                "mid_fee_bps",
                float(request["mid_fee_bps"]) * (1.0 + sign * fee_shift_frac),
            )
            shifted["out_fee_bps"] = clamp_request_value(
                "out_fee_bps",
                float(request["out_fee_bps"]) * (1.0 + sign * fee_shift_frac),
            )
            if shifted["out_fee_bps"] >= shifted["mid_fee_bps"]:
                neighbors.append(shifted)

    center_key = dict(request)
    center_key.pop("id", None)
    unique = []
    for neighbor in dedupe_requests(neighbors):
        compare = dict(neighbor)
        compare.pop("id", None)
        if compare != center_key:
            unique.append(neighbor)
    return unique


def extract_metric(response: dict[str, Any], metric: str, fallback: float) -> float:
    return as_float(response, metric, fallback)


def compute_robust_penalty(
    center_response: dict[str, Any], neighbor_responses: list[dict[str, Any]]
) -> float:
    if not bool(ROBUST_EVAL.get("enabled", False)) or not neighbor_responses:
        return 0.0

    metric = str(ROBUST_EVAL["metric"])
    mode = str(ROBUST_EVAL["mode"])
    tolerance_frac = float(ROBUST_EVAL["tolerance_frac"])
    penalty_weight = float(ROBUST_EVAL["penalty_weight"])
    metric_floor = float(ROBUST_EVAL["metric_floor"])
    aggregate = str(ROBUST_EVAL.get("aggregate", "mean"))

    center_metric = extract_metric(center_response, metric, float("nan"))
    if not math.isfinite(center_metric):
        return FAIL_PENALTY

    ref = max(abs(center_metric), metric_floor)
    excesses: list[float] = []
    for response in neighbor_responses:
        if not bool(response.get("ok", False)):
            excesses.append(1.0)
            continue
        neighbor_metric = extract_metric(response, metric, float("nan"))
        if not math.isfinite(neighbor_metric):
            excesses.append(1.0)
            continue

        if mode == "maximize":
            worsening = max(0.0, (center_metric - neighbor_metric) / ref)
        elif mode == "minimize":
            worsening = max(0.0, (neighbor_metric - center_metric) / ref)
        else:
            raise ValueError(f"unsupported robust mode: {mode}")

        excesses.append(max(0.0, worsening - tolerance_frac) ** 2)

    if not excesses:
        return 0.0
    if aggregate == "max":
        fragility = max(excesses)
    else:
        fragility = sum(excesses) / len(excesses)
    return penalty_weight * fragility


def build_request(
    eval_id: int, candidate: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float], str | None]:
    request: dict[str, Any] = {"id": eval_id}
    params: dict[str, float] = {}

    for name, spec in OPTIMIZABLE_VARS.items():
        value = float(candidate[name])
        request_key = str(spec.get("request_key", name))
        request_value = clamp_request_value(
            request_key, candidate_value_to_request_value(spec, value)
        )
        request[request_key] = request_value
        params[name] = request_value_to_display_value(spec, request_value)
        if request_key != name and float(spec.get("scale", 1.0)) == 1.0:
            params[request_key] = request_value

    if "spread_bps" in params:
        if "mid_fee_bps" not in params:
            return request, params, "spread_bps requires mid_fee_bps"
        out_fee_bps = params["mid_fee_bps"] + params["spread_bps"]
        request.pop("spread_bps", None)
        request["out_fee_bps"] = out_fee_bps
        params["out_fee_bps"] = out_fee_bps

    out_fee_bps = request.get("out_fee_bps")
    if out_fee_bps is not None and float(out_fee_bps) > OUT_FEE_BPS_MAX:
        return (
            request,
            params,
            f"out_fee_bps {float(out_fee_bps):.8f} exceeds max {OUT_FEE_BPS_MAX:.8f}",
        )

    return request, params, None


def evaluate_candidate(
    client: EvalServerClient, eval_id: int, candidate: dict[str, Any]
) -> dict[str, Any]:
    request, params, error = build_request(eval_id, candidate)
    if error is not None:
        return {
            "step": eval_id,
            "loss": FAIL_PENALTY,
            "params": params,
            "request": request,
            "response": {"ok": False, "error": error},
        }

    response = client.evaluate(request)
    robust_neighbors = build_robust_neighbor_requests(request)
    neighbor_responses: list[dict[str, Any]] = []
    for idx, neighbor_request in enumerate(robust_neighbors, start=1):
        with_id = dict(neighbor_request)
        with_id["id"] = f"{eval_id}:n{idx}"
        neighbor_responses.append(client.evaluate(with_id))

    if bool(response.get("ok", False)):
        center_loss = compute_loss(response)
        robust_penalty = compute_robust_penalty(response, neighbor_responses)
        loss = center_loss + robust_penalty
    else:
        center_loss = FAIL_PENALTY
        robust_penalty = 0.0
        loss = FAIL_PENALTY

    return {
        "step": eval_id,
        "loss": loss,
        "center_loss": center_loss,
        "robust_penalty": robust_penalty,
        "params": params,
        "request": request,
        "response": response,
        "neighbor_responses": neighbor_responses,
    }


PARAM_LABELS = {
    "mid_fee_bps": "mid",
    "out_fee_bps": "out",
    "spread_bps": "spread",
    "fee_gamma": "gamma",
    "A_units": "A",
    "lp_profit_fraction": "lpf",
}


def format_value(key: str, value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if key.endswith("_bps"):
        return f"{value:.1f}"
    if key == "fee_gamma":
        return f"{value:.3e}"
    if key == "A_units":
        return f"{value:.1f}"
    if key == "lp_profit_fraction":
        return f"{value:.2f}"
    if key.endswith("_apy") or key.endswith("_rate"):
        return f"{value:.4f}"
    if abs(value) >= 1000 and float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.4f}"


def format_loss(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def format_apy(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def format_rel_bps(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value * 10_000.0:.1f}bps"


def ordered_param_keys(params: dict[str, Any]) -> list[str]:
    ordered = list(OPTIMIZABLE_VARS)
    if "out_fee_bps" in params and "out_fee_bps" not in ordered:
        if "mid_fee_bps" in ordered:
            ordered.insert(ordered.index("mid_fee_bps") + 1, "out_fee_bps")
        else:
            ordered.append("out_fee_bps")

    for key in sorted(params):
        if key not in ordered:
            ordered.append(key)
    return ordered


def format_param_block(params: dict[str, Any]) -> str:
    parts = []
    for key in ordered_param_keys(params):
        if key not in params:
            continue
        label = PARAM_LABELS.get(key, key)
        parts.append(f"{label}={format_value(key, float(params[key]))}")
    return "(" + ", ".join(parts) + ")"


def print_progress(
    step: int, budget: int, row: dict[str, Any], best_seen: dict[str, Any] | None
) -> None:
    response = row["response"]
    params = row["params"]
    apy_net = as_float(response, "apy_net", float("nan"))
    avg_rel = as_float(response, "avg_rel_price_diff", float("nan"))
    max_rel = as_float(response, "max_rel_price_diff", float("nan"))
    best_loss = as_float(best_seen or {}, "loss", float("nan"))
    best_apy_net = (
        as_float(best_seen["response"], "apy_net", float("nan"))
        if best_seen is not None
        else float("nan")
    )
    best_avg_rel = (
        as_float(best_seen["response"], "avg_rel_price_diff", float("nan"))
        if best_seen is not None
        else float("nan")
    )
    best_params = (
        format_param_block(best_seen["params"]) if best_seen is not None else "-"
    )
    current_params = format_param_block(params)
    center_loss = as_float(row, "center_loss", float("nan"))
    robust_penalty = as_float(row, "robust_penalty", 0.0)

    left = (
        f"[{step}/{budget}] cur_l={format_loss(float(row['loss']))} "
        f"center={format_loss(center_loss)} rpen={format_loss(robust_penalty)} "
        f"apy_net={format_apy(apy_net)} avg_pdif={format_rel_bps(avg_rel)} "
        f"max={format_rel_bps(max_rel)} x={current_params}"
    )
    right = (
        f"best_l={format_loss(best_loss)} apy_net={format_apy(best_apy_net)} "
        f"avg_pdif={format_rel_bps(best_avg_rel)} "
        f"x={best_params}"
    )

    if bool(response.get("ok", False)):
        print(f"{left} | {right}", flush=True)
        return

    error = str(response.get("error", "unknown error"))
    print(
        f"[{step}/{budget}] curr error={error} x={current_params} | {right}", flush=True
    )


def save_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_optimizer(ng: Any, parametrization: Any) -> Any:
    workers = max(1, int(OPTIMIZATION["workers"]))
    optimizer_name = str(OPTIMIZATION["optimizer"])
    optimizer_cls = getattr(ng.optimizers, optimizer_name)
    optimizer = optimizer_cls(
        parametrization=parametrization,
        budget=int(OPTIMIZATION["budget"]),
        num_workers=workers,
    )

    seed = OPTIMIZATION.get("seed")
    if seed is not None:
        optimizer.parametrization.random_state.seed(int(seed))
    return optimizer


def open_eval_clients(
    stack: ExitStack,
    binary_path: Path,
    template_path: Path,
    candles_path: Path,
    workers: int,
) -> list[EvalServerClient]:
    clients: list[EvalServerClient] = []
    for _ in range(workers):
        clients.append(
            stack.enter_context(
                EvalServerClient(
                    binary=binary_path,
                    template_path=template_path,
                    candles_path=candles_path,
                    server_config=dict(SERVER_CONFIG),
                )
            )
        )
    return clients


def run_parallel_batch(
    executor: ThreadPoolExecutor,
    optimizer: Any,
    clients: list[EvalServerClient],
    next_step: int,
    batch_size: int,
) -> list[tuple[Any, dict[str, Any]]]:
    jobs: list[tuple[Any, Any]] = []
    for offset in range(batch_size):
        candidate = optimizer.ask()
        eval_id = next_step + offset
        future = executor.submit(
            evaluate_candidate,
            clients[offset],
            eval_id,
            candidate.kwargs,
        )
        jobs.append((candidate, future))

    rows: list[tuple[Any, dict[str, Any]]] = []
    for candidate, future in jobs:
        rows.append((candidate, future.result()))
    return rows


def run_optimization(ng: Any, binary_path: Path, candles_path: Path) -> dict[str, Any]:
    parametrization = build_parametrization(ng)
    optimizer = build_optimizer(ng, parametrization)
    optimizer_name = str(OPTIMIZATION["optimizer"])
    budget = int(OPTIMIZATION["budget"])
    seed = OPTIMIZATION.get("seed")
    verbose_every = max(1, int(OPTIMIZATION["verbose_every"]))
    workers = max(1, min(int(OPTIMIZATION["workers"]), budget))

    started = time.time()
    best_seen: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="arb_eval_template_") as tmp_dir:
        template_path = write_template_file(candles_path, Path(tmp_dir))

        with ExitStack() as stack:
            clients = open_eval_clients(
                stack,
                binary_path=binary_path,
                template_path=template_path,
                candles_path=candles_path,
                workers=workers,
            )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                next_step = 1
                while next_step <= budget:
                    batch_size = min(workers, budget - next_step + 1)
                    batch_rows = run_parallel_batch(
                        executor,
                        optimizer,
                        clients,
                        next_step,
                        batch_size,
                    )

                    for candidate, row in batch_rows:
                        optimizer.tell(candidate, float(row["loss"]))

                        if best_seen is None or float(row["loss"]) < float(
                            best_seen["loss"]
                        ):
                            best_seen = row

                        step = int(row["step"])
                        if step == 1 or step == budget or step % verbose_every == 0:
                            print_progress(step, budget, row, best_seen)

                    next_step += batch_size

                recommendation_candidate = optimizer.provide_recommendation()
                recommendation = evaluate_candidate(
                    clients[0], budget + 1, recommendation_candidate.kwargs
                )

    elapsed_s = time.time() - started
    return {
        "optimizer": optimizer_name,
        "budget": budget,
        "seed": seed,
        "workers": workers,
        "elapsed_s": elapsed_s,
        "candles_path": str(candles_path),
        "server_config": SERVER_CONFIG,
        "template_pool": TEMPLATE_POOL,
        "template_costs": TEMPLATE_COSTS,
        "optimizable_vars": OPTIMIZABLE_VARS,
        "loss_config": LOSS_CONFIG,
        "robust_eval": ROBUST_EVAL,
        "best_seen": best_seen,
        "recommendation": recommendation,
        "best_pool": build_saved_pool_payload("best_seen", best_seen, candles_path),
        "recommendation_pool": build_saved_pool_payload(
            "recommendation", recommendation, candles_path
        ),
    }


def main() -> int:
    ng = require_nevergrad()
    binary_path = ensure_binary()
    candles_path = CANDLES_PATH.resolve()
    if not candles_path.exists():
        raise SystemExit(f"candles file not found: {candles_path}")

    payload = run_optimization(ng, binary_path, candles_path)
    save_result(payload)

    best_seen = payload["best_seen"]
    if best_seen is None:
        raise SystemExit("no evaluations were executed")

    print("best_seen:", json.dumps(best_seen, separators=(",", ":"), sort_keys=True))
    print(
        "recommendation:",
        json.dumps(payload["recommendation"], separators=(",", ":"), sort_keys=True),
    )
    print(f"saved: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
