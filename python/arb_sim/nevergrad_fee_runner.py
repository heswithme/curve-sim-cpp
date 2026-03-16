#!/usr/bin/env python3
"""Canonical autonomous fee-search runner.

This is the only entrypoint the autonomous code-research loop should call.
It optimizes a fixed in-code `fee_param_[0..19]` search surface against one
exact pool entry and one candle file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    from pool_helpers import _first_candle_ts, _initial_price_from_file
except ModuleNotFoundError:
    try:
        from .pool_helpers import _first_candle_ts, _initial_price_from_file
    except ImportError:
        from python.arb_sim.pool_helpers import (
            _first_candle_ts,
            _initial_price_from_file,
        )

try:
    from pool_helpers import strify_pool
except ModuleNotFoundError:
    try:
        from .pool_helpers import strify_pool
    except ImportError:
        from python.arb_sim.pool_helpers import strify_pool


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY_PATH = REPO_ROOT / "cpp_modular" / "build" / "arb_eval_server"
DEFAULT_CANDLES_PATH = (
    REPO_ROOT / "python" / "arb_sim" / "trade_data" / "btcusd" / "btcusd-2023-2026.json"
)
DEFAULT_RESULT_PATH = REPO_ROOT / "comparison-results" / "nevergrad_fee_result.json"
FAIL_PENALTY = 1_000_000.0
FEE_PARAM_COUNT = 20

FEE_PARAM_FEE_VALUE = 0
ACTIVE_SEARCH_SPEC_MODEL = "constant_fee_v1"

FEE_PARAM_LABELS = [
    "fee_value",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "reserved_5",
    "reserved_6",
    "reserved_7",
    "reserved_8",
    "reserved_9",
    "reserved_10",
    "reserved_11",
    "reserved_12",
    "reserved_13",
    "reserved_14",
    "reserved_15",
    "reserved_16",
    "reserved_17",
    "reserved_18",
    "reserved_19",
]

# ============================================================================
# HUMAN-OWNED EVALUATION / OBJECTIVE DEFAULTS
#
# AGENT RULE:
# - Do not auto-modify these defaults in autonomous fee-model experiments.
# - These settings define the fixed evaluation environment around the fee model.
# ============================================================================
DEFAULT_SERVER_CONFIG = {
    "pool_index": 0,
    "n_candles": 0,
    "candle_filter": 99.0,
    "dustswapfreq": 600,
    "min_swap": 1e-6,
    "max_swap": 1.0,
    "disable_slippage_probes": True,
}

DEFAULT_LOSS_CONFIG = {
    "metric": "apy_net",
    "avg_rel_target": 500.0 / 10_000.0,
    "avg_rel_lambda": 0.1,
}

DEFAULT_CONSTRAINTS = {
    "enabled": False,
    "avg_rel_price_diff_max": 500.0 / 10_000.0,
    "max_rel_price_diff_max": 3000.0 / 10_000.0,
    "tw_real_slippage_5pct_max": 500.0 / 10_000.0,
    "min_trades": 1000,
}

DEFAULT_ROBUST_EVAL = {
    "enabled": False,
    "metric": "apy_net",
    "mode": "maximize",
    "tolerance_frac": 0.20,
    "penalty_weight": 0.1,
    "metric_floor": 0.01,
    "A_shift_frac": 0.1,
    "aggregate": "mean",
}

DEFAULT_OPTIMIZATION = {
    "optimizer": "TwoPointsDE",
    "budget": 2500,
    "seed": 0,
    "verbose_every": 10,
    "workers": 16,
}

EVALUATOR_HASH_INPUTS = [
    REPO_ROOT / "cpp_modular" / "include" / "pools" / "twocrypto_fx" / "fee_model.hpp",
    REPO_ROOT / "cpp_modular" / "include" / "pools" / "twocrypto_fx" / "twocrypto.hpp",
    REPO_ROOT / "cpp_modular" / "include" / "trading" / "arbitrageur.hpp",
    REPO_ROOT / "cpp_modular" / "include" / "harness" / "event_loop.hpp",
    REPO_ROOT / "cpp_modular" / "include" / "harness" / "runner.hpp",
    REPO_ROOT / "cpp_modular" / "include" / "harness" / "output.hpp",
    REPO_ROOT / "cpp_modular" / "src" / "eval_server.cpp",
]


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


def zero_fee_params() -> list[float]:
    return [0.0] * FEE_PARAM_COUNT


def make_constant_fee_params(*, fee_value: float = 1.0 / 10_000.0) -> list[float]:
    params = zero_fee_params()
    params[FEE_PARAM_FEE_VALUE] = float(fee_value)
    return params


def wad(value: float) -> int:
    return int(round(value * 10**18))


def fee_bps(value: float) -> int:
    return wad(value / 10_000.0)


# ============================================================================
# HUMAN-OWNED BENCHMARK / OBJECTIVE DEFAULTS
#
# AGENT RULE:
# - Do not auto-modify these defaults in autonomous fee-model experiments.
# - This block fixes the benchmark pool, server behavior, and objective surface
#   so fee-law comparisons stay comparable.
# ============================================================================

# ============================================================================
# HUMAN-OWNED BENCHMARK POOL TEMPLATE
#
# AGENT RULE:
# - Do not auto-modify this block in autonomous fee-model experiments.
# - The outer loop may edit `fee_model.hpp` and optimize `fee_params`, but this
#   benchmark pool definition is intentionally fixed unless a human changes it.
#
# HUMAN USE:
# - Edit this block if you want a different fixed benchmark pool for
#   `nevergrad_fee_runner.py` when `--template-pools` is not provided.
# - Keep this in high-level pool terms. The runner will derive the canonical
#   single-pool JSON payload and inject the optimized `fee_params`.
# ============================================================================
EMBEDDED_TEMPLATE_TAG = "agent_constant_fee_template"
EMBEDDED_TEMPLATE_INITIAL_LIQ_COIN0 = 10_000_000.0
EMBEDDED_TEMPLATE_POOL = {
    "A": int(6 * 10_000),
    "gamma": 1e-4,
    "fee_params": [fee_bps(1)] + [0] * (FEE_PARAM_COUNT - 1),
    "allowed_extra_profit": wad(1e-10),
    "adjustment_step": wad(0.005),
    "ma_time": 866,
    "donation_apy": 0.036,
    "donation_frequency": 86400,
    "donation_coins_ratio": 0.5,
    "lp_profit_fraction": 0.5,
}
EMBEDDED_TEMPLATE_COSTS = {
    "arb_fee_bps": 3.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1.0,
}


# ============================================================================
# AGENT-EDITABLE SEARCH SPEC
#
# AGENT RULE:
# - You may modify only this block in Python when a new fee model needs
#   different active slots, ranges, kinds, or step sizes.
# - Do not change the benchmark pool, costs, server defaults, or objective
#   defaults above.
# - Keep the active slot set aligned with the fee model in `fee_model.hpp`.
# ============================================================================
def build_constant_fee_optimizable_vars() -> dict[str, dict[str, Any]]:
    return {
        "fee_param_0": scalar(0.0, 0.05, step=1.0 / 10_000.0),
    }


def build_default_optimizable_vars() -> dict[str, dict[str, Any]]:
    if ACTIVE_SEARCH_SPEC_MODEL == "constant_fee_v1":
        return build_constant_fee_optimizable_vars()
    raise ValueError(
        f"unsupported ACTIVE_SEARCH_SPEC_MODEL: {ACTIVE_SEARCH_SPEC_MODEL}"
    )


DEFAULT_OPTIMIZABLE_VARS = build_default_optimizable_vars()


def validate_optimizable_vars(optimizable_vars: dict[str, dict[str, Any]]) -> None:
    if not optimizable_vars:
        raise ValueError("optimizable_vars must contain at least one fee_param slot")

    seen_indices: list[int] = []
    for key in optimizable_vars:
        if not key.startswith("fee_param_"):
            raise ValueError(f"unsupported optimizable var key: {key}")
        try:
            idx = int(key.removeprefix("fee_param_"))
        except ValueError as exc:
            raise ValueError(f"invalid fee_param key: {key}") from exc
        if idx < 0 or idx >= FEE_PARAM_COUNT:
            raise ValueError(f"fee_param index out of range: {key}")
        seen_indices.append(idx)

    if seen_indices != sorted(seen_indices):
        raise ValueError("optimizable fee_param slots must be ordered by slot index")


def scale_1e18(value: float) -> int:
    return int(round(value * 1e18))


def parse_real(value: Any) -> float:
    return float(value)


def parse_scaled_1e18(value: Any) -> float:
    return float(value) / 1e18


def canonical_fee_params_from_pool(pool: dict[str, Any]) -> list[float]:
    raw = pool.get("fee_params")
    if not isinstance(raw, list) or len(raw) != FEE_PARAM_COUNT:
        raise ValueError("pool config must contain fee_params with length 20")
    return [parse_scaled_1e18(v) for v in raw]


def load_root_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def root_entries(root: Any) -> list[dict[str, Any]]:
    if isinstance(root, dict):
        if "pools" in root and isinstance(root["pools"], list):
            return [dict(entry) for entry in root["pools"]]
        if "pool" in root:
            return [dict(root)]
        raise ValueError("invalid pools json: expected 'pools' array or single 'pool'")
    if isinstance(root, list):
        return [dict(entry) for entry in root]
    raise ValueError("invalid pools json root type")


def load_pool_config_entry(template_path: Path, pool_index: int) -> dict[str, Any]:
    root = load_root_json(template_path)
    entries = root_entries(root)
    if pool_index < 0 or pool_index >= len(entries):
        raise IndexError(f"pool-index out of range ({pool_index} >= {len(entries)})")
    return copy.deepcopy(entries[pool_index])


def write_embedded_template_file(path: Path, candles_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = _first_candle_ts(str(candles_path))
    initial_price = _initial_price_from_file(str(candles_path))
    pool = dict(EMBEDDED_TEMPLATE_POOL)
    pool["initial_liquidity"] = [
        int((EMBEDDED_TEMPLATE_INITIAL_LIQ_COIN0 * 1e18) // 2),
        int((EMBEDDED_TEMPLATE_INITIAL_LIQ_COIN0 * 1e18) // 2 / initial_price),
    ]
    pool["initial_price"] = scale_1e18(initial_price)
    pool["start_timestamp"] = int(start_ts)
    payload = {
        "meta": {
            "created_by": "nevergrad_fee_runner.py",
            "source": "embedded_template_pool_entry",
            "datafile": str(candles_path),
        },
        "pools": [
            {
                "tag": EMBEDDED_TEMPLATE_TAG,
                "pool": strify_pool(pool),
                "costs": copy.deepcopy(EMBEDDED_TEMPLATE_COSTS),
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def fee_params_with_override(
    base_pool: dict[str, Any],
    fee_params_override: list[float] | None,
) -> list[float]:
    fee_params = canonical_fee_params_from_pool(base_pool)
    if fee_params_override is None:
        return fee_params
    if len(fee_params_override) != FEE_PARAM_COUNT:
        raise ValueError("fee_params_override must have length 20")
    return [float(value) for value in fee_params_override]


def canonical_pool_config_entry(
    entry: dict[str, Any],
    request_params: dict[str, Any],
    default_fee_params: list[float],
) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    if "pool" in result:
        pool = dict(result["pool"])
    else:
        pool = dict(result)

    fee_params = [
        float(value) for value in request_params.get("fee_params", default_fee_params)
    ]
    pool["fee_params"] = [str(scale_1e18(value)) for value in fee_params]

    float_pool_fields = (
        "A",
        "gamma",
        "lp_profit_fraction",
        "donation_apy",
        "donation_frequency",
        "donation_coins_ratio",
        "ma_time",
    )
    scaled_1e18_fields = ("allowed_extra_profit", "adjustment_step")
    for key in float_pool_fields:
        if key in request_params:
            pool[key] = str(float(request_params[key]))
    for key in scaled_1e18_fields:
        if key in request_params:
            pool[key] = str(scale_1e18(float(request_params[key])))

    if "pool" in result:
        result["pool"] = pool
    else:
        result = pool

    if "costs" in result:
        costs = dict(result["costs"])
    else:
        costs = {}
    for key in ("arb_fee_bps", "gas_coin0"):
        if key in request_params:
            costs[key] = float(request_params[key])
    if costs:
        result["costs"] = costs

    return result


def git_sha(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_eval_server(repo_root: Path) -> None:
    build_dir = repo_root / "cpp_modular" / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repo_root / "cpp_modular"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "arb_eval_server",
            "-j8",
        ],
        check=True,
    )


def ensure_binary(binary_path: Path, *, skip_rebuild: bool) -> Path:
    resolved = binary_path.resolve()
    if not skip_rebuild:
        build_eval_server(REPO_ROOT)
    if not resolved.exists():
        raise SystemExit(f"evaluator binary not found: {resolved}")
    return resolved


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


def build_parametrization(ng: Any, optimizable_vars: dict[str, Any]) -> Any:
    instruments: dict[str, Any] = {}
    for name, spec in optimizable_vars.items():
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


def constraint_violations(
    response: dict[str, Any], constraints: dict[str, Any]
) -> list[dict[str, float | str]]:
    violations: list[dict[str, float | str]] = []

    avg_rel = abs(as_float(response, "avg_rel_price_diff", float("nan")))
    avg_rel_max = float(constraints["avg_rel_price_diff_max"])
    if not math.isfinite(avg_rel) or avg_rel > avg_rel_max:
        violations.append(
            {
                "name": "avg_rel_price_diff",
                "value": avg_rel,
                "threshold": avg_rel_max,
            }
        )

    max_rel = abs(as_float(response, "max_rel_price_diff", float("nan")))
    max_rel_max = float(constraints["max_rel_price_diff_max"])
    if not math.isfinite(max_rel) or max_rel > max_rel_max:
        violations.append(
            {
                "name": "max_rel_price_diff",
                "value": max_rel,
                "threshold": max_rel_max,
            }
        )

    slippage = as_float(response, "tw_real_slippage_5pct", float("nan"))
    slippage_max = float(constraints["tw_real_slippage_5pct_max"])
    if not math.isfinite(slippage) or slippage < 0.0 or slippage > slippage_max:
        violations.append(
            {
                "name": "tw_real_slippage_5pct",
                "value": slippage,
                "threshold": slippage_max,
            }
        )

    trades = as_float(response, "trades", float("nan"))
    min_trades = float(constraints["min_trades"])
    if not math.isfinite(trades) or trades < min_trades:
        violations.append(
            {
                "name": "trades",
                "value": trades,
                "threshold": min_trades,
            }
        )

    return violations


def violation_penalty(violations: list[dict[str, float | str]]) -> float:
    if not violations:
        return 0.0
    penalty = FAIL_PENALTY
    for item in violations:
        value = float(item["value"])
        threshold = max(float(item["threshold"]), 1e-12)
        if not math.isfinite(value):
            penalty += 1000.0
        else:
            penalty += 100.0 * abs(value - threshold) / threshold
    return penalty


def compute_loss(
    response: dict[str, Any],
    loss_config: dict[str, Any],
    constraints: dict[str, Any],
) -> tuple[float, list[dict[str, float | str]], float]:
    metric = str(loss_config["metric"])
    metric_value = as_float(response, metric, float("-inf"))
    if not math.isfinite(metric_value):
        return (
            FAIL_PENALTY,
            [{"name": metric, "value": metric_value, "threshold": 0.0}],
            FAIL_PENALTY,
        )

    avg_rel = abs(as_float(response, "avg_rel_price_diff", float("inf")))
    target = float(loss_config["avg_rel_target"])
    lam = float(loss_config["avg_rel_lambda"])
    excess = max(0.0, avg_rel / target - 1.0)
    soft_loss = -metric_value + lam * (excess**2)

    if not bool(constraints.get("enabled", False)):
        return soft_loss, [], 0.0

    violations = constraint_violations(response, constraints)
    constraint_penalty = violation_penalty(violations)
    if constraint_penalty > 0.0:
        return constraint_penalty, violations, constraint_penalty
    return soft_loss, violations, 0.0


def clamp_request_value(
    request_key: str, value: float, optimizable_vars: dict[str, Any]
) -> float:
    for name, spec in optimizable_vars.items():
        if str(spec.get("request_key", name)) != request_key:
            continue
        scale = float(spec.get("scale", 1.0))
        lower = float(spec["lower"]) * scale
        upper = float(spec["upper"]) * scale
        return min(max(value, lower), upper)
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
    request: dict[str, Any],
    key: str,
    shift_frac: float,
    optimizable_vars: dict[str, Any],
) -> dict[str, Any]:
    shifted = dict(request)
    shifted[key] = clamp_request_value(
        key, float(request[key]) * (1.0 + shift_frac), optimizable_vars
    )
    for name, spec in optimizable_vars.items():
        if str(spec.get("request_key", name)) != key:
            continue
        step = spec.get("step")
        scale = float(spec.get("scale", 1.0))
        if step is not None and step > 0.0:
            shifted[key] = clamp_request_value(
                key,
                quantize_value(float(shifted[key]) / scale, step) * scale,
                optimizable_vars,
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


def build_robust_neighbor_requests(
    request: dict[str, Any],
    optimizable_vars: dict[str, Any],
    robust_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(robust_eval.get("enabled", False)):
        return []

    neighbors: list[dict[str, Any]] = []
    a_shift_frac = float(robust_eval.get("A_shift_frac", 0.0))
    if a_shift_frac > 0.0 and "A" in request:
        neighbors.append(shifted_request(request, "A", +a_shift_frac, optimizable_vars))
        neighbors.append(shifted_request(request, "A", -a_shift_frac, optimizable_vars))

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
    center_response: dict[str, Any],
    neighbor_responses: list[dict[str, Any]],
    robust_eval: dict[str, Any],
) -> float:
    if not bool(robust_eval.get("enabled", False)) or not neighbor_responses:
        return 0.0

    metric = str(robust_eval["metric"])
    mode = str(robust_eval["mode"])
    tolerance_frac = float(robust_eval["tolerance_frac"])
    penalty_weight = float(robust_eval["penalty_weight"])
    metric_floor = float(robust_eval["metric_floor"])
    aggregate = str(robust_eval.get("aggregate", "mean"))

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
    fragility = max(excesses) if aggregate == "max" else sum(excesses) / len(excesses)
    return penalty_weight * fragility


def build_request(
    eval_id: int,
    candidate: dict[str, Any],
    *,
    default_fee_params: list[float],
    optimizable_vars: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], str | None]:
    request: dict[str, Any] = {"id": eval_id}
    params: dict[str, float] = {}
    fee_params = [float(value) for value in default_fee_params]

    for name, spec in optimizable_vars.items():
        value = float(candidate[name])
        request_key = str(spec.get("request_key", name))
        request_value = clamp_request_value(
            request_key,
            candidate_value_to_request_value(spec, value),
            optimizable_vars,
        )
        params[name] = request_value_to_display_value(spec, request_value)
        if request_key.startswith("fee_param_"):
            idx = int(request_key.removeprefix("fee_param_"))
            fee_params[idx] = request_value
        else:
            request[request_key] = request_value

    request["fee_params"] = fee_params
    return request, params, None


def evaluate_candidate(
    client: EvalServerClient,
    eval_id: int,
    candidate: dict[str, Any],
    *,
    default_fee_params: list[float],
    optimizable_vars: dict[str, Any],
    loss_config: dict[str, Any],
    constraints: dict[str, Any],
    robust_eval: dict[str, Any],
) -> dict[str, Any]:
    request, params, error = build_request(
        eval_id,
        candidate,
        default_fee_params=default_fee_params,
        optimizable_vars=optimizable_vars,
    )
    if error is not None:
        return {
            "step": eval_id,
            "loss": FAIL_PENALTY,
            "params": params,
            "request": request,
            "response": {"ok": False, "error": error},
            "constraint_violations": [],
            "constraint_penalty": FAIL_PENALTY,
        }

    response = client.evaluate(request)
    robust_neighbors = build_robust_neighbor_requests(
        request, optimizable_vars, robust_eval
    )
    neighbor_responses: list[dict[str, Any]] = []
    for idx, neighbor_request in enumerate(robust_neighbors, start=1):
        with_id = dict(neighbor_request)
        with_id["id"] = f"{eval_id}:n{idx}"
        neighbor_responses.append(client.evaluate(with_id))

    if bool(response.get("ok", False)):
        center_loss, constraint_violations, constraint_penalty = compute_loss(
            response, loss_config, constraints
        )
        robust_penalty = (
            0.0
            if constraint_penalty > 0.0
            else compute_robust_penalty(response, neighbor_responses, robust_eval)
        )
        loss = center_loss + robust_penalty
    else:
        center_loss = FAIL_PENALTY
        robust_penalty = 0.0
        constraint_penalty = FAIL_PENALTY
        constraint_violations = []
        loss = FAIL_PENALTY

    return {
        "step": eval_id,
        "loss": loss,
        "center_loss": center_loss,
        "robust_penalty": robust_penalty,
        "constraint_penalty": constraint_penalty,
        "constraint_violations": constraint_violations,
        "params": params,
        "request": request,
        "response": response,
        "neighbor_responses": neighbor_responses,
    }


PARAM_LABELS = {f"fee_param_{i}": FEE_PARAM_LABELS[i] for i in range(FEE_PARAM_COUNT)}


def format_value(key: str, value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if key.startswith("fee_param_"):
        return f"{value:.4g}"
    if key.endswith("_ref"):
        return f"{value:.3e}"
    if abs(value) >= 1000 and float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.4f}"


def format_loss(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def format_apy(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def format_rel_bps(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value * 10_000.0:.1f}bps"


def ordered_param_keys(
    params: dict[str, Any], optimizable_vars: dict[str, Any]
) -> list[str]:
    ordered = list(optimizable_vars)
    for key in sorted(params):
        if key not in ordered:
            ordered.append(key)
    return ordered


def format_param_block(params: dict[str, Any], optimizable_vars: dict[str, Any]) -> str:
    parts = []
    for key in ordered_param_keys(params, optimizable_vars):
        if key not in params:
            continue
        label = PARAM_LABELS.get(key, key)
        parts.append(f"{label}={format_value(key, float(params[key]))}")
    return "(" + ", ".join(parts) + ")"


def format_violations(row: dict[str, Any]) -> str:
    violations = row.get("constraint_violations", [])
    if not violations:
        return "-"
    return ",".join(str(item["name"]) for item in violations)


def print_progress(
    step: int,
    budget: int,
    row: dict[str, Any],
    best_seen: dict[str, Any] | None,
    optimizable_vars: dict[str, Any],
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
        format_param_block(best_seen["params"], optimizable_vars)
        if best_seen is not None
        else "-"
    )
    current_params = format_param_block(params, optimizable_vars)
    center_loss = as_float(row, "center_loss", float("nan"))
    robust_penalty = as_float(row, "robust_penalty", 0.0)
    constraint_penalty = as_float(row, "constraint_penalty", 0.0)
    guardrails = format_violations(row)

    left = (
        f"[{step}/{budget}] cur_l={format_loss(float(row['loss']))} "
        f"center={format_loss(center_loss)} cpen={format_loss(constraint_penalty)} "
        f"rpen={format_loss(robust_penalty)} apy_net={format_apy(apy_net)} "
        f"avg_pdif={format_rel_bps(avg_rel)} max={format_rel_bps(max_rel)} "
        f"guard={guardrails} x={current_params}"
    )
    right = (
        f"best_l={format_loss(best_loss)} apy_net={format_apy(best_apy_net)} "
        f"avg_pdif={format_rel_bps(best_avg_rel)} x={best_params}"
    )

    if bool(response.get("ok", False)):
        print(f"{left} | {right}", flush=True)
        return

    error = str(response.get("error", "unknown error"))
    print(
        f"[{step}/{budget}] curr error={error} x={current_params} | {right}",
        flush=True,
    )


def save_result(payload: dict[str, Any], result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_inspect_best(result_path: Path) -> None:
    inspect_script = REPO_ROOT / "python" / "arb_sim" / "inspect_nevergrad_result.py"
    if not inspect_script.exists():
        print(f"inspect script not found: {inspect_script}", flush=True)
        return

    cmd = ["uv", "run", "python", str(inspect_script), "--result", str(result_path)]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"inspect replay failed: {exc}", flush=True)


def build_optimizer(ng: Any, parametrization: Any, optimization: dict[str, Any]) -> Any:
    workers = max(1, int(optimization["workers"]))
    optimizer_name = str(optimization["optimizer"])
    optimizer_cls = getattr(ng.optimizers, optimizer_name)
    optimizer = optimizer_cls(
        parametrization=parametrization,
        budget=int(optimization["budget"]),
        num_workers=workers,
    )

    seed = optimization.get("seed")
    if seed is not None:
        optimizer.parametrization.random_state.seed(int(seed))
    return optimizer


def open_eval_clients(
    stack: ExitStack,
    binary_path: Path,
    template_path: Path,
    candles_path: Path,
    workers: int,
    server_config: dict[str, Any],
) -> list[EvalServerClient]:
    clients: list[EvalServerClient] = []
    for _ in range(workers):
        clients.append(
            stack.enter_context(
                EvalServerClient(
                    binary=binary_path,
                    template_path=template_path,
                    candles_path=candles_path,
                    server_config=dict(server_config),
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
    *,
    default_fee_params: list[float],
    optimizable_vars: dict[str, Any],
    loss_config: dict[str, Any],
    constraints: dict[str, Any],
    robust_eval: dict[str, Any],
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
            default_fee_params=default_fee_params,
            optimizable_vars=optimizable_vars,
            loss_config=loss_config,
            constraints=constraints,
            robust_eval=robust_eval,
        )
        jobs.append((candidate, future))

    rows: list[tuple[Any, dict[str, Any]]] = []
    for candidate, future in jobs:
        rows.append((candidate, future.result()))
    return rows


def build_saved_pool_payload(
    which: str,
    row: dict[str, Any] | None,
    *,
    candles_path: Path,
    server_config: dict[str, Any],
    template_pools_path: str,
    pool_index: int,
    base_entry: dict[str, Any],
    default_fee_params: list[float],
) -> dict[str, Any] | None:
    if row is None:
        return None

    request_params = dict(row.get("request", {}))
    request_params.pop("id", None)
    pool_config_entry = canonical_pool_config_entry(
        base_entry,
        request_params,
        default_fee_params,
    )
    pool_obj = pool_config_entry.get("pool", {})
    return {
        "which": which,
        "candles_path": str(candles_path),
        "server_config": dict(server_config),
        "template_pools_path": template_pools_path,
        "pool_index": pool_index,
        "template_pool": dict(pool_obj) if isinstance(pool_obj, dict) else {},
        "template_costs": dict(pool_config_entry.get("costs", {})),
        "params_display": dict(row.get("params", {})),
        "request_params": request_params,
        "pool_config_entry": pool_config_entry,
    }


def run_optimization(
    ng: Any,
    binary_path: Path,
    args: argparse.Namespace,
    *,
    fee_params_override: list[float] | None = None,
    optimizable_vars_override: dict[str, Any] | None = None,
    entrypoint: str = "nevergrad_fee_runner",
) -> dict[str, Any]:
    optimizable_vars = copy.deepcopy(
        DEFAULT_OPTIMIZABLE_VARS
        if optimizable_vars_override is None
        else optimizable_vars_override
    )
    validate_optimizable_vars(optimizable_vars)
    server_config = {
        "pool_index": int(args.pool_index),
        "n_candles": int(args.n_candles),
        "candle_filter": float(args.candle_filter),
        "dustswapfreq": int(args.dustswapfreq),
        "min_swap": float(args.min_swap),
        "max_swap": float(args.max_swap),
        "disable_slippage_probes": bool(args.disable_slippage_probes),
    }
    optimization = {
        "optimizer": args.optimizer,
        "budget": int(args.budget),
        "seed": int(args.seed),
        "verbose_every": int(args.verbose_every),
        "workers": int(args.workers),
    }
    loss_config = {
        "metric": args.score_metric,
        "avg_rel_target": float(args.avg_rel_target_bps) / 10_000.0,
        "avg_rel_lambda": float(args.avg_rel_lambda),
    }
    constraints = {
        "enabled": bool(args.hard_constraints),
        "avg_rel_price_diff_max": float(args.max_avg_rel_bps) / 10_000.0,
        "max_rel_price_diff_max": float(args.max_rel_bps) / 10_000.0,
        "tw_real_slippage_5pct_max": float(args.max_tw_slippage_5pct_bps) / 10_000.0,
        "min_trades": int(args.min_trades),
    }
    robust_eval = dict(DEFAULT_ROBUST_EVAL)

    candles_path = args.candles.resolve()
    git_commit = git_sha(REPO_ROOT)
    fee_model_hash = file_hash(
        REPO_ROOT
        / "cpp_modular"
        / "include"
        / "pools"
        / "twocrypto_fx"
        / "fee_model.hpp"
    )
    evaluator_code_hash = combined_hash(EVALUATOR_HASH_INPUTS)

    parametrization = build_parametrization(ng, optimizable_vars)
    optimizer = build_optimizer(ng, parametrization, optimization)
    optimizer_name = str(optimization["optimizer"])
    budget = int(optimization["budget"])
    seed = optimization.get("seed")
    verbose_every = max(1, int(optimization["verbose_every"]))
    workers = max(1, min(int(optimization["workers"]), budget))

    started = time.time()
    best_seen: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    interrupted = False
    model_metadata: dict[str, Any] = {
        "git_sha": git_commit,
        "fee_model_hash": fee_model_hash,
        "evaluator_code_hash": evaluator_code_hash,
    }

    with tempfile.TemporaryDirectory(prefix="nevergrad_fee_runner_") as tmp_dir:
        if args.template_pools is not None:
            template_pools_path = args.template_pools.resolve()
            template_source = str(template_pools_path)
        else:
            template_pools_path = write_embedded_template_file(
                Path(tmp_dir) / "embedded_template_pools.json",
                candles_path,
            )
            template_source = "<embedded-template>"

        base_entry = load_pool_config_entry(template_pools_path, int(args.pool_index))
        base_pool = dict(base_entry.get("pool", base_entry))
        default_fee_params = fee_params_with_override(base_pool, fee_params_override)

        with ExitStack() as stack:
            clients = open_eval_clients(
                stack,
                binary_path=binary_path,
                template_path=template_pools_path,
                candles_path=candles_path,
                workers=workers,
                server_config=server_config,
            )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                next_step = 1
                try:
                    while next_step <= budget:
                        batch_size = min(workers, budget - next_step + 1)
                        batch_rows = run_parallel_batch(
                            executor,
                            optimizer,
                            clients,
                            next_step,
                            batch_size,
                            default_fee_params=default_fee_params,
                            optimizable_vars=optimizable_vars,
                            loss_config=loss_config,
                            constraints=constraints,
                            robust_eval=robust_eval,
                        )

                        for candidate, row in batch_rows:
                            optimizer.tell(candidate, float(row["loss"]))

                            response = row["response"]
                            if bool(response.get("ok", False)):
                                model_metadata.setdefault(
                                    "fee_model_name", response.get("fee_model_name")
                                )
                                model_metadata.setdefault(
                                    "fee_param_labels", response.get("fee_param_labels")
                                )

                            if best_seen is None or float(row["loss"]) < float(
                                best_seen["loss"]
                            ):
                                best_seen = row

                            step = int(row["step"])
                            if step == 1 or step == budget or step % verbose_every == 0:
                                print_progress(
                                    step, budget, row, best_seen, optimizable_vars
                                )

                        next_step += batch_size

                    recommendation_candidate = optimizer.provide_recommendation()
                    recommendation = evaluate_candidate(
                        clients[0],
                        budget + 1,
                        recommendation_candidate.kwargs,
                        default_fee_params=default_fee_params,
                        optimizable_vars=optimizable_vars,
                        loss_config=loss_config,
                        constraints=constraints,
                        robust_eval=robust_eval,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    print("\nInterrupted, saving best-so-far...", flush=True)

    elapsed_s = time.time() - started
    return {
        "entrypoint": entrypoint,
        "optimizer": optimizer_name,
        "budget": budget,
        "seed": seed,
        "workers": workers,
        "elapsed_s": elapsed_s,
        "candles_path": str(candles_path),
        "template_pools_path": template_source,
        "pool_index": int(args.pool_index),
        "server_config": server_config,
        "default_fee_params": default_fee_params,
        "base_pool_entry": base_entry,
        "optimizable_vars": optimizable_vars,
        "loss_config": loss_config,
        "constraints": constraints,
        "robust_eval": robust_eval,
        "model_metadata": model_metadata,
        "interrupted": interrupted,
        "best_seen": best_seen,
        "recommendation": recommendation,
        "best_pool": build_saved_pool_payload(
            "best_seen",
            best_seen,
            candles_path=candles_path,
            server_config=server_config,
            template_pools_path=template_source,
            pool_index=int(args.pool_index),
            base_entry=base_entry,
            default_fee_params=default_fee_params,
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical Nevergrad fee search against arb_eval_server"
    )
    parser.add_argument(
        "--template-pools",
        type=Path,
        default=None,
        help="Optional external pools file. If omitted, use the embedded human-owned template in this script.",
    )
    parser.add_argument(
        "--pool-index", type=int, default=DEFAULT_SERVER_CONFIG["pool_index"]
    )
    parser.add_argument("--candles", type=Path, default=DEFAULT_CANDLES_PATH)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--budget", type=int, default=DEFAULT_OPTIMIZATION["budget"])
    parser.add_argument("--workers", type=int, default=DEFAULT_OPTIMIZATION["workers"])
    parser.add_argument("--seed", type=int, default=DEFAULT_OPTIMIZATION["seed"])
    parser.add_argument(
        "--optimizer", type=str, default=DEFAULT_OPTIMIZATION["optimizer"]
    )
    parser.add_argument(
        "--verbose-every", type=int, default=DEFAULT_OPTIMIZATION["verbose_every"]
    )
    parser.add_argument(
        "--n-candles", type=int, default=DEFAULT_SERVER_CONFIG["n_candles"]
    )
    parser.add_argument(
        "--candle-filter", type=float, default=DEFAULT_SERVER_CONFIG["candle_filter"]
    )
    parser.add_argument(
        "--dustswapfreq", type=int, default=DEFAULT_SERVER_CONFIG["dustswapfreq"]
    )
    parser.add_argument(
        "--min-swap", type=float, default=DEFAULT_SERVER_CONFIG["min_swap"]
    )
    parser.add_argument(
        "--max-swap", type=float, default=DEFAULT_SERVER_CONFIG["max_swap"]
    )
    parser.add_argument(
        "--disable-slippage-probes",
        dest="disable_slippage_probes",
        action="store_true",
        help="Disable slippage probes in arb_eval_server.",
    )
    parser.add_argument(
        "--enable-slippage-probes",
        dest="disable_slippage_probes",
        action="store_false",
        help="Enable slippage probes in arb_eval_server.",
    )
    parser.set_defaults(
        disable_slippage_probes=DEFAULT_SERVER_CONFIG["disable_slippage_probes"]
    )
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--inspect-best", action="store_true")
    parser.add_argument(
        "--score-metric", type=str, default=DEFAULT_LOSS_CONFIG["metric"]
    )
    parser.add_argument(
        "--avg-rel-target-bps",
        type=float,
        default=DEFAULT_LOSS_CONFIG["avg_rel_target"] * 10_000.0,
    )
    parser.add_argument(
        "--avg-rel-lambda",
        type=float,
        default=DEFAULT_LOSS_CONFIG["avg_rel_lambda"],
    )
    parser.add_argument("--hard-constraints", action="store_true")
    parser.add_argument(
        "--max-avg-rel-bps",
        type=float,
        default=DEFAULT_CONSTRAINTS["avg_rel_price_diff_max"] * 10_000.0,
    )
    parser.add_argument(
        "--max-rel-bps",
        type=float,
        default=DEFAULT_CONSTRAINTS["max_rel_price_diff_max"] * 10_000.0,
    )
    parser.add_argument(
        "--max-tw-slippage-5pct-bps",
        type=float,
        default=DEFAULT_CONSTRAINTS["tw_real_slippage_5pct_max"] * 10_000.0,
    )
    parser.add_argument(
        "--min-trades", type=int, default=DEFAULT_CONSTRAINTS["min_trades"]
    )
    return parser.parse_args(argv)


def run_with_args(
    args: argparse.Namespace,
    *,
    fee_params_override: list[float] | None = None,
    optimizable_vars_override: dict[str, Any] | None = None,
    entrypoint: str = "nevergrad_fee_runner",
) -> dict[str, Any]:
    ng = require_nevergrad()
    binary_path = ensure_binary(
        DEFAULT_BINARY_PATH, skip_rebuild=bool(args.skip_rebuild)
    )
    candles_path = args.candles.resolve()
    if not candles_path.exists():
        raise SystemExit(f"candles file not found: {candles_path}")
    if args.template_pools is not None and not args.template_pools.exists():
        raise SystemExit(f"template pools file not found: {args.template_pools}")
    return run_optimization(
        ng,
        binary_path,
        args,
        fee_params_override=fee_params_override,
        optimizable_vars_override=optimizable_vars_override,
        entrypoint=entrypoint,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_with_args(args)
    best_seen = payload["best_seen"]
    if best_seen is None:
        save_result(payload, args.result)
        raise SystemExit("no evaluations were executed")

    save_result(payload, args.result)

    print("best_seen:", json.dumps(best_seen, separators=(",", ":"), sort_keys=True))
    print(f"saved: {args.result}")
    if args.inspect_best:
        run_inspect_best(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
