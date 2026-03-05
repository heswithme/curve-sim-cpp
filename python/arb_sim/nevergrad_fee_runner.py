#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
import nevergrad as ng

try:
    from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool
except ModuleNotFoundError:  # Support module import via python.arb_sim
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

SERVER_CONFIG = {
    "pool_index": 0,
    "n_candles": 0,
    "candle_filter": 99.0,
    "min_swap": 1e-6,
    "max_swap": 1.0,
}

TEMPLATE_POOL = {
    "A": 6 * 10_000,
    "gamma": 0.0001,
    "allowed_extra_profit": 1e-10,
    "adjustment_step": 0.005,
    "ma_time": 866.0,
    "donation_apy": 0.036,
    "donation_frequency": 86400.0,
    "donation_coins_ratio": 0.5,
    "initial_liq_coin0": 10_000_000.0,
    "mid_fee_bps_default": 1.0,
    "out_fee_bps_default": 1.0,
    "fee_gamma_default": 0.003,
}

TEMPLATE_COSTS = {
    "arb_fee_bps": 3.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1.0,
}

OPTIMIZABLE_VARS = {
    "mid_fee_bps": {"lower": 1.0, "upper": 300.0},
    "spread_bps": {"lower": 0.0, "upper": 300.0},
    "fee_gamma": {"lower": 1e-6, "upper": 0.05, "log": True},
}

OUT_FEE_BPS_MAX = 600.0

MAX_AVG_PRICE_REL_DIFF = 0.1
PRICE_DIFF_PENALTY = 1_000.0

OPTIMIZATION = {
    "optimizer": "TwoPointsDE",
    "budget": 1000,
    "seed": 42,
    "verbose_every": 10,
}

FAIL_PENALTY = 1_000_000.0


def scale_1e18(value: float) -> int:
    return int(round(value * 1e18))


def fee_bps_to_1e10(value_bps: float) -> int:
    return int(round((value_bps / 10_000.0) * 1e10))


def build_embedded_template(candles_path: Path) -> dict[str, Any]:
    start_ts = _first_candle_ts(str(candles_path))
    init_price = _initial_price_from_file(str(candles_path))

    init_liq_coin0 = float(TEMPLATE_POOL["initial_liq_coin0"])
    init_liq_0 = int((init_liq_coin0 * 1e18) // 2)
    init_liq_1 = int((init_liq_coin0 * 1e18) // 2 / init_price)

    pool_raw = {
        "initial_liquidity": [init_liq_0, init_liq_1],
        "A": float(TEMPLATE_POOL["A"]),
        "gamma": float(TEMPLATE_POOL["gamma"]),
        "mid_fee": fee_bps_to_1e10(float(TEMPLATE_POOL["mid_fee_bps_default"])),
        "out_fee": fee_bps_to_1e10(float(TEMPLATE_POOL["out_fee_bps_default"])),
        "fee_gamma": scale_1e18(float(TEMPLATE_POOL["fee_gamma_default"])),
        "allowed_extra_profit": scale_1e18(
            float(TEMPLATE_POOL["allowed_extra_profit"])
        ),
        "adjustment_step": scale_1e18(float(TEMPLATE_POOL["adjustment_step"])),
        "ma_time": float(TEMPLATE_POOL["ma_time"]),
        "initial_price": scale_1e18(init_price),
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
                "pool": strify_pool(pool_raw),
                "costs": dict(TEMPLATE_COSTS),
            }
        ],
    }


def write_embedded_template(candles_path: Path, workdir: Path) -> Path:
    template = build_embedded_template(candles_path)
    out_path = workdir / "embedded_pool_config.json"
    out_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return out_path


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


class EvalServerClient:
    def __init__(
        self,
        binary: Path,
        template_path: Path,
        candles_path: Path,
        pool_index: int = 0,
        n_candles: int = 0,
        candle_filter: float = 99.0,
        min_swap: float = 1e-6,
        max_swap: float = 1.0,
    ) -> None:
        self.binary = binary
        self.template_path = template_path
        self.candles_path = candles_path
        self.pool_index = pool_index
        self.n_candles = n_candles
        self.candle_filter = candle_filter
        self.min_swap = min_swap
        self.max_swap = max_swap
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "EvalServerClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        if self.proc is not None:
            return
        cmd = [
            str(self.binary),
            str(self.template_path),
            str(self.candles_path),
            "--pool-index",
            str(self.pool_index),
            "--n-candles",
            str(self.n_candles),
            "--candle-filter",
            str(self.candle_filter),
            "--min-swap",
            str(self.min_swap),
            "--max-swap",
            str(self.max_swap),
        ]
        self.proc = subprocess.Popen(
            cmd,
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
                if self.proc.stdin:
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

        payload = json.dumps(request, separators=(",", ":"))
        self.proc.stdin.write(payload + "\n")
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
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return number


def compute_loss(response: dict[str, Any]) -> float:
    apy_net = as_float(response, "apy_net", -1.0)
    avg_price_rel_diff = abs(
        as_float(
            response,
            "price_diff",
            as_float(response, "avg_rel_price_diff", 1.0),
        )
    )

    loss = -apy_net
    if avg_price_rel_diff > MAX_AVG_PRICE_REL_DIFF:
        excess = avg_price_rel_diff - MAX_AVG_PRICE_REL_DIFF
        loss += PRICE_DIFF_PENALTY * (1.0 + excess / MAX_AVG_PRICE_REL_DIFF)
    return loss


def build_request(
    eval_id: int, mid_fee_bps: float, spread_bps: float, fee_gamma: float
) -> dict[str, Any]:
    out_fee_bps = mid_fee_bps + spread_bps
    request: dict[str, Any] = {
        "id": eval_id,
        "mid_fee_bps": mid_fee_bps,
        "out_fee_bps": out_fee_bps,
        "fee_gamma": fee_gamma,
    }
    return request


def evaluate_candidate(
    client: EvalServerClient,
    eval_id: int,
    mid_fee_bps: float,
    spread_bps: float,
    fee_gamma: float,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    out_fee_bps = mid_fee_bps + spread_bps
    if out_fee_bps > OUT_FEE_BPS_MAX:
        penalty = FAIL_PENALTY + (out_fee_bps - OUT_FEE_BPS_MAX) * 1000.0
        response = {
            "ok": False,
            "error": f"out_fee_bps {out_fee_bps:.8f} exceeds max {OUT_FEE_BPS_MAX:.8f}",
        }
        params = {
            "mid_fee_bps": mid_fee_bps,
            "out_fee_bps": out_fee_bps,
            "fee_gamma": fee_gamma,
            "spread_bps": spread_bps,
        }
        return penalty, response, params

    request = build_request(eval_id, mid_fee_bps, spread_bps, fee_gamma)
    response = client.evaluate(request)
    ok = bool(response.get("ok", False))
    loss = compute_loss(response) if ok else FAIL_PENALTY
    params = {
        "mid_fee_bps": mid_fee_bps,
        "out_fee_bps": out_fee_bps,
        "fee_gamma": fee_gamma,
        "spread_bps": spread_bps,
    }
    return loss, response, params


def ensure_binary() -> Path:
    binary_path = BINARY_PATH.resolve()
    if FORCE_REBUILD_BINARY or (BUILD_BINARY_IF_MISSING and not binary_path.exists()):
        build_eval_server(REPO_ROOT)
    if not binary_path.exists():
        raise SystemExit(f"evaluator binary not found: {binary_path}")
    return binary_path


def print_progress(
    step: int,
    budget: int,
    loss: float,
    params: dict[str, float],
    response: dict[str, Any],
    best_seen: dict[str, Any] | None,
) -> None:
    apy_net = as_float(response, "apy_net", float("nan"))
    price_diff = as_float(response, "price_diff", float("nan"))
    ok = bool(response.get("ok", False))

    best_loss = float("nan")
    best_mid = float("nan")
    best_out = float("nan")
    best_gamma = float("nan")
    if best_seen is not None:
        best_loss = as_float(best_seen, "loss", float("nan"))
        best_params = best_seen.get("params", {})
        if isinstance(best_params, dict):
            best_mid = as_float(best_params, "mid_fee_bps", float("nan"))
            best_out = as_float(best_params, "out_fee_bps", float("nan"))
            best_gamma = as_float(best_params, "fee_gamma", float("nan"))

    print(
        f"[{step}/{budget}] ok={ok} loss={loss:.9f} "
        f"mid={params['mid_fee_bps']:.6f} out={params['out_fee_bps']:.6f} "
        f"gamma={params['fee_gamma']:.9f} apy_net={apy_net:.9f} "
        f"price_diff={price_diff:.9f} best_loss={best_loss:.9f} "
        f"best_x=({best_mid:.6f},{best_out:.6f},{best_gamma:.9f})",
        flush=True,
    )


def save_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    binary_path = ensure_binary()

    candles_path = CANDLES_PATH.resolve()
    if not candles_path.exists():
        raise SystemExit(f"candles file not found: {candles_path}")

    parametrization = ng.p.Instrumentation(
        mid_fee_bps=ng.p.Scalar(
            lower=float(OPTIMIZABLE_VARS["mid_fee_bps"]["lower"]),
            upper=float(OPTIMIZABLE_VARS["mid_fee_bps"]["upper"]),
        ),
        spread_bps=ng.p.Scalar(
            lower=float(OPTIMIZABLE_VARS["spread_bps"]["lower"]),
            upper=float(OPTIMIZABLE_VARS["spread_bps"]["upper"]),
        ),
        fee_gamma=ng.p.Log(
            lower=float(OPTIMIZABLE_VARS["fee_gamma"]["lower"]),
            upper=float(OPTIMIZABLE_VARS["fee_gamma"]["upper"]),
        ),
    )

    optimizer_name = str(OPTIMIZATION["optimizer"])
    optimizer_cls = getattr(ng.optimizers, optimizer_name)
    optimizer = optimizer_cls(
        parametrization=parametrization,
        budget=int(OPTIMIZATION["budget"]),
        num_workers=1,
    )

    seed = OPTIMIZATION.get("seed")
    if seed is not None:
        optimizer.parametrization.random_state.seed(int(seed))

    budget = int(OPTIMIZATION["budget"])
    verbose_every = max(1, int(OPTIMIZATION["verbose_every"]))
    started = time.time()

    best_seen: dict[str, Any] | None = None
    template_path_str = ""

    with tempfile.TemporaryDirectory(prefix="arb_eval_template_") as tmp_dir:
        template_path = write_embedded_template(candles_path, Path(tmp_dir))
        template_path_str = str(template_path)

        with EvalServerClient(
            binary=binary_path,
            template_path=template_path,
            candles_path=candles_path,
            pool_index=int(SERVER_CONFIG["pool_index"]),
            n_candles=int(SERVER_CONFIG["n_candles"]),
            candle_filter=float(SERVER_CONFIG["candle_filter"]),
            min_swap=float(SERVER_CONFIG["min_swap"]),
            max_swap=float(SERVER_CONFIG["max_swap"]),
        ) as client:
            for step in range(1, budget + 1):
                candidate = optimizer.ask()
                kw = candidate.kwargs
                loss, response, params = evaluate_candidate(
                    client=client,
                    eval_id=step,
                    mid_fee_bps=float(kw["mid_fee_bps"]),
                    spread_bps=float(kw["spread_bps"]),
                    fee_gamma=float(kw["fee_gamma"]),
                )
                optimizer.tell(candidate, loss)

                row = {
                    "step": step,
                    "loss": loss,
                    "params": params,
                    "response": response,
                }

                if best_seen is None or loss < float(best_seen["loss"]):
                    best_seen = row

                if step == 1 or step == budget or (step % verbose_every == 0):
                    print_progress(step, budget, loss, params, response, best_seen)

            rec = optimizer.provide_recommendation()
            rec_kw = rec.kwargs
            rec_loss, rec_response, rec_params = evaluate_candidate(
                client=client,
                eval_id=budget + 1,
                mid_fee_bps=float(rec_kw["mid_fee_bps"]),
                spread_bps=float(rec_kw["spread_bps"]),
                fee_gamma=float(rec_kw["fee_gamma"]),
            )

    elapsed = time.time() - started

    payload = {
        "optimizer": optimizer_name,
        "budget": budget,
        "seed": seed,
        "elapsed_s": elapsed,
        "template_path": template_path_str,
        "candles_path": str(candles_path),
        "server_config": SERVER_CONFIG,
        "template_pool": TEMPLATE_POOL,
        "template_costs": TEMPLATE_COSTS,
        "optimizable_vars": OPTIMIZABLE_VARS,
        "max_avg_price_rel_diff": MAX_AVG_PRICE_REL_DIFF,
        "price_diff_penalty": PRICE_DIFF_PENALTY,
        "best_seen": best_seen,
        "recommendation": {
            "loss": rec_loss,
            "params": rec_params,
            "response": rec_response,
        },
    }
    save_result(payload)

    if best_seen is None:
        print("no evaluations were executed", file=sys.stderr)
        return 1

    print("best_seen:", json.dumps(best_seen, separators=(",", ":"), sort_keys=True))
    print(
        "recommendation:",
        json.dumps(payload["recommendation"], separators=(",", ":"), sort_keys=True),
    )
    print(f"saved: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
