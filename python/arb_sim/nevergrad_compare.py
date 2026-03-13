#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    import nevergrad_fee_runner as runner
except ModuleNotFoundError:
    try:
        from . import nevergrad_fee_runner as runner
    except ImportError:
        from python.arb_sim import nevergrad_fee_runner as runner


# Flip this between "inventory_only", "full", and "gap_only" before running.
ACTIVE_PRESET = "gap_only"


BASE_TEMPLATE_POOL = copy.deepcopy(runner.TEMPLATE_POOL)


def current_fee_params(
    *,
    base_floor: float = 0.0,
    mid_fee_bps: float = 1.0,
    out_fee_bps: float = 1.0,
    fee_gamma: float = 0.003,
    calm_discount_max: float = 0.0,
    fee_volatility_ref: float = 0.0,
    gap_fee_scale: float = 0.0,
    gap_fee_const_discount: float = 0.0,
) -> list[float]:
    return runner.make_current_fee_params(
        base_floor=base_floor / 10_000.0,
        mid_fee=mid_fee_bps / 10_000.0,
        out_fee=out_fee_bps / 10_000.0,
        fee_gamma=fee_gamma,
        calm_discount_max=calm_discount_max,
        fee_volatility_ref=fee_volatility_ref,
        gap_fee_scale=gap_fee_scale,
        gap_fee_const_discount=gap_fee_const_discount,
    )


def fee_var(index: int, spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["request_key"] = f"fee_param_{index}"
    return out


PRESETS: dict[str, dict[str, Any]] = {
    "inventory_only": {
        "description": "Inventory-only fee sketch on the generic fee_params surface.",
        "result_path": runner.REPO_ROOT
        / "comparison-results"
        / "nevergrad_fee_result_inventory_only.json",
        "template_overrides": {
            "fee_params": current_fee_params(
                base_floor=0.0,
                mid_fee_bps=1.0,
                out_fee_bps=1.0,
                fee_gamma=0.003,
            ),
        },
        "optimizable_vars": {
            "fee_param_1": fee_var(runner.FEE_PARAM_MID_FEE, runner.scalar(1.0 / 10_000.0, 300.0 / 10_000.0, step=1.0 / 10_000.0)),
            "fee_param_2": fee_var(runner.FEE_PARAM_SPREAD, runner.scalar(0.0, 300.0 / 10_000.0, step=1.0 / 10_000.0)),
            # "fee_param_3": fee_var(runner.FEE_PARAM_FEE_GAMMA, runner.log_scale(1e-6, 0.05)),
        },
    },
    "full": {
        "description": "Current additive sketch on the generic fee_params surface.",
        "result_path": runner.REPO_ROOT
        / "comparison-results"
        / "nevergrad_fee_result_full.json",
        "template_overrides": {
            "fee_params": current_fee_params(
                base_floor=0.0,
                mid_fee_bps=1.0,
                out_fee_bps=1.0,
                fee_gamma=0.003,
                calm_discount_max=0.25,
                fee_volatility_ref=0.01,
                gap_fee_scale=1.0,
                gap_fee_const_discount=0.0025,
            ),
        },
        "optimizable_vars": {
            "fee_param_0": fee_var(runner.FEE_PARAM_BASE_FLOOR, runner.scalar(0.0, 300.0 / 10_000.0, step=1.0 / 10_000.0)),
            "fee_param_1": fee_var(runner.FEE_PARAM_MID_FEE, runner.scalar(1.0 / 10_000.0, 300.0 / 10_000.0, step=1.0 / 10_000.0)),
            "fee_param_2": fee_var(runner.FEE_PARAM_SPREAD, runner.scalar(0.0, 300.0 / 10_000.0, step=1.0 / 10_000.0)),
            # "fee_param_3": fee_var(runner.FEE_PARAM_FEE_GAMMA, runner.log_scale(1e-6, 0.05)),
            "fee_param_4": fee_var(runner.FEE_PARAM_CALM_DISCOUNT_MAX, runner.scalar(0.0, 1.0, step=0.01)),
            "fee_param_5": fee_var(runner.FEE_PARAM_FEE_VOLATILITY_REF, runner.log_scale(1e-4, 0.1)),
            "fee_param_6": fee_var(runner.FEE_PARAM_GAP_FEE_SCALE, runner.log_scale(1e-2, 10.0)),
            "fee_param_7": fee_var(runner.FEE_PARAM_GAP_FEE_CONST_DISCOUNT, runner.scalar(0.0, 0.05, step=0.0005)),
        },
    },
    "gap_only": {
        "description": "Standalone base-plus-gap sketch on the generic fee_params surface.",
        "result_path": runner.REPO_ROOT
        / "comparison-results"
        / "nevergrad_fee_result_gap_only.json",
        "template_overrides": {
            "fee_params": current_fee_params(
                base_floor=10.0,
                mid_fee_bps=0.0,
                out_fee_bps=0.0,
                fee_gamma=0.0,
                calm_discount_max=0.0,
                fee_volatility_ref=0.0,
                gap_fee_scale=1.0,
                gap_fee_const_discount=0.0025,
            ),
        },
        "optimizable_vars": {
            "fee_param_0": fee_var(runner.FEE_PARAM_BASE_FLOOR, runner.scalar(0.0, 300.0 / 10_000.0, step=1.0 / 10_000.0)),
            "fee_param_6": fee_var(runner.FEE_PARAM_GAP_FEE_SCALE, runner.log_scale(1e-2, 10.0)),
            "fee_param_7": fee_var(runner.FEE_PARAM_GAP_FEE_CONST_DISCOUNT, runner.scalar(0.0, 0.05, step=0.0005)),
        },
    },
}


def apply_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        raise SystemExit(
            f"unknown preset: {name} (expected one of: {', '.join(sorted(PRESETS))})"
        )

    preset = PRESETS[name]
    template_pool = copy.deepcopy(BASE_TEMPLATE_POOL)
    template_pool.update(copy.deepcopy(preset["template_overrides"]))

    runner.TEMPLATE_POOL = template_pool
    runner.OPTIMIZABLE_VARS = copy.deepcopy(preset["optimizable_vars"])
    runner.RESULT_PATH = Path(preset["result_path"])

    return preset


def main() -> int:
    preset = apply_preset(ACTIVE_PRESET)

    print(f"preset: {ACTIVE_PRESET}")
    print(f"description: {preset['description']}")
    print(
        "optimizable_vars:",
        ", ".join(runner.OPTIMIZABLE_VARS.keys()),
        flush=True,
    )

    ng = runner.require_nevergrad()
    binary_path = runner.ensure_binary()
    candles_path = runner.CANDLES_PATH.resolve()
    if not candles_path.exists():
        raise SystemExit(f"candles file not found: {candles_path}")

    payload = runner.run_optimization(ng, binary_path, candles_path)
    payload["active_preset"] = ACTIVE_PRESET
    payload["preset_description"] = preset["description"]
    payload["preset_result_path"] = str(runner.RESULT_PATH)

    best_seen = payload["best_seen"]
    if best_seen is None:
        runner.save_result(payload)
        raise SystemExit("no evaluations were executed")

    runner.save_result(payload)

    print("best_seen:", json.dumps(best_seen, separators=(",", ":"), sort_keys=True))
    print(f"saved: {runner.RESULT_PATH}")
    runner.run_inspect_best()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
