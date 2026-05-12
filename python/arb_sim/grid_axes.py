from __future__ import annotations

import math
from typing import Any


PoolObject = dict[str, Any]
RunObject = dict[str, Any]


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def finite_float_or_none(value: Any) -> float | None:
    v = to_float(value)
    return v if v is not None and math.isfinite(v) else None


def pool_for_run(run: RunObject) -> PoolObject:
    """Return the pool config echo from postprocessed or raw harness output."""
    params = run.get("params")
    if isinstance(params, dict):
        pool = params.get("pool")
        if isinstance(pool, dict):
            return pool
    pool = run.get("pool")
    return pool if isinstance(pool, dict) else {}


def pool_value(pool: PoolObject, name: str) -> Any:
    """Look up a possibly dotted pool parameter name."""
    if name in pool:
        return pool.get(name)
    current: Any = pool
    for part in name.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def iter_pool_scalar_values(pool: PoolObject) -> list[tuple[str, float]]:
    """Yield scalar numeric pool params in config order.

    Raw harness outputs do not carry `metadata.grid`, so reporting tools infer
    axes from the scalar fields that vary across `params.pool`. When the
    generator stores both `policy.fee_bps` and serialized `policy.fee`, keep the
    user-facing bps field and skip the redundant internal fee units.
    """
    skip_policy_fee = isinstance(pool.get("policy"), dict) and "fee_bps" in pool["policy"]
    values: list[tuple[str, float]] = []

    def visit(obj: PoolObject, prefix: str = "") -> None:
        for key, raw in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if name == "policy.fee" and skip_policy_fee:
                continue
            if isinstance(raw, dict):
                visit(raw, name)
                continue
            if isinstance(raw, (list, tuple)):
                continue
            v = finite_float_or_none(raw)
            if v is not None:
                values.append((name, v))

    visit(pool)
    return values


def infer_grid_from_runs(
    runs: list[RunObject],
) -> tuple[list[str], list[list[float]], list[str]]:
    """Infer varying scalar pool axes from a raw or postprocessed run list."""
    positions: dict[str, int] = {}
    values: dict[str, set[float]] = {}

    for run in runs:
        for name, value in iter_pool_scalar_values(pool_for_run(run)):
            if name not in positions:
                positions[name] = len(positions)
            values.setdefault(name, set()).add(value)

    names = [name for name, vals in values.items() if len(vals) > 1]
    names.sort(key=lambda name: positions[name])
    grid_values = [sorted(values[name]) for name in names]
    x_keys = [f"x{i + 1}" for i in range(len(names))]
    return names, grid_values, x_keys
