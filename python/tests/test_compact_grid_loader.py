import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "cpp_modular" / "build" / "arb_harness"


def _base_pool() -> dict[str, Any]:
    price = 100.0
    return {
        "initial_liquidity": [
            str(int(500_000 * 10**18)),
            str(int(500_000 * 10**18 / price)),
        ],
        "A": str(50 * 10_000),
        "gamma": str(10**14),
        "mid_fee": str(int(4 / 10_000 * 10**10)),
        "out_fee": str(int(30 / 10_000 * 10**10)),
        "fee_gamma": str(int(0.1 * 10**18)),
        "adjustment_step_min": str(int(0.0000000001 * 10**18)),
        "adjustment_step_max": str(int(0.5 / 100 * 10**18)),
        "ma_time": "865",
        "reserved_profit_fraction": str(int(0.5 * 10**10)),
        "admin_fee": "0",
        "policy": {"kind": "none"},
        "initial_price": str(int(price * 10**18)),
        "start_timestamp": "1704067200",
        "donation_apy": "0.01",
        "donation_frequency": "604800",
        "donation_coins_ratio": "0.5",
    }


def _costs() -> dict[str, Any]:
    return {
        "arb_fee_bps": 2,
        "gas_coin0": 0.0,
        "use_volume_cap": False,
        "volume_cap_mult": 1,
    }


def _candles(path: Path, n: int = 240) -> None:
    ts0 = 1704067200
    rows = []
    for i in range(n):
        ts = ts0 + i * 60
        price = 100.0 + 5.0 * math.sin(i / 17.0) + i * 0.002
        rows.append([ts, price, price * 1.001, price * 0.999, price, 1.0])
    path.write_text(json.dumps(rows))


def _set_grid_value(pool: dict[str, Any], name: str, value: Any) -> None:
    if name == "policy.fee_bps":
        pool["policy"] = {"kind": "fixed_fee", "fee_bps": value}
        return
    if "." not in name:
        pool[name] = str(value)
        return
    current = pool
    parts = name.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _expanded_config(axes: dict[str, list[Any]], *, fee_equalize: bool = False) -> dict[str, Any]:
    base_pool = _base_pool()
    costs = _costs()
    axis_items = list(axes.items())
    grid_meta = {
        f"x{i}": {"name": name, "values": values}
        for i, (name, values) in enumerate(axis_items, 1)
    }
    pools = []
    for a in axis_items[0][1]:
        for b in axis_items[1][1]:
            pool = copy.deepcopy(base_pool)
            costs_for_pool = dict(costs)
            coords = [a, b]
            for (name, _values), value in zip(axis_items, coords):
                if name.startswith("costs."):
                    costs_for_pool[name.removeprefix("costs.")] = value
                else:
                    _set_grid_value(pool, name, value)
            if fee_equalize and any(name in {"mid_fee", "out_fee"} for name, _ in axis_items):
                pool["out_fee"] = pool["mid_fee"]
            pools.append(
                {
                    "tag": "__".join(
                        f"{name}_{value}" for (name, _values), value in zip(axis_items, coords)
                    ),
                    "pool": pool,
                    "costs": costs_for_pool,
                }
            )
    return {
        "meta": {
            "grid": grid_meta,
            "base_pool": base_pool,
            "base_costs": costs,
            "pool_count": len(pools),
            "fee_equalize": fee_equalize,
        },
        "pools": pools,
    }


def _compact_config(axes: dict[str, list[Any]], *, fee_equalize: bool = False) -> dict[str, Any]:
    expanded = _expanded_config(axes, fee_equalize=fee_equalize)
    return {"meta": {**expanded["meta"], "compact_grid": True}}


def _run_harness(
    cfg: dict[str, Any],
    candles: Path,
    tmp_path: Path,
    name: str,
    *,
    pool_start: int | None = None,
    pool_end: int | None = None,
) -> dict[str, Any]:
    cfg_path = tmp_path / f"{name}.json"
    out_path = tmp_path / f"{name}_out.json"
    cfg_path.write_text(json.dumps(cfg))
    cmd = [
        str(HARNESS),
        str(cfg_path),
        str(candles),
        str(out_path),
        "--threads",
        "4",
        "--dustswapfreq",
        "600",
        "--disable-slippage-probes",
        "--quiet",
    ]
    if pool_start is not None:
        cmd.extend(["--pool-start", str(pool_start)])
    if pool_end is not None:
        cmd.extend(["--pool-end", str(pool_end)])
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(out_path.read_text())


def _assert_runs_equal(legacy: dict[str, Any], compact: dict[str, Any]) -> None:
    assert len(legacy["runs"]) == len(compact["runs"])
    skip_result_keys = {"pool_exec_ms"}
    for legacy_run, compact_run in zip(legacy["runs"], compact["runs"]):
        assert legacy_run["pool_index"] == compact_run["pool_index"]
        assert legacy_run["success"] is True
        assert compact_run["success"] is True

        legacy_result = {
            k: v for k, v in legacy_run["result"].items() if k not in skip_result_keys
        }
        compact_result = {
            k: v for k, v in compact_run["result"].items() if k not in skip_result_keys
        }
        assert legacy_result == compact_result
        assert legacy_run["final_state"] == compact_run["final_state"]


def test_compact_grid_loader_matches_expanded_16x16_fee_grid(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    _candles(candles)
    axes = {
        "A": [int(a * 10_000) for a in range(1, 17)],
        "mid_fee": [int(round(a / 10_000 * 10**10)) for a in range(10, 170, 10)],
    }

    legacy = _run_harness(_expanded_config(axes), candles, tmp_path, "legacy_fee")
    compact = _run_harness(_compact_config(axes), candles, tmp_path, "compact_fee")

    _assert_runs_equal(legacy, compact)


def test_compact_grid_loader_matches_expanded_16x16_policy_donation_grid(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    _candles(candles)
    axes = {
        "policy.fee_bps": [float(x) for x in range(10, 170, 10)],
        "donation_apy": [i / 1000 for i in range(16)],
    }

    legacy = _run_harness(_expanded_config(axes), candles, tmp_path, "legacy_policy")
    compact = _run_harness(_compact_config(axes), candles, tmp_path, "compact_policy")

    _assert_runs_equal(legacy, compact)


def test_compact_grid_loader_matches_expanded_sliced_16x16_grid(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    _candles(candles)
    axes = {
        "out_fee": [int(round(a / 10_000 * 10**10)) for a in range(50, 210, 10)],
        "reserved_profit_fraction": [
            int(round((0.15 + i * 0.01) * 10**10)) for i in range(16)
        ],
    }

    legacy = _run_harness(
        _expanded_config(axes),
        candles,
        tmp_path,
        "legacy_slice",
        pool_start=17,
        pool_end=203,
    )
    compact = _run_harness(
        _compact_config(axes),
        candles,
        tmp_path,
        "compact_slice",
        pool_start=17,
        pool_end=203,
    )

    _assert_runs_equal(legacy, compact)
