import copy
import itertools
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


def _normalize_axis(name: Any, values: list[Any]) -> tuple[tuple[str, ...], list[Any]]:
    if isinstance(name, tuple):
        return tuple(str(part) for part in name), values
    return (str(name),), values


def _axis_meta(names: tuple[str, ...], values: list[Any]) -> dict[str, Any]:
    if len(names) == 1:
        return {"name": names[0], "values": values}
    return {"names": list(names), "values": values}


def _set_axis_value(
    pool: dict[str, Any],
    costs: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    if name.startswith("costs."):
        costs[name.removeprefix("costs.")] = value
    else:
        _set_grid_value(pool, name, value)


def _expanded_config(axes: dict[Any, list[Any]], *, fee_equalize: bool = False) -> dict[str, Any]:
    base_pool = _base_pool()
    costs = _costs()
    axis_items = [_normalize_axis(name, values) for name, values in axes.items()]
    grid_meta = {
        f"x{i}": _axis_meta(names, values)
        for i, (names, values) in enumerate(axis_items, 1)
    }
    pools = []
    for coords in itertools.product(*(values for _names, values in axis_items)):
        pool = copy.deepcopy(base_pool)
        costs_for_pool = dict(costs)
        tag_parts = []
        touches_fee = False
        for (names, _values), value in zip(axis_items, coords):
            if len(names) == 1:
                name = names[0]
                _set_axis_value(pool, costs_for_pool, name, value)
                tag_parts.append(f"{name}_{value}")
                touches_fee = touches_fee or name in {"mid_fee", "out_fee"}
                continue
            for name, item in zip(names, value):
                _set_axis_value(pool, costs_for_pool, name, item)
                tag_parts.append(f"{name}_{item}")
                touches_fee = touches_fee or name in {"mid_fee", "out_fee"}
        if fee_equalize and touches_fee:
            pool["out_fee"] = pool["mid_fee"]
        elif touches_fee and int(pool["out_fee"]) < int(pool["mid_fee"]):
            pool["out_fee"] = pool["mid_fee"]
        pools.append(
            {
                "tag": "__".join(tag_parts),
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


def _compact_config(axes: dict[Any, list[Any]], *, fee_equalize: bool = False) -> dict[str, Any]:
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
    pool_ranges: list[tuple[int, int]] | None = None,
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
        "3600",
        "--disable-slippage-probes",
        "--quiet",
    ]
    if pool_start is not None:
        cmd.extend(["--pool-start", str(pool_start)])
    if pool_end is not None:
        cmd.extend(["--pool-end", str(pool_end)])
    if pool_ranges is not None:
        ranges_path = tmp_path / f"{name}_ranges.txt"
        ranges_path.write_text("".join(f"{start} {end}\n" for start, end in pool_ranges))
        cmd.extend(["--pool-ranges", str(ranges_path)])
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


def _assert_runs_equal_by_pool_index(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    expected_by_index = {run["pool_index"]: run for run in expected["runs"]}
    actual_by_index = {run["pool_index"]: run for run in actual["runs"]}
    assert expected_by_index.keys() == actual_by_index.keys()

    skip_result_keys = {"pool_exec_ms"}
    for pool_index, expected_run in expected_by_index.items():
        actual_run = actual_by_index[pool_index]
        assert expected_run["success"] is True
        assert actual_run["success"] is True
        assert {
            k: v for k, v in expected_run["result"].items() if k not in skip_result_keys
        } == {
            k: v for k, v in actual_run["result"].items() if k not in skip_result_keys
        }
        assert expected_run["final_state"] == actual_run["final_state"]


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


def test_compact_grid_loader_pool_ranges_preserve_global_indices(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    _candles(candles)
    axes = {
        "A": [int(a * 10_000) for a in range(1, 5)],
        "mid_fee": [int(round(a / 10_000 * 10**10)) for a in range(10, 50, 10)],
    }
    ranges = [(0, 3), (6, 8), (13, 16)]

    full = _run_harness(_expanded_config(axes), candles, tmp_path, "expanded_ranges_full")
    expected_indices = {idx for start, end in ranges for idx in range(start, end)}
    expected = {
        "runs": [
            run for run in full["runs"] if int(run["pool_index"]) in expected_indices
        ]
    }
    ranged = _run_harness(
        _compact_config(axes),
        candles,
        tmp_path,
        "compact_ranges",
        pool_ranges=ranges,
    )

    assert [run["pool_index"] for run in ranged["runs"]] == [0, 1, 2, 6, 7, 13, 14, 15]
    _assert_runs_equal_by_pool_index(expected, ranged)


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


def test_compact_grid_loader_matches_expanded_coupled_axis_10x10x3(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    _candles(candles)
    coupled_pairs = [
        [int(5 * 10_000), int(round(0.15 * 10**10))],
        [int(7 * 10_000), int(round(0.20 * 10**10))],
        [int(10 * 10_000), int(round(0.25 * 10**10))],
    ]
    axes = {
        "mid_fee": [int(round(a / 10_000 * 10**10)) for a in range(30, 130, 10)],
        "fee_gamma": [int(round((0.001 + i * 0.0005) * 10**18)) for i in range(10)],
        ("A", "reserved_profit_fraction"): coupled_pairs,
    }

    expanded = _expanded_config(axes)
    compact_cfg = _compact_config(axes)
    assert expanded["meta"]["pool_count"] == 300
    assert compact_cfg["meta"]["grid"]["x3"] == {
        "names": ["A", "reserved_profit_fraction"],
        "values": coupled_pairs,
    }

    legacy = _run_harness(expanded, candles, tmp_path, "legacy_coupled")
    compact = _run_harness(compact_cfg, candles, tmp_path, "compact_coupled")

    observed_pairs = {
        (
            run["params"]["pool"]["A"],
            run["params"]["pool"]["reserved_profit_fraction"],
        )
        for run in compact["runs"]
    }
    assert observed_pairs == {
        (str(a), str(reserved_profit_fraction))
        for a, reserved_profit_fraction in coupled_pairs
    }
    _assert_runs_equal(legacy, compact)


def test_compact_grid_loader_matches_expanded_a_rpf_with_coupled_fee_surface(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    _candles(candles)
    fee_triples = [
        [
            int(round(30 / 10_000 * 10**10)),
            int(round(101 / 10_000 * 10**10)),
            int(round(1e-5 * 10**18)),
        ],
        [
            int(round(90 / 10_000 * 10**10)),
            int(round(150 / 10_000 * 10**10)),
            int(round(1e-3 * 10**18)),
        ],
        [
            int(round(150 / 10_000 * 10**10)),
            int(round(250 / 10_000 * 10**10)),
            int(round(1e-1 * 10**18)),
        ],
    ]
    axes = {
        "A": [int(a * 10_000) for a in range(2, 12)],
        "reserved_profit_fraction": [
            int(round((0.10 + i * 0.03) * 10**10)) for i in range(10)
        ],
        ("mid_fee", "out_fee", "fee_gamma"): fee_triples,
    }

    expanded = _expanded_config(axes)
    compact_cfg = _compact_config(axes)
    assert expanded["meta"]["pool_count"] == 300
    assert compact_cfg["meta"]["grid"]["x3"] == {
        "names": ["mid_fee", "out_fee", "fee_gamma"],
        "values": fee_triples,
    }

    legacy = _run_harness(expanded, candles, tmp_path, "legacy_coupled_fees")
    compact = _run_harness(compact_cfg, candles, tmp_path, "compact_coupled_fees")

    observed_fee_triples = {
        (
            run["params"]["pool"]["mid_fee"],
            run["params"]["pool"]["out_fee"],
            run["params"]["pool"]["fee_gamma"],
        )
        for run in compact["runs"]
    }
    assert observed_fee_triples == {
        (str(mid), str(out), str(fee_gamma))
        for mid, out, fee_gamma in fee_triples
    }
    _assert_runs_equal(legacy, compact)
