import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from vyper_pool.vyper_pool_runner import run_vyper_pool
from benchmark_pool.generate_data import generate_action_sequences

CPP_BUILD = ROOT / "cpp_modular" / "build"
CPP_HARNESS = CPP_BUILD / "benchmark_harness_i"

STATE_FIELDS = (
    "balances",
    "admin_balances",
    "xp",
    "D",
    "virtual_price",
    "xcp_profit",
    "lp_xcp_profit",
    "price_scale",
    "price_oracle",
    "last_prices",
    "totalSupply",
    "donation_shares",
    "donation_shares_unlocked",
    "donation_protection_expiry_ts",
    "last_donation_release_ts",
)


def _build_cpp_harness() -> None:
    subprocess.run(
        [
            "cmake",
            "--build",
            str(CPP_BUILD),
            "--target",
            "benchmark_harness_i",
        ],
        cwd=ROOT,
        check=True,
    )


def _pool(name: str, policy_kind: str) -> dict:
    return {
        "name": name,
        "A": "400000",
        "gamma": "145000000000000",
        "mid_fee": "26000000",
        "out_fee": "45000000",
        "fee_gamma": "230000000000000",
        "adjustment_step_min": "1000000000000",
        "adjustment_step_max": "100000000000000000",
        "ma_time": "866",
        "reserved_profit_fraction": "5000000000",
        "admin_fee": "5000000000",
        "initial_price": "1000000000000000000",
        "policy": {"kind": policy_kind},
        "initial_liquidity": [
            "10000000000000000000000",
            "10000000000000000000000",
        ],
    }


def _sequence(pool_configs: list[dict]) -> dict:
    random.seed(7)
    sequences = generate_action_sequences(
        trades_per_sequence=64,
        start_ts=1_700_000_000,
        pool_configs=pool_configs,
    )
    sequences[0]["name"] = "fxswap_ext_fee_generated_core_actions"
    return {"sequences": sequences}


def _assert_sequence_mix(sequence: dict) -> None:
    actions = sequence["sequences"][0]["actions"]
    counts = Counter(action["type"] for action in actions)
    donations = sum(
        1
        for action in actions
        if action["type"] == "add_liquidity" and action.get("donation")
    )
    assert counts["time_travel"] > 0
    assert counts["exchange"] > 0
    assert counts["add_liquidity"] > donations
    assert counts["remove_liquidity"] > 0
    assert donations > 0


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _run_cpp(pools_path: Path, sequences_path: Path, output_path: Path) -> dict:
    subprocess.run(
        [str(CPP_HARNESS), str(pools_path), str(sequences_path), str(output_path)],
        cwd=ROOT,
        check=True,
    )
    return json.loads(output_path.read_text())


def _states_by_pool(result: dict) -> dict[str, list[dict]]:
    out = {}
    for item in result["results"]:
        assert item["result"]["success"], item
        out[item["pool_config"]] = item["result"]["states"]
    return out


@pytest.mark.parametrize(
    ("pool_patch", "expected_error"),
    [
        ({"fee_model_name": "legacy"}, "legacy pool config fields are not supported"),
        ({"policy": {"kind": 7}}, "policy kind must be a string"),
        (
            {"policy": {"kind": "fixed_fee", "fee_bps": []}},
            "policy fee_bps must be a string or number",
        ),
    ],
)
def test_cpp_harness_rejects_stale_or_malformed_policy_config(
    tmp_path: Path,
    pool_patch: dict,
    expected_error: str,
) -> None:
    _build_cpp_harness()

    pool = _pool("invalid_config", "none")
    pool.update(pool_patch)
    pools_path = tmp_path / "pools.json"
    sequences_path = tmp_path / "sequences.json"
    cpp_output = tmp_path / "cpp.json"

    _write_json(pools_path, {"pools": [pool]})
    _write_json(sequences_path, _sequence([pool]))

    proc = subprocess.run(
        [str(CPP_HARNESS), str(pools_path), str(sequences_path), str(cpp_output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    result = json.loads(cpp_output.read_text())["results"][0]["result"]
    assert not result["success"]
    assert expected_error in result["error"]


def _field_subset(state: dict | None, fields: set[str]) -> dict | None:
    if state is None:
        return None
    return {field: state.get(field) for field in sorted(fields)}


def _action_at(actions: list[dict], action_idx: int) -> dict:
    if action_idx < len(actions):
        return actions[action_idx]
    return {
        "error": "no action for state index",
        "n_actions": len(actions),
    }


def _assert_state_equal(
    pool_name: str,
    action_idx: int,
    action: dict,
    cpp: dict,
    boa: dict,
    prev_cpp: dict | None,
    prev_boa: dict | None,
) -> None:
    mismatches = {}
    for field in STATE_FIELDS:
        if cpp.get(field) != boa.get(field):
            mismatches[field] = {"cpp": cpp.get(field), "boa": boa.get(field)}
    mismatch_fields = set(mismatches)
    assert not mismatches, {
        "pool": pool_name,
        "action_idx": action_idx,
        "action": action,
        "previous_matching_fields": {
            "cpp": _field_subset(prev_cpp, mismatch_fields),
            "boa": _field_subset(prev_boa, mismatch_fields),
        },
        "mismatches": mismatches,
    }


def test_boa_cpp_uint256_parity_for_fxswap_ext_fee_policy_modes(tmp_path: Path) -> None:
    _build_cpp_harness()

    pool_configs = [
        _pool("policy_none", "none"),
        _pool("policy_twocrypto", "twocrypto_policy"),
        _pool("policy_zero_stub", "zero_stub"),
        _pool("policy_oracle_x2", "oracle_x2"),
    ]
    pools = {"pools": pool_configs}
    sequence = _sequence(pool_configs)
    _assert_sequence_mix(sequence)
    pools_path = tmp_path / "pools.json"
    sequences_path = tmp_path / "sequences.json"
    cpp_output = tmp_path / "cpp.json"
    boa_output = tmp_path / "boa.json"

    _write_json(pools_path, pools)
    _write_json(sequences_path, sequence)

    cpp_result = _run_cpp(pools_path, sequences_path, cpp_output)
    boa_result = run_vyper_pool(str(pools_path), str(sequences_path), str(boa_output))

    cpp_states = _states_by_pool(cpp_result)
    boa_states = _states_by_pool(boa_result)

    assert set(cpp_states) == set(boa_states)
    actions = sequence["sequences"][0]["actions"]
    for pool_name in sorted(cpp_states):
        assert len(cpp_states[pool_name]) == len(boa_states[pool_name])
        for action_idx, (cpp, boa) in enumerate(
            zip(cpp_states[pool_name], boa_states[pool_name])
        ):
            _assert_state_equal(
                pool_name,
                action_idx,
                _action_at(actions, action_idx),
                cpp,
                boa,
                cpp_states[pool_name][action_idx - 1] if action_idx > 0 else None,
                boa_states[pool_name][action_idx - 1] if action_idx > 0 else None,
            )
