import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from benchmark_pool.generate_data import (  # noqa: E402
    BPS,
    LP_RESERVE,
    MILLI_WAD,
    REMOVE_BPS,
    SWAP_DIRECTION_BATCH,
    WAD,
    SequenceState,
    generate_action_sequences,
    regular_add_liquidity_action,
    remove_liquidity_action,
)


def _pool(initial0: int, initial1: int) -> dict:
    return {"initial_liquidity": [str(initial0), str(initial1)]}


def test_generated_swaps_are_batched_and_below_one_percent_initial_balance() -> None:
    random.seed(123)
    pool_configs = [_pool(10_000 * WAD, 10_000 * WAD)]

    sequence = generate_action_sequences(
        trades_per_sequence=90,
        start_ts=1_700_000_000,
        pool_configs=pool_configs,
    )[0]
    exchanges = [action for action in sequence["actions"] if action["type"] == "exchange"]

    assert len(exchanges) >= 20
    first_direction = exchanges[0]["i"]
    initial_balances = [int(value) for value in pool_configs[0]["initial_liquidity"]]
    one_percent_bps = 100

    for idx, action in enumerate(exchanges):
        expected_direction = (first_direction + idx // SWAP_DIRECTION_BATCH) % 2
        coin_in = action["i"]
        assert coin_in == expected_direction
        assert int(action["dx"]) <= initial_balances[coin_in] * one_percent_bps // BPS


def test_remove_liquidity_generator_respects_owned_lp_reserve() -> None:
    random.seed(321)
    initial_balances = [[10_000 * WAD, 10_000 * WAD]]
    state = SequenceState(
        owned_lp=LP_RESERVE,
        total_lp=LP_RESERVE,
        balances=[row.copy() for row in initial_balances],
        initial_balances=[row.copy() for row in initial_balances],
    )

    assert remove_liquidity_action(state) is None

    regular_add_liquidity_action(state)
    budget_before = state.remove_budget()
    action = remove_liquidity_action(state)

    assert action is not None
    amount = int(action["amount"])
    assert MILLI_WAD <= amount <= budget_before
    assert amount <= budget_before * REMOVE_BPS[1] // BPS
    assert state.owned_lp >= LP_RESERVE
