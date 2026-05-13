#!/usr/bin/env python3
"""
Generate benchmark pool configs and one mixed action sequence.

Time travel actions use absolute timestamps ("timestamp").
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

Action = dict[str, Any]
PoolConfig = dict[str, Any]

WAD = 10**18
MILLI_WAD = 10**15
FEE_SCALE = 10**10
BPS = 10_000

DEFAULT_START_TS = 1_700_000_000
MINIMUM_LIQUIDITY = 10**4

INITIAL_POOL_TOKENS = (3_000, 10_000)
INITIAL_LIQUIDITY_JITTER_BPS = (9_500, 10_500)

SWAP_INITIAL_BPS = (10, 20)  # 0.10% to 0.20% of initial input balance
SWAP_OUTPUT_CAP_BPS = 500  # Avoid draining scarce output balances in long runs.
SWAP_DIRECTION_BATCH = 10

REGULAR_ADD_MILLI = (50, 5_000)  # 0.05 to 5 tokens
DONATION_MILLI = (1, 50)  # 0.001 to 0.05 tokens
DONATION_CAP_RATIO_ESTIMATE = 8 * 10**16  # 8%, below the contract's 10% cap.
DONATION_MAX_TRIES = 16

LP_RESERVE = WAD
MIN_REMOVE_LP = MILLI_WAD
REMOVE_BPS = (1, 25)  # 0.01% to 0.25% of estimated removable LP.

ADJUSTMENT_STEP_MIN = 10**12  # 0.000001 * 1e18
ADJUSTMENT_STEP_MAX_BPS = (100, 2_000)  # 1% to 20%
RESERVED_PROFIT_FRACTION_BPS = (0, BPS)
ADMIN_FEE_BPS = (0, int(0.9 * BPS))

TIME_TRAVEL_EVERY_STEPS = 4
TIME_TRAVEL_SECONDS = (60, 3_600)

FORCED_ACTION_KINDS = ("exchange", "add_liquidity", "donation", "remove_liquidity")
RANDOM_ACTION_MIX = (
    (0.45, "exchange"),
    (0.68, "add_liquidity"),
    (0.85, "donation"),
    (1.00, "remove_liquidity"),
)
POLICY_KINDS = ("none", "twocrypto_policy", "zero_stub", "oracle_x2")


@dataclass
class SequenceState:
    """Conservative state used to keep generated actions executable."""

    owned_lp: int
    total_lp: int
    balances: list[list[int]]
    initial_balances: list[list[int]]
    donation_lp: int = 0
    swap_count: int = 0
    first_swap_direction: int = 0

    @classmethod
    def from_pools(cls, pool_configs: list[PoolConfig] | None) -> "SequenceState":
        if pool_configs:
            balances = [
                [int(pool["initial_liquidity"][0]), int(pool["initial_liquidity"][1])]
                for pool in pool_configs
            ]
        else:
            balances = [[1_000 * WAD, 1_000 * WAD]]

        min_initial_balance = min(min(row) for row in balances)
        owned_lp = max(LP_RESERVE, (min_initial_balance - MINIMUM_LIQUIDITY) // 2)
        return cls(
            owned_lp=owned_lp,
            total_lp=owned_lp,
            balances=[row.copy() for row in balances],
            initial_balances=[row.copy() for row in balances],
            first_swap_direction=random.randint(0, 1),
        )

    def record_deposit(self, amounts: tuple[int, int]) -> None:
        for balances in self.balances:
            balances[0] += amounts[0]
            balances[1] += amounts[1]

    def record_regular_add(self, amounts: tuple[int, int]) -> None:
        minted_estimate = max(1, min(amounts) // 2)
        self.record_deposit(amounts)
        self.owned_lp += minted_estimate
        self.total_lp += minted_estimate

    def can_accept_donation(self, minted_estimate: int) -> bool:
        new_total = self.total_lp + minted_estimate
        new_donation = self.donation_lp + minted_estimate
        return (
            new_total > 0
            and new_donation * WAD // new_total <= DONATION_CAP_RATIO_ESTIMATE
        )

    def record_donation(self, amounts: tuple[int, int], minted_estimate: int) -> None:
        self.record_deposit(amounts)
        self.donation_lp += minted_estimate
        self.total_lp += minted_estimate

    def remove_budget(self) -> int:
        return max(0, self.owned_lp - LP_RESERVE)

    def record_remove(self, amount: int) -> None:
        old_total_lp = max(1, self.total_lp)
        for balances in self.balances:
            balances[0] = max(0, balances[0] - balances[0] * amount // old_total_lp)
            balances[1] = max(0, balances[1] - balances[1] * amount // old_total_lp)
        self.owned_lp -= amount
        self.total_lp = max(0, self.total_lp - amount)

    def swap_directions(self) -> list[int]:
        batch = self.swap_count // SWAP_DIRECTION_BATCH
        direction = (self.first_swap_direction + batch) % 2
        return [direction, 1 - direction]

    def swap_bounds(self, coin_in: int) -> tuple[int, int] | None:
        coin_out = 1 - coin_in
        min_initial_input = min(row[coin_in] for row in self.initial_balances)
        lower = min_initial_input * SWAP_INITIAL_BPS[0] // BPS
        upper = min_initial_input * SWAP_INITIAL_BPS[1] // BPS
        output_cap = min(
            balances[coin_out] * SWAP_OUTPUT_CAP_BPS // BPS
            for balances in self.balances
        )
        upper = min(upper, output_cap)

        lower_units = ceil_to_milli_wad(lower) // MILLI_WAD
        upper_units = upper // MILLI_WAD
        if lower_units > upper_units:
            return None
        return lower_units * MILLI_WAD, upper_units * MILLI_WAD

    def record_swap(self, coin_in: int, dx: int) -> None:
        coin_out = 1 - coin_in
        for balances in self.balances:
            dy_estimate = dx * balances[coin_out] // (balances[coin_in] + dx)
            balances[coin_in] += dx
            balances[coin_out] = max(0, balances[coin_out] - dy_estimate)
        self.swap_count += 1


def ceil_to_milli_wad(amount: int) -> int:
    return ((amount + MILLI_WAD - 1) // MILLI_WAD) * MILLI_WAD


def random_milli_wad(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1]) * MILLI_WAD


def jitter_bps(amount: int, bounds: tuple[int, int]) -> int:
    return amount * random.randint(bounds[0], bounds[1]) // BPS


def random_fee_fraction(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1]) * FEE_SCALE // BPS


def generate_pool_configs(
    num_pools: int = 3,
    policy_kind: str = "none",
) -> list[PoolConfig]:
    pools: list[PoolConfig] = []
    base_liquidity = random.randint(*INITIAL_POOL_TOKENS) * WAD

    for i in range(num_pools):
        mid_fee = random.randint(1, FEE_SCALE // 2)
        out_fee = max(mid_fee, random.randint(1, FEE_SCALE // 2))
        pools.append(
            {
                "name": f"pool_{i:02d}",
                "A": str(random.randint(2, 10_000) * 10_000),
                "gamma": str(random.randint(10**11, 10**16)),
                "mid_fee": str(mid_fee),
                "out_fee": str(out_fee),
                "fee_gamma": str(random.randint(10**10, 10**18)),
                "adjustment_step_min": str(ADJUSTMENT_STEP_MIN),
                "adjustment_step_max": str(
                    random.randint(*ADJUSTMENT_STEP_MAX_BPS) * WAD // BPS
                ),
                "reserved_profit_fraction": str(
                    random_fee_fraction(RESERVED_PROFIT_FRACTION_BPS)
                ),
                "admin_fee": str(random_fee_fraction(ADMIN_FEE_BPS)),
                "policy": {"kind": policy_kind},
                "ma_time": str(1 + int(random.randint(60, 3_600) / math.log(2))),
                "initial_price": str(WAD),
                "initial_liquidity": [
                    str(jitter_bps(base_liquidity, INITIAL_LIQUIDITY_JITTER_BPS)),
                    str(jitter_bps(base_liquidity, INITIAL_LIQUIDITY_JITTER_BPS)),
                ],
            }
        )
    return pools


def exchange_action(state: SequenceState) -> Action | None:
    for coin_in in state.swap_directions():
        bounds = state.swap_bounds(coin_in)
        if bounds is None:
            continue

        dx = random_milli_wad((bounds[0] // MILLI_WAD, bounds[1] // MILLI_WAD))
        state.record_swap(coin_in, dx)
        return {
            "type": "exchange",
            "i": coin_in,
            "j": 1 - coin_in,
            "dx": str(dx),
        }
    return None


def regular_add_liquidity_action(state: SequenceState) -> Action:
    amounts = (
        random_milli_wad(REGULAR_ADD_MILLI),
        random_milli_wad(REGULAR_ADD_MILLI),
    )
    state.record_regular_add(amounts)
    return {
        "type": "add_liquidity",
        "amounts": [str(amounts[0]), str(amounts[1])],
        "donation": False,
    }


def donation_action(state: SequenceState) -> Action | None:
    for _ in range(DONATION_MAX_TRIES):
        amounts = (
            random_milli_wad(DONATION_MILLI),
            random_milli_wad(DONATION_MILLI),
        )
        minted_estimate = amounts[0] + amounts[1]
        if state.can_accept_donation(minted_estimate):
            state.record_donation(amounts, minted_estimate)
            return {
                "type": "add_liquidity",
                "amounts": [str(amounts[0]), str(amounts[1])],
                "donation": True,
            }
    return None


def remove_liquidity_action(state: SequenceState) -> Action | None:
    budget = state.remove_budget()
    if budget < MIN_REMOVE_LP:
        return None

    remove_bps = random.randint(*REMOVE_BPS)
    amount = max(MIN_REMOVE_LP, budget * remove_bps // BPS)
    amount = min(amount, budget)
    state.record_remove(amount)
    return {
        "type": "remove_liquidity",
        "amount": str(amount),
        "min_amounts": ["0", "0"],
    }


def action_for_kind(kind: str, state: SequenceState) -> Action | None:
    if kind == "exchange":
        return exchange_action(state)
    if kind == "add_liquidity":
        return regular_add_liquidity_action(state)
    if kind == "donation":
        return donation_action(state)
    if kind == "remove_liquidity":
        return remove_liquidity_action(state)
    raise ValueError(f"unknown action kind: {kind}")


def random_action_kind() -> str:
    roll = random.random()
    for cutoff, kind in RANDOM_ACTION_MIX:
        if roll < cutoff:
            return kind
    return RANDOM_ACTION_MIX[-1][1]


def next_pool_action(step: int, state: SequenceState) -> Action:
    if step < len(FORCED_ACTION_KINDS):
        kind = FORCED_ACTION_KINDS[step]
    else:
        kind = random_action_kind()
    return action_for_kind(kind, state) or regular_add_liquidity_action(state)


def generate_action_sequences(
    trades_per_sequence: int = 20,
    start_ts: int = DEFAULT_START_TS,
    pool_configs: list[PoolConfig] | None = None,
) -> list[dict[str, Any]]:
    actions: list[Action] = []
    timestamp = start_ts
    state = SequenceState.from_pools(pool_configs)

    for step in range(trades_per_sequence):
        if step % TIME_TRAVEL_EVERY_STEPS == 0:
            timestamp += random.randint(*TIME_TRAVEL_SECONDS)
            actions.append({"type": "time_travel", "timestamp": timestamp})
        actions.append(next_pool_action(step, state))

    return [{"name": "default", "start_timestamp": start_ts, "actions": actions}]


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def action_summary(actions: list[Action]) -> str:
    counts = Counter(action["type"] for action in actions)
    donations = sum(
        1
        for action in actions
        if action["type"] == "add_liquidity" and action.get("donation")
    )
    regular_adds = counts["add_liquidity"] - donations
    return (
        f"time_travel={counts['time_travel']}, "
        f"exchange={counts['exchange']}, "
        f"add_liquidity={regular_adds}, "
        f"donation={donations}, "
        f"remove_liquidity={counts['remove_liquidity']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TwoCrypto benchmark data (single sequence)"
    )
    parser.add_argument("--pools", type=int, default=3, help="Number of pools")
    parser.add_argument(
        "--trades",
        "--actions",
        dest="trades",
        type=int,
        default=20,
        help="Pool action steps; time_travel actions are inserted in addition",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--start-ts",
        type=int,
        default=DEFAULT_START_TS,
        help="Sequence start timestamp",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: python/benchmark_pool/data)",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_KINDS,
        default="none",
        help="External policy mode written into generated pool configs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    data_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    print("Generating pool configurations...")
    pools = generate_pool_configs(num_pools=args.pools, policy_kind=args.policy)
    write_json(os.path.join(data_dir, "pools.json"), {"pools": pools})
    print(f"✓ Generated {len(pools)} pool configurations")

    print("Generating action sequences...")
    sequences = generate_action_sequences(
        trades_per_sequence=args.trades,
        start_ts=args.start_ts,
        pool_configs=pools,
    )
    actions = sequences[0]["actions"]
    write_json(os.path.join(data_dir, "sequences.json"), {"sequences": sequences})
    print(f"✓ Generated 1 sequence with {len(actions)} actions")
    print(f"  mix: {action_summary(actions)}")

    print(f"\n✓ Total tests: {len(pools)}")
    print(f"✓ Data saved to {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
