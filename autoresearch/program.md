# Dynamic Fee Autoresearch

This repo supports an autonomous research loop for AMM fee models.

Your job is to improve LP outcome on a fixed benchmark by changing the fee law, not by changing the benchmark.

## Big Picture

This repo is a backtesting engine for a TwoCrypto AMM pool against a centralized exchange.

The active benchmark replays a fixed BTC candlestick / event file and simulates:

- one fixed templated AMM pool
- an outside centralized exchange with effectively infinite liquidity at the event price
- arbitrageurs who may trade between the pool and the CEX at every event

The goal is not to design the most general or deployable fee model. The goal, for this phase, is to find the fee law that squeezes the best LP outcome from this exact replay and this exact pool template.

This simulator is numerically gated, highly nonlinear, and strongly path-dependent. Stable-swap curvature, boosted LP profit / virtual price logic, donations, `allowed_extra_profit`, and pool recentring / liquidity rebalancing all interact. Small fee-law changes can materially change trade flow, rebalance timing, donation timing, and long-run APR.

Treat the benchmark as a difficult nonlinear control problem, not as a smooth optimizer. Intuitive fee ideas often fail, and tiny changes can shift the whole trajectory.

## Goal

Maximize `apy_net` for the fixed benchmark driven by:

- `python/arb_sim/nevergrad_fee_runner.py`
- `cpp_modular/include/pools/twocrypto_fx/fee_model.hpp`

Primary decision metric:
- `best_seen.loss` from the runner result JSON

Secondary human-readable metrics:
- `apy_net`
- `avg_rel_price_diff`
- `max_rel_price_diff`
- `trades`
- `tw_real_slippage_5pct`

For now, deliberate overfitting to this one BTC replay is allowed and intended. Generalization, deployability, and out-of-sample robustness are secondary in this phase.

Prefer simpler fee laws when performance is similar.

## Files In Scope

You may edit only:

- `cpp_modular/include/pools/twocrypto_fx/fee_model.hpp`
- the agent-editable search-spec block in `python/arb_sim/nevergrad_fee_runner.py`
- `autoresearch/observations.md`

Do not edit:

- the benchmark pool template in `python/arb_sim/nevergrad_fee_runner.py`
- objective / constraint defaults in `python/arb_sim/nevergrad_fee_runner.py`
- evaluator / event loop / arbitrageur / pool core
- candle data, template pools, or comparison methodology

`autoresearch/observations.md` is a writable research notebook, not a benchmark input. Use it to accumulate qualitative observations that help future iterations.

## Benchmark Contract

The benchmark is fixed unless a human changes it.

Important rules:

- Always run the canonical runner:
  - `uv run python python/arb_sim/nevergrad_fee_runner.py ...`
- Do not pass `--skip-rebuild` for normal research runs.
- Do not change the embedded benchmark pool or costs.
- Keep `FEE_PARAM_COUNT` and `FEE_STATE_COUNT` fixed at 20.

When you change fee-model semantics:

- update `FEE_MODEL_NAME`
- update `FEE_PARAM_LABELS`
- keep the Python search-spec aligned with the active slots used by the model

If the active model only uses a few slots, only optimize those slots.

## What You Can Build

Inside `fee_model.hpp`, you may:

- implement stateful fee logic using `fee_state`
- use current pool state from `FeeInputs`
- use timestamps and `ma_time` via `FeeStateInputs`
- build signals such as:
  - imbalance
  - oracle gap
  - spot momentum
  - volatility trackers
  - calm/stress regimes
  - directional trade-aware fee ramps

Good candidate families:

- constant fee baseline
- imbalance-sensitive fee
- volatility-sensitive fee
- oracle-gap fee
- calm-market discounts
- interaction models such as imbalance × volatility

Rich and stateful fee laws are encouraged if they improve this replay. Use `fee_state` aggressively when it helps capture more LP outcome on this exact path.

## Experiment Loop

### Baseline

First, establish the current baseline on the fixed benchmark.

Use:

```bash
uv run python python/arb_sim/nevergrad_fee_runner.py \
  --budget 2500 \
  --workers 8 \
  --n-candles 0 \
  --result autoresearch/fee_autoresearch_baseline.json
```

Record:

- git commit
- `FEE_MODEL_NAME`
- active search slots
- `best_seen.loss`
- `apy_net`
- `avg_rel_price_diff`

### Iteration

Loop autonomously:

1. Read the current best result and current fee model.
2. Propose one concrete fee-model idea.
3. Implement it in `fee_model.hpp`.
4. If the model uses different knobs, update only the agent-editable search-spec block in `nevergrad_fee_runner.py`.
5. Run the full benchmark for that fee family:

```bash
uv run python python/arb_sim/nevergrad_fee_runner.py \
  --budget 2500 \
  --workers 8 \
  --n-candles 0 \
  --result autoresearch/fee_autoresearch_candidate.json
```

6. If the result is not better than the current best, revert the change and try a different idea.
7. If the result is better, keep it as the new best.
8. Append useful qualitative observations to `autoresearch/observations.md`.
9. Continue until interrupted.

## Keep / Revert Rule

Keep a change only if:

- `best_seen.loss` improves on the current best, and
- the result does not clearly degrade market quality metrics

Overfitting this benchmark is allowed. The bar is not “would this generalize?” but “does this improve the fixed replay without obviously breaking the pool.”

Reject changes that “win” by making the pool unusable.

If two fee laws are similar in score, keep the simpler one.

## Logging

Maintain an untracked append-only log at:

- `autoresearch/fee_autoresearch_runs.jsonl`

Each line should include:

- timestamp
- git commit
- `FEE_MODEL_NAME`
- edited files
- active search slots
- benchmark command
- result path
- `best_seen.loss`
- `apy_net`
- `avg_rel_price_diff`
- decision: `keep` or `revert`
- one short sentence describing the idea

Also maintain a qualitative notebook at:

- `autoresearch/observations.md`

Use it for short durable observations such as:

- arbitrageur behavior
- whether trades cluster around specific gaps or imbalance regimes
- pool recentring / tweak-price behavior
- donation / rebalance interactions
- why a fee family failed even if the idea looked plausible
- repeated anti-patterns or promising mechanisms

Keep entries concise and cumulative. This file is for future agent context, not for benchmark scoring.

## Safety Rules

- Do not rewrite the benchmark to get a better score.
- Do not expand the editable surface beyond the two allowed files.
- Do not silently change the objective or benchmark pool.
- Do not optimize unused fee slots.
- Do optimize aggressively for this one fixed benchmark, but only by changing the fee law.
- Do not treat `autoresearch/observations.md` as benchmark truth; it is only a qualitative notebook.
- Do not stop after one experiment unless explicitly interrupted.

The benchmark is the judge. The fee law is the thing to improve.
