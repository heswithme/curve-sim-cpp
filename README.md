# TwoCrypto C++ Modular

High-performance C++ implementation of Curve's TwoCrypto AMM pool for simulation and parameter grid search.

Python commands below are shown from the repository root. If you are already in
`python/`, drop `--project python` and the leading `python/` path prefix.

## Two Main Use Cases

### 1. C++ Pool Parity Validation

Verify C++ pool math matches Vyper reference. Generate test data, run C++ variants (double/float), compare against Vyper.

```bash
# Check math parity
uv --project python run python python/benchmark_math/main.py

# Generate benchmark data
uv --project python run python python/benchmark_pool/generate_data.py --pools 3 --trades 20 --seed 42

# Run C++ variants and Vyper, compare results
uv --project python run python python/benchmark_pool/run_full_benchmark.py --n-cpp 8 --n-py 1

```

### 2. Arbitrage Grid Search

Run parameter sweeps over historical candle data to evaluate pool configurations. Optionally replay historical CoWSwap trades.

**Get data first:**
- Candles: https://github.com/heswithme/FX-1-Minute-Data (put in `python/arb_sim/trade_data/`)
- CoWSwap trades (optional): https://github.com/heswithme/cowswap-trades-fetch

**Run grid search:**

```bash
# 1. Generate pool parameter grid
uv --project python run python python/arb_sim/generate_pools_nd.py

# 2. Run simulation
uv --project python run python python/arb_sim/arb_sim.py --real double --dustswapfreq 600 -n 14

# 3. Plot results
uv --project python run python python/arb_sim/plot_heatmap.py --metrics apy,apy_net,vp,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,avg_imbalance,tw_real_slippage_5pct --ncol 5
```

**With CoWSwap replay (add `--cow`):**

```bash
uv --project python run python python/arb_sim/arb_sim.py --real double --dustswapfreq 600 -n 14 --cow
```

**One-liner:**

```bash
uv --project python run python python/arb_sim/generate_pools_nd.py && uv --project python run python python/arb_sim/arb_sim.py --real double --dustswapfreq 600 -n 14 && uv --project python run python python/arb_sim/plot_heatmap_nd_opt.py --metrics apy_masked,apy,apy_net,vp,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,tw_real_slippage_5pct --ncol 4 --arb python/arb_sim/run_data/arb_run_1.json
```

**BTC `A x policy fee` sweep:**

Edit the active setup block in `python/arb_sim/generate_pools_nd.py`. The grid is
intentionally explicit, for example:

```python
N_DENSE = 32
DATAFILE = SCRIPT_DIR / "trade_data" / "btcusd" / "btcusd-2023-2026.json"
COWSWAP_FILE = SCRIPT_DIR / "trade_data" / "btcusd" / "btcusd-cow.csv"
START_TIME = None
LAST_YEARS = 2.0

GRID = {
    "A": [int(a * 10_000) for a in np.linspace(1, 10, N_DENSE)],
    "policy.fee_bps": np.linspace(10, 200, N_DENSE),
    # "donation_apy": np.linspace(0.0, 0.2, N_DENSE),
}
```

Then run:

```bash
uv --project python run python python/arb_sim/generate_pools_nd.py
uv --project python run python python/arb_sim/arb_sim.py \
  --real double --dustswapfreq 600 -n 14 --disable-slippage-probes --quiet-harness --skip-build
uv --project python run python python/arb_sim/plot_heatmap_nd_opt.py --metrics apy_masked,apy,apy_net,vp,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,tw_real_slippage_5pct --ncol 4 --arb python/arb_sim/run_data/arb_run_1.json
```

For a 16x16x16 or 24x24x24 BTC sweep over `A`, policy fee, and donation APY,
set `N_DENSE` to `16` or `24` and uncomment the `donation_apy` axis.

For lower-overhead timing/debug runs, the raw C++ harness output is also
plottable now; `plot_heatmap_nd_opt.py` infers varying axes from `params.pool`,
including nested axes such as `policy.fee_bps`. Raw output metadata also records
the run knobs (`dustswapfreq`, probe mode, swap bounds, thread count) and total
trade count:

```bash
cpp_modular/build/arb_harness \
  python/arb_sim/run_data/pool_config.json \
  python/arb_sim/trade_data/btcusd/btcusd-2023-2026.json \
  /tmp/arb_btc_policy_24x24x24_donation_raw.json \
  --threads 14 --start-time 1709638320 --dustswapfreq 600 --disable-slippage-probes --quiet

uv --project python run python python/arb_sim/plot_heatmap_nd_opt.py \
  --arb /tmp/arb_btc_policy_24x24x24_donation_raw.json \
  --metrics apy_net,avg_rel_price_diff --out /tmp/heatmap_24x24x24.png

uv --project python run python python/arb_sim/find_ranked_maxima.py \
  --arb /tmp/arb_btc_policy_24x24x24_donation_raw.json \
  --desc-metrics apy_net --asc-metrics avg_rel_price_diff --top 10
```

For full BTC sweeps on the tested 10-core MacBook, `-n 10`, `-n 12`, and
`-n 14` are close enough that machine load can move the winner. The examples
use `-n 14` to match the retained large-grid artifacts; retest the thread count
after meaningful hot-path or machine-load changes.

Reference timings with the full BTC window, probes disabled, `--dustswapfreq
600`, and unchanged trade counts across compared artifacts. These use the
current generator default `adjustment_step_min = 0.000001 * 1e18`. The 32x32
AppleClang row was refreshed after removing the event-loop decision-input cache
for policy correctness; refresh the larger grids before treating their
AppleClang rows as current-source timings:

| Grid | Pools | AppleClang `exec_ms` | Homebrew LLVM `exec_ms` |
| --- | ---: | ---: | ---: |
| `32 x 32` (`A x policy.fee_bps`) | 1,024 | 12,812.927 | not refreshed |
| `16 x 16 x 16` (+ `donation_apy`) | 4,096 | 40,262.456 cached-era | 39,707.814 cached-era |
| `24 x 24 x 24` (+ `donation_apy`) | 13,824 | 132,125.467 cached-era | 130,722.803 cached-era |

The exact wrapper flow below was also refreshed with `donation_apy=0:0.2:N`,
`--dustswapfreq 600`, probes disabled, and `-n 14`: `16^3` measured
`exec_ms=40056.208` over `202,540,187` trades; `24^3` measured
`exec_ms=132064.664` over `662,912,645` trades.

The current full thread sweeps on the same wrapper configs show the local
plateau. `16^3` measured 6/8/10/12/14/16 threads at `48734.738`, `43428.027`,
`40185.514`, `39885.572`, `39705.318`, and `40096.770` `exec_ms`. `24^3`
measured 10/12/14/16 threads at `132276.003`, `132409.546`, `131975.272`, and
`132315.783` `exec_ms`. On this MacBook, 10/12/14 are close for 32x32 and
16^3; 24^3 is effectively flat through 16 threads. Normalized throughput at the
retained best points is about `5.10M` trades/sec for `16^3` and `5.02M`
trades/sec for `24^3`.
A later 16^3 best-block refresh over 10/12/14 selected 14 threads at
`39712.392ms`, consistent with the retained large-grid default.
A limited 24^3 best-block refresh over 10/14 selected 10 threads at
`132015.294ms`, and a matching 12/16 refresh measured `132177.870ms` and
`132177.187ms`, still within the same flat plateau.
A complete all-thread 24^3 artifact at
`/private/tmp/thread_sweep_btc_policy_24x24x24_donation_best_all.json`
selected 12 threads at `131940.837ms`; use this artifact when a single
top-level `best` block is convenient.

Treat these as reference artifacts, not promises. Rerun `thread_sweep.py` and a
same-artifact comparison after changing C++ hot-path code or compiler flags.
The sweep summary also reports a top-level `best` block plus
`trades_per_second` and `ms_per_million_trades`, which are the easiest fields
to compare across grid sizes when the trade count changes.
For cleaner long-run timing without worker progress logging, pass
`--quiet-harness` through the Python wrapper or `--quiet` to `arb_harness`
directly. A same-session 32x32 check matched exactly and measured quiet
`9919.960ms` versus verbose `10090.800ms`.

To remeasure local thread scaling on the same pool config and candle file:

```bash
uv --project python run python python/arb_sim/thread_sweep.py \
  --pool-config /tmp/pool_config_btc_policy_24x24x24_donation.json \
  --threads 8,10,12,14 \
  --real double --dustswapfreq 600 --disable-slippage-probes --quiet-harness --skip-build \
  --out /tmp/thread_sweep_btc_policy.json
```

To check that a performance patch did not change economics or final states,
compare two `arb_run` artifacts while ignoring timing fields:

```bash
uv --project python run python python/arb_sim/compare_arb_runs.py \
  /tmp/before.json /tmp/after.json
```

To test a custom harness without replacing `cpp_modular/build`, pass the binary
explicitly:

```bash
uv --project python run python python/arb_sim/arb_sim.py \
  --harness-exe /tmp/custom_arb_harness \
  --real double --dustswapfreq 600 -n 14 --disable-slippage-probes --skip-build
```

On the tested MacBook, a Homebrew LLVM build was exact against the default
AppleClang artifact and faster on the BTC policy grids:

```bash
cmake -S cpp_modular -B /tmp/cpp_modular_hb_llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build /tmp/cpp_modular_hb_llvm --target arb_harness -j 8

uv --project python run python python/arb_sim/arb_sim.py \
  --harness-exe /tmp/cpp_modular_hb_llvm/arb_harness \
  --real double --dustswapfreq 600 -n 14 --disable-slippage-probes --skip-build
```

The safe PGO helper trains and builds under `/tmp` by default. It is useful for
experiments, but the latest current-source PGO check was exact and slower than
the normal build on both 32x32 and 16^3, so do not use it as the default local
long-sweep path without remeasuring.

```bash
cpp_modular/scripts/pgo_build.sh \
  --pools python/arb_sim/run_data/pool_config.json \
  --candles python/arb_sim/trade_data/btcusd/btcusd-2023-2026.json \
  --start-time 1709638320 \
  --threads 14
```

Add `--out /tmp/heatmap.png` to either N-D heatmap command to save the current
slice without opening interactive windows. For detailed trajectory output, run
the simulator with `--detailed-log` and then:

```bash
uv --project python run python python/arb_sim/plot_price_scale.py python/arb_sim/run_data/detailed-output.json --out /tmp/price_scale.png
```

Add more sweep dimensions by editing `GRID` directly:

```python
GRID = {
    "A": [int(a * 10_000) for a in np.linspace(1, 10, N_DENSE)],
    "policy.fee_bps": np.linspace(10, 200, N_DENSE),
    "donation_apy": np.linspace(0.0, 0.2, N_DENSE),
    "adjustment_step_min": [int(a * 10**18) for a in np.linspace(0.000001, 0.000002, N_DENSE)],
}
```

**Cluster sweep (orchestration + analysis):**

```bash
# 1. Generate N-dimensional pool grid
uv --project python run python python/arb_sim/generate_pools_nd.py

# 2. Run cluster sweep. Remove --quiet-harness if you want streamed harness progress.
uv --project python run python python/arb_sim/cluster_orchestration/orchestrate.py \
  --pools python/arb_sim/run_data/pool_config.json \
  --dustswap-freq 600 --disable-slippage-probes --quiet-harness \
  --stream-blade blade-a5

# 3. Plot optimized N-d heatmaps from merged results
uv --project python run python python/arb_sim/plot_heatmap_nd_opt.py --metrics apy_masked,apy,apy_net,apy_xcp,apy_xcp_net,vp,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,avg_imbalance,tw_real_slippage_5pct --ncol 4 --arb python/arb_sim/cluster_orchestration/results/cluster_sweep_latest.json --pricethr 50

# 4. Enumerate local maxima for a metric
uv --project python run python python/arb_sim/find_local_maxima_orjson.py --arb python/arb_sim/cluster_orchestration/results/cluster_sweep_latest.json --metric apy_masked --local --top 100 --enumerate --pricethr 50

# 5. Rank grid points by multiple metrics (rank aggregation)
uv --project python run --with orjson python python/arb_sim/find_ranked_maxima.py \
  --arb python/arb_sim/cluster_orchestration/results/cluster_sweep_latest.json \
  --desc-metrics apy_net --asc-metrics avg_rel_price_diff,tw_real_slippage_5pct \
  --weights apy_net=1 --top 20
```

## Key CLI Flags

| Flag | Description |
|------|-------------|
| `--real double/float/longdouble` | Numeric precision |
| `-n N` | Thread count |
| `--n-candles N` | Limit candles processed |
| `--dustswapfreq S` | Idle tick frequency (seconds) |
| `--cow` | Enable CoWSwap trade replay |
| `--save-actions` | Record all trades for replay |
| `--detailed-log` | Per-candle state output |
| `--disable-slippage-probes` | Skip slippage probe sampling |
| `--quiet-harness` | Suppress C++ harness progress logs in Python/cluster wrappers |

## Requirements

- C++17 compiler, CMake, Boost (json)
- Python 3.10+, uv
- For Vyper validation: titanoboa

```bash
git submodule update --init --recursive
```

## Build

```bash
cmake -B cpp_modular/build cpp_modular -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_modular/build -j
```

## Repository Structure

```
cpp_modular/          # C++ pool implementation and harness
python/
  arb_sim/            # Grid search simulator
    trade_data/       # Candle data (btcusd/, ethusd/, etc.)
    run_data/         # Output: pool_config.json, arb_run_*.json
  benchmark_pool/     # Parity validation tools
  vyper_pool/         # Vyper runner via titanoboa
contracts/twocrypto-ng/  # Vyper reference (submodule)
```

## Outputs

Results written to:
- `python/arb_sim/run_data/` - grid search results
- `python/benchmark_pool/data/results/` - parity benchmark results

`arb_run_*.json` metadata includes the pool config path, numeric type, candle
file, pool count, requested/loaded candles, event count, thread count, probe
flags, harness binary path, C++ execution time, wall time, and total trades so
timing artifacts are self-describing.

Both directories are gitignored.
