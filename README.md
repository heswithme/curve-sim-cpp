# TwoCrypto C++ Modular

High-performance C++ implementation of Curve's TwoCrypto AMM pool for simulation and parameter grid search.

## Two Main Use Cases

### 1. C++ Pool Parity Validation

Verify C++ pool math matches Vyper reference. Generate test data, run C++ variants (double/float), compare against Vyper.

```bash
# Check math parity
uv run python/benchmark_math/main.py

# Generate benchmark data
uv run python/benchmark_pool/generate_data.py --pools 3 --trades 20 --seed 42

# Run C++ variants and Vyper, compare results
uv run python/benchmark_pool/run_full_benchmark.py --n-cpp 8 --n-py 1

```

### 2. Arbitrage Grid Search

Run parameter sweeps over historical candle data to evaluate pool configurations. Optionally replay historical CoWSwap trades.

**Get data first:**
- Candles: https://github.com/heswithme/FX-1-Minute-Data (put in `python/arb_sim/trade_data/`)
- CoWSwap trades (optional): https://github.com/heswithme/cowswap-trades-fetch

**Run grid search:**

```bash
# 1. Generate pool parameter grid
uv run python/arb_sim/generate_pools_generic.py

# 2. Run simulation
uv run python/arb_sim/arb_sim.py --real double --dustswapfreq 600 -n 10

# 3. Plot results
uv run python/arb_sim/plot_heatmap.py --metrics apy,apy_net,apy_geom_mean,vp,tw_avg_pool_fee,n_rebalances,trades,total_notional_coin0,avg_rel_price_diff,tw_slippage,tw_liq_density --ncol 5
```

**With CoWSwap replay (add `--cow`):**

```bash
uv run python/arb_sim/arb_sim.py --real double --dustswapfreq 600 -n 10 --cow
```

**One-liner:**

```bash
uv run python/arb_sim/generate_pools_generic.py && uv run python/arb_sim/arb_sim.py --real double --dustswapfreq 600 -n 10 && uv run python/arb_sim/plot_heatmap.py --metrics apy,apy_net,apy_geom_mean,vp,tw_avg_pool_fee,n_rebalances,trades,total_notional_coin0,avg_rel_price_diff,tw_slippage,tw_liq_density --ncol 5
```

**Cluster sweep (orchestration + analysis):**

```bash
# 1. Generate N-dimensional pool grid
uv run python/arb_sim/generate_pools_nd.py

# 2. Run cluster sweep (streams one blade)
uv run python/arb_sim/cluster_orchestration/orchestrate.py --pools python/arb_sim/run_data/pool_config.json --stream-blade blade-a5

# 3. Plot optimized N-d heatmaps from merged results
uv run python/arb_sim/plot_heatmap_nd_opt.py --metrics apy_masked,apy,apy_net,vp,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,tw_real_slippage_5pct --ncol 4 --arb python/arb_sim/cluster_orchestration/results/cluster_sweep_latest.json --pricethr 50

# 4. Enumerate local maxima for a metric
uv run python/arb_sim/find_local_maxima_orjson.py --arb python/arb_sim/cluster_orchestration/results/cluster_sweep_latest.json --metric apy_mask_3 --local --top 100 --enumerate

# 5. Rank grid points by multiple metrics (rank aggregation)
uv run --with orjson python/arb_sim/find_ranked_maxima.py \
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

Both directories are gitignored.
