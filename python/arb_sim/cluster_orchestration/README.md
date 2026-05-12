# Cluster Orchestration

Run TwoCrypto pool parameter sweeps on the NixOS HPC cluster.

## Quick Start

### 1. Generate pool configs (N-dimensional grid)

```bash
uv run python arb_sim/generate_pools_nd.py
```

Edit the active setup block in `arb_sim/generate_pools_nd.py` before running.
The grid is intentionally explicit; add dimensions directly in `GRID`, for
example `donation_apy`, `adjustment_step_min`, or nested policy axes such as
`policy.fee_bps`. Those axes are preserved through collection and plotting.

The cluster runner reads `meta.datafile` and `meta.start_time` from the pool
config, so remote blades use the same candle file and window as the local
`arb_sim.py` workflow.

### 2. Run sweep on cluster

Remove `--quiet-harness` when you want `--stream-blade` to show per-pool
harness progress.

```bash
uv run python arb_sim/cluster_orchestration/orchestrate.py \
  --pools arb_sim/run_data/pool_config.json \
  --dustswap-freq 600 \
  --disable-slippage-probes \
  --quiet-harness \
  --stream-blade blade-b10
```

Options:
- `--skip-build` - Skip C++ compilation (use if binary already built)
- `--stream-blade <blade>` - Stream stdout from one blade for progress monitoring
- `--disable-slippage-probes` - Match fast local grid sweeps by skipping probe sampling
- `--quiet-harness` - Suppress remote harness progress logs while keeping blade status logs

### 3. Plot results

```bash
uv run python arb_sim/plot_heatmap_nd_opt.py \
  --metrics vp,apy,apy_net,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,apy_mask_5,apy_mask_10,apy_mask_30,apy_mask_inv_A,tw_real_slippage_5pct \
  --ncol 5 \
  --arb arb_sim/cluster_orchestration/results/cluster_sweep_latest.json
```

## Utility Commands

```bash
# Check cluster status (blade reachability, load, RAM)
uv run python arb_sim/cluster_orchestration/utils.py check

# Kill all running jobs
uv run python arb_sim/cluster_orchestration/utils.py kill -y

# Clean job artifacts
uv run python arb_sim/cluster_orchestration/utils.py clean
```

## Architecture

- **14 blades**: blade-a5 to blade-a10, blade-b1 to blade-b10 (excluding a1-a4, a9, b3)
- **Shared NFS**: `/home/heswithme` accessible from all blades
- **Per blade**: 128 logical cores, ~3.5TB RAM

### Workflow

1. `build.py` - Compile C++ harness once on any blade (shared via NFS)
2. `distribute.py` - Upload single pools.json, compute index ranges per blade
3. `run.py` - Execute harness on all blades in parallel with `--pool-start/--pool-end`
4. `collect.py` - Download and merge results from shared NFS

All transfers use `rsync -z` for compression.
