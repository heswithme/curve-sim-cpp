# Cluster Orchestration

Run TwoCrypto pool parameter sweeps on the NixOS HPC cluster.

## Quick Start

### 1. Generate pool configs (N-dimensional grid)

```bash
uv run arb_sim/generate_pools_nd.py
```

Edit the script to configure grid dimensions and parameter ranges.

### 2. Run sweep on cluster

```bash
uv run arb_sim/cluster_orchestration/orchestrate.py \
  --pools arb_sim/run_data/pool_config.json \
  --stream-blade blade-b10
```

Options:
- `--skip-build` - Skip C++ compilation (use if binary already built)
- `--stream-blade <blade>` - Stream stdout from one blade for progress monitoring

### 3. Plot results

```bash
uv run arb_sim/plot_heatmap_nd.py \
  --metrics vp,apy,apy_net,tvl_growth,total_notional_coin0,n_rebalances,trades,donations,tw_avg_pool_fee,avg_rel_price_diff,apy_mask_5,apy_mask_10,apy_mask_30,apy_mask_inv_A,tw_real_slippage_5pct \
  --ncol 5 \
  --arb arb_sim/cluster_orchestration/results/cluster_sweep_latest.json
```

## Utility Commands

```bash
# Check cluster status (blade reachability, load, RAM)
uv run arb_sim/cluster_orchestration/utils.py check

# Kill all running jobs
uv run arb_sim/cluster_orchestration/utils.py kill -y

# Clean job artifacts
uv run arb_sim/cluster_orchestration/utils.py clean
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
