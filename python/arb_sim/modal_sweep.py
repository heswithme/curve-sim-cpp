#!/usr/bin/env python3
"""
Modal cloud sweep for TwoCrypto pool parameter optimization.

Distributes pools across up to MAX_CONTAINERS containers, each with N_CPU cores.
"""

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

# -------------------- Configuration --------------------
N_CPU = 64
MAX_CONTAINERS = 100
MEMORY_MB = 32768

# -------------------- Modal Setup --------------------
app = modal.App("twocrypto-sweep")

_LOCAL_SCRIPT_PATH = Path(__file__).resolve()
_IS_LOCAL = len(_LOCAL_SCRIPT_PATH.parents) > 2

if _IS_LOCAL:
    _cpp_dir = _LOCAL_SCRIPT_PATH.parents[2] / "cpp_modular"
    image = (
        modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
        .apt_install("build-essential", "cmake", "libboost-all-dev")
        .add_local_dir(str(_cpp_dir / "include"), "/app/cpp_modular/include", copy=True)
        .add_local_dir(str(_cpp_dir / "src"), "/app/cpp_modular/src", copy=True)
        .add_local_file(
            str(_cpp_dir / "CMakeLists.txt"),
            "/app/cpp_modular/CMakeLists.txt",
            copy=True,
        )
        .run_commands(
            "sed -i 's/-march=native/-march=x86-64 -mtune=generic/g' /app/cpp_modular/CMakeLists.txt && "
            "cd /app/cpp_modular && mkdir -p build && cd build && "
            "cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) arb_harness_ld"
        )
    )
else:
    image = modal.Image.from_registry("ubuntu:24.04", add_python="3.11")

volume = modal.Volume.from_name("twocrypto-trade-data", create_if_missing=True)


@app.function(
    image=image, volumes={"/data": volume}, cpu=N_CPU, memory=MEMORY_MB, timeout=3600
)
def run_batch(pools: list[dict], harness_args: dict, candles_file: str) -> list[dict]:
    """Run a batch of pools on N_CPU threads."""
    import json
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    exe = Path("/app/cpp_modular/build/arb_harness_ld")
    candles = Path(f"/data/{candles_file}")

    if not exe.exists() or not candles.exists():
        err = (
            "harness not found" if not exe.exists() else f"candles not found: {candles}"
        )
        return [{"tag": p.get("tag", "?"), "error": err} for p in pools]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"pools": pools}, f)
        cfg_path = f.name

    out_path = tempfile.mktemp(suffix=".json")
    cmd = [str(exe), cfg_path, str(candles), out_path, f"--threads={N_CPU}"]

    for k, v in harness_args.items():
        if v is not None:
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        err = (result.stderr or result.stdout)[:200]
        return [
            {"tag": p.get("tag", "?"), "error": f"rc={result.returncode}: {err}"}
            for p in pools
        ]

    try:
        runs = json.load(open(out_path)).get("runs", [])
        return [
            {
                "tag": pools[i].get("tag", f"pool_{i}"),
                "pool": pools[i].get("pool", {}),
                "result": runs[i].get("result", {}),
                "final_state": runs[i].get("final_state", {}),
                "elapsed_s": elapsed / len(pools),
            }
            if i < len(runs)
            else {"tag": pools[i].get("tag", "?"), "error": "missing"}
            for i in range(len(pools))
        ]
    except Exception as e:
        return [{"tag": p.get("tag", "?"), "error": str(e)} for p in pools]


def distribute(pools: list, max_containers: int, cpus: int) -> list[list]:
    """Split pools across containers. Target ~cpus pools per container."""
    n = min(max_containers, max(1, math.ceil(len(pools) / cpus)))
    batch_size = math.ceil(len(pools) / n)
    return [pools[i : i + batch_size] for i in range(0, len(pools), batch_size)]


def upload_data(datafile: str):
    """Upload trade data to volume if needed."""
    data_path = Path(datafile)
    filename = data_path.name

    result = subprocess.run(
        ["modal", "volume", "ls", "twocrypto-trade-data"],
        capture_output=True,
        text=True,
    )
    if filename in result.stdout:
        print(f"Trade data present: {filename}")
        return filename

    print(f"Uploading {data_path}...")
    subprocess.run(
        [
            "modal",
            "volume",
            "put",
            "twocrypto-trade-data",
            str(data_path),
            f"/{filename}",
        ],
        check=True,
    )
    return filename


def load_pools(max_pools: int = 0) -> tuple[list, dict, str]:
    """Load pool configs from run_data/pool_config.json."""
    cfg_path = Path(__file__).parent / "run_data/pool_config.json"
    cfg = json.load(open(cfg_path))
    pools = cfg.get("pools", [])
    meta = cfg.get("meta", {})
    datafile = meta.get(
        "datafile", str(Path(__file__).parent / "trade_data/ethusd/ethusdt-2yup.json")
    )
    if max_pools > 0:
        pools = pools[:max_pools]
    return pools, meta, datafile


@app.local_entrypoint()
def main(
    max_pools: int = 0,
    dustswapfreq: int = 600,
    apy_period_days: float = 1.0,
    apy_period_cap: int = 20,
    candle_filter: float = None,
    skip_upload: bool = False,
):
    """Run sweep: distributes pools across up to 100 containers with 64 CPUs each."""
    print(f"=== TwoCrypto Modal Sweep (max {MAX_CONTAINERS} x {N_CPU} CPU) ===")

    pools, meta, datafile = load_pools(max_pools)
    if not pools:
        print("No pools!")
        return

    if not skip_upload:
        candles_file = upload_data(datafile)
    else:
        candles_file = Path(datafile).name
        print(f"Skipping upload, using: {candles_file}")

    harness_args = {
        "dustswapfreq": dustswapfreq,
        "apy_period_days": apy_period_days,
        "apy_period_cap": apy_period_cap,
    }
    if candle_filter is not None:
        harness_args["candle_filter"] = candle_filter

    batches = distribute(pools, MAX_CONTAINERS, N_CPU)
    print(
        f"{len(pools)} pools -> {len(batches)} containers ({len(batches[0])} pools/container)"
    )

    start = datetime.now(timezone.utc)
    results = [
        r
        for batch_results in run_batch.map(
            batches, kwargs={"harness_args": harness_args, "candles_file": candles_file}
        )
        for r in batch_results
    ]
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    errors = [r for r in results if "error" in r]
    print(
        f"\nDone: {len(results)} pools in {elapsed:.1f}s ({len(pools) / elapsed:.1f} pools/sec)"
    )
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"  {e['tag']}: {e['error']}")

    # Save results
    out_path = (
        Path(__file__).parent
        / "run_data"
        / f"modal_sweep_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    json.dump(
        {
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "total_pools": len(pools),
                "errors": len(errors),
                "wall_clock_s": elapsed,
                "containers": len(batches),
                "harness_args": harness_args,
                "grid": meta.get("grid"),
            },
            "runs": results,
        },
        open(out_path, "w"),
        indent=2,
    )
    print(f"Saved: {out_path}")

    # Summary
    vps = [
        r["result"]["vp"]
        for r in results
        if "result" in r and "vp" in r.get("result", {})
    ]
    if vps:
        print(
            f"\nVP: min={min(vps):.3f} max={max(vps):.3f} avg={sum(vps) / len(vps):.3f}"
        )


if __name__ == "__main__":
    print(f"Usage: modal run arb_sim/modal_sweep.py [--max-pools N] [--skip-upload]")
    print(
        f"Config: {MAX_CONTAINERS} containers x {N_CPU} CPUs = {MAX_CONTAINERS * N_CPU} parallel pools"
    )
