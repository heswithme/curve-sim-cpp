#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULT_RESULT_PATH = REPO_ROOT / "comparison-results" / "nevergrad_fee_result.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "comparison-results" / "nevergrad_inspect_output.json"
DEFAULT_POOL_PATH = (
    REPO_ROOT / "python" / "arb_sim" / "run_data" / "inspect_nevergrad_pool.json"
)
DEFAULT_DETAILED_INTERVAL = 1000
DEFAULT_THREADS = 8
DEFAULT_REAL = "double"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and inspect best Nevergrad result"
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pool-config", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--real", type=str, default=DEFAULT_REAL)
    parser.add_argument(
        "--detailed-interval", type=int, default=DEFAULT_DETAILED_INTERVAL
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_best_pool(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    payload = result.get("best_pool")
    if isinstance(payload, dict) and "pool_config_entry" in payload:
        return payload
    raise SystemExit(
        f"Missing replayable payload best_pool in {result_path}. Rerun the optimizer to regenerate the result file."
    )


def write_pool_config(path: Path, saved_pool: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_by": "inspect_nevergrad_result.py",
            "source": saved_pool.get("which", "unknown"),
            "datafile": saved_pool["candles_path"],
            "server_config": saved_pool.get("server_config", {}),
            "params_display": saved_pool.get("params_display", {}),
            "request_params": saved_pool.get("request_params", {}),
            "template_pool": saved_pool.get("template_pool", {}),
            "template_costs": saved_pool.get("template_costs", {}),
        },
        "pools": [saved_pool["pool_config_entry"]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_arb_sim_command(
    pool_config: Path,
    output_path: Path,
    candles_path: Path,
    server_config: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "arb_sim/arb_sim.py",
        "--real",
        args.real,
        "--dustswapfreq",
        str(int(server_config.get("dustswapfreq", 3600))),
        "--min-swap",
        str(float(server_config.get("min_swap", 1e-6))),
        "--max-swap",
        str(float(server_config.get("max_swap", 1.0))),
        "--candle-filter",
        str(float(server_config.get("candle_filter", 99.0))),
        "-n",
        str(int(args.threads)),
        "--detailed-log",
        "--detailed-interval",
        str(int(args.detailed_interval)),
        "--out",
        str(output_path),
        "--pool-config",
        str(pool_config),
    ]
    if server_config.get("disable_slippage_probes", False):
        cmd.append("--disable-slippage-probes")
    if args.skip_build:
        cmd.append("--skip-build")
    cmd.append(str(candles_path))
    return cmd


def load_output(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_dict(title: str, data: dict[str, Any]) -> None:
    print(f"\n{title}:")
    for key in sorted(data):
        print(f"  {key}: {data[key]}")


def print_report(saved_pool: dict[str, Any], output: dict[str, Any]) -> None:
    runs = output.get("runs", [])
    if not runs:
        raise SystemExit("No runs found in inspect output")

    run = runs[0]
    print("selected:")
    print(f"  which: {saved_pool.get('which')}")
    print(f"  candles_path: {saved_pool.get('candles_path')}")
    print_dict("params_display", saved_pool.get("params_display", {}))
    print_dict("request_params", saved_pool.get("request_params", {}))

    print_dict("result", run.get("result", {}))
    print_dict("final_state", run.get("final_state", {}))


def maybe_plot(output_path: Path, args: argparse.Namespace) -> None:
    if not args.plot:
        return
    detailed_path = output_path.parent / "detailed-output.json"
    if not detailed_path.exists():
        print(f"No detailed output found at {detailed_path}")
        return
    cmd = ["uv", "run", "arb_sim/plot_price_scale.py", "--no-save", str(detailed_path)]
    subprocess.Popen(cmd, cwd=PYTHON_DIR, start_new_session=True)


def main() -> int:
    args = parse_args()
    result = load_result(args.result)
    saved_pool = select_best_pool(result, args.result)

    candles_path = Path(saved_pool["candles_path"]).resolve()
    if not candles_path.exists():
        raise SystemExit(f"Candles file not found: {candles_path}")

    pool_config_path = write_pool_config(args.pool_config, saved_pool)
    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_arb_sim_command(
        pool_config=pool_config_path,
        output_path=output_path,
        candles_path=candles_path,
        server_config=saved_pool.get("server_config", {}),
        args=args,
    )

    subprocess.run(cmd, cwd=PYTHON_DIR, check=True)
    output = load_output(output_path)
    print_report(saved_pool, output)
    maybe_plot(output_path, args)
    print(f"\nSaved inspect output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
