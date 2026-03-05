#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "cpp_modular" / "build" / "arb_eval_server"


def build_eval_server(repo_root: Path) -> None:
    build_dir = repo_root / "cpp_modular" / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repo_root / "cpp_modular"),
            "-B",
            str(build_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "arb_eval_server",
            "-j8",
        ],
        check=True,
    )


class EvalServerClient:
    def __init__(
        self,
        binary: Path,
        template_path: Path,
        candles_path: Path,
        pool_index: int = 0,
        n_candles: int = 0,
        candle_filter: float = 99.0,
        min_swap: float = 1e-6,
        max_swap: float = 1.0,
    ) -> None:
        self.binary = binary
        self.template_path = template_path
        self.candles_path = candles_path
        self.pool_index = pool_index
        self.n_candles = n_candles
        self.candle_filter = candle_filter
        self.min_swap = min_swap
        self.max_swap = max_swap
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "EvalServerClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        if self.proc is not None:
            return
        cmd = [
            str(self.binary),
            str(self.template_path),
            str(self.candles_path),
            "--pool-index",
            str(self.pool_index),
            "--n-candles",
            str(self.n_candles),
            "--candle-filter",
            str(self.candle_filter),
            "--min-swap",
            str(self.min_swap),
            "--max-swap",
            str(self.max_swap),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write("quit\n")
                    self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        self.proc = None

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None:
            self.start()
        assert self.proc is not None
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Evaluator process missing stdio pipes")

        payload = json.dumps(request, separators=(",", ":"))
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()

        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("Evaluator process terminated before sending a response")

        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("Evaluator returned non-object JSON")
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single fee-vector evaluation via arb_eval_server",
    )
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="Template pool config json (one or many pools)",
    )
    parser.add_argument(
        "--candles",
        required=True,
        type=Path,
        help="Candles json used by modular sim",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"Path to arb_eval_server (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip cmake build of arb_eval_server",
    )

    parser.add_argument("--pool-index", type=int, default=0)
    parser.add_argument("--n-candles", type=int, default=0)
    parser.add_argument("--candle-filter", type=float, default=99.0)
    parser.add_argument("--min-swap", type=float, default=1e-6)
    parser.add_argument("--max-swap", type=float, default=1.0)

    parser.add_argument("--mid-fee", type=float, default=None)
    parser.add_argument("--out-fee", type=float, default=None)
    parser.add_argument("--mid-fee-bps", type=float, default=None)
    parser.add_argument("--out-fee-bps", type=float, default=None)
    parser.add_argument("--fee-gamma", type=float, default=None)

    parser.add_argument("--full", action="store_true", help="Print full response json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    binary_path = args.binary.resolve()
    if not args.skip_build:
        build_eval_server(REPO_ROOT)
    if not binary_path.exists():
        print(f"error: evaluator binary not found at {binary_path}", file=sys.stderr)
        return 1

    request: dict[str, Any] = {"id": 1}
    if args.mid_fee is not None:
        request["mid_fee"] = args.mid_fee
    if args.out_fee is not None:
        request["out_fee"] = args.out_fee
    if args.mid_fee_bps is not None:
        request["mid_fee_bps"] = args.mid_fee_bps
    if args.out_fee_bps is not None:
        request["out_fee_bps"] = args.out_fee_bps
    if args.fee_gamma is not None:
        request["fee_gamma"] = args.fee_gamma

    with EvalServerClient(
        binary=binary_path,
        template_path=args.template.resolve(),
        candles_path=args.candles.resolve(),
        pool_index=args.pool_index,
        n_candles=args.n_candles,
        candle_filter=args.candle_filter,
        min_swap=args.min_swap,
        max_swap=args.max_swap,
    ) as client:
        response = client.evaluate(request)

    if args.full:
        print(json.dumps(response, separators=(",", ":"), sort_keys=True))
        return 0 if response.get("ok", False) else 1

    simple = {
        "ok": bool(response.get("ok", False)),
        "vp": response.get("vp"),
        "apy": response.get("apy"),
        "apy_net": response.get("apy_net"),
        "price_diff": response.get("price_diff"),
    }
    if not simple["ok"]:
        simple["error"] = response.get("error", "unknown error")
    print(json.dumps(simple, separators=(",", ":"), sort_keys=True))
    return 0 if simple["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
