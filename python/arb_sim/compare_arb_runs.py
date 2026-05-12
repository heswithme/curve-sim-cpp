#!/usr/bin/env python3
"""Compare two arb_run JSON artifacts.

The default comparison ignores timing and execution-control metadata, so it is
suitable for checking whether a performance patch preserved simulator economics
and final state while allowing thread count, wall time, and per-pool timing to
change.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_IGNORE_RESULT_FIELDS = {"pool_exec_ms"}
DEFAULT_IGNORE_RUN_FIELDS = {"success"}
DEFAULT_IGNORE_METADATA_FIELDS = {
    "exec_ms",
    "candles_read_ms",
    "harness_wall_ms",
    "postprocess_ms",
    "wall_ms",
    "run_started_at",
    "threads",
    "quiet",
    "quiet_harness",
    "harness_exe",
    "candles_file",
}


def is_coordinate_field(key: str) -> bool:
    if not key.startswith("x"):
        return False
    for suffix in ("_key", "_val"):
        if key.endswith(suffix):
            axis = key[1:-len(suffix)]
            return axis.isdigit()
    return False


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        raise ValueError(f"expected JSON object in {path}")
    return root


def filtered_metadata(root: dict[str, Any]) -> dict[str, Any]:
    metadata = root.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return {
        key: value
        for key, value in metadata.items()
        if key not in DEFAULT_IGNORE_METADATA_FIELDS
    }


def filtered_run(run: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in run.items()
        if key not in DEFAULT_IGNORE_RUN_FIELDS and not is_coordinate_field(key)
    }
    result = out.get("result")
    if isinstance(result, dict):
        out["result"] = {
            key: value
            for key, value in result.items()
            if key not in DEFAULT_IGNORE_RESULT_FIELDS
        }
    return out


def comparable_metadata(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_filtered = filtered_metadata(left)
    right_filtered = filtered_metadata(right)
    common_keys = set(left_filtered) & set(right_filtered)
    return (
        {key: left_filtered[key] for key in common_keys},
        {key: right_filtered[key] for key in common_keys},
    )


def equal_numeric_strings(a: str, b: str) -> bool:
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False


def first_diff(a: Any, b: Any, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        a_keys = set(a)
        b_keys = set(b)
        if a_keys != b_keys:
            missing_a = sorted(b_keys - a_keys)
            missing_b = sorted(a_keys - b_keys)
            return f"{path}: key mismatch missing_left={missing_a} missing_right={missing_b}"
        for key in sorted(a_keys):
            diff = first_diff(a[key], b[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for idx, (left, right) in enumerate(zip(a, b)):
            diff = first_diff(left, right, f"{path}[{idx}]")
            if diff is not None:
                return diff
        return None
    if a != b:
        if isinstance(a, str) and isinstance(b, str) and equal_numeric_strings(a, b):
            return None
        return f"{path}: {a!r} != {b!r}"
    return None


def compare_runs(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    left_metadata, right_metadata = comparable_metadata(left, right)
    metadata_diff = first_diff(
        left_metadata,
        right_metadata,
        "$.metadata",
    )
    if metadata_diff is not None:
        return metadata_diff

    left_runs = left.get("runs", [])
    right_runs = right.get("runs", [])
    if not isinstance(left_runs, list) or not isinstance(right_runs, list):
        return "$.runs: both artifacts must contain run arrays"
    if len(left_runs) != len(right_runs):
        return f"$.runs: len {len(left_runs)} != {len(right_runs)}"

    for idx, (left_run, right_run) in enumerate(zip(left_runs, right_runs)):
        if not isinstance(left_run, dict) or not isinstance(right_run, dict):
            return f"$.runs[{idx}]: both runs must be objects"
        diff = first_diff(
            filtered_run(left_run),
            filtered_run(right_run),
            f"$.runs[{idx}]",
        )
        if diff is not None:
            return diff
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diff = compare_runs(load_json(args.left), load_json(args.right))
    if diff is not None:
        print(f"DIFF: {diff}")
        return 1
    print("MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
