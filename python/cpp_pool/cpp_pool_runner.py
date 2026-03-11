#!/usr/bin/env python3
"""
Unified C++ pool benchmark runner (mode: i | d | ld).

Prefers the runtime benchmark harness and falls back to typed binaries only when
the runtime binary is unavailable.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from typing import Dict


SUPPORTED_MODES = ("i", "d", "ld")
TYPED_TARGETS = {
    "i": "benchmark_harness_i",
    "d": "benchmark_harness_d",
    "ld": "benchmark_harness_ld",
}


class CppPoolRunner:
    def __init__(self, cpp_project_path: str):
        self.cpp_project_path = Path(cpp_project_path)
        self.build_dir = self.cpp_project_path / "build"
        self.harness_path = self.build_dir / "benchmark_harness"
        self.typed_harness_paths = {
            mode: self.build_dir / target for mode, target in TYPED_TARGETS.items()
        }

    def configure_build(self):
        # If build dir was generated from a different checkout, wipe it.
        cache_path = self.build_dir / "CMakeCache.txt"
        if cache_path.exists():
            try:
                home_dir = None
                with cache_path.open("r") as f:
                    for line in f:
                        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                            home_dir = os.path.realpath(line.split("=", 1)[1].strip())
                            break
                if home_dir and os.path.realpath(self.cpp_project_path) != home_dir:
                    print(
                        f"Found stale CMake cache for {home_dir}; wiping {self.build_dir}"
                    )
                    shutil.rmtree(self.build_dir)
            except OSError:
                pass

        self.build_dir.mkdir(parents=True, exist_ok=True)
        print("Configuring C++ build (Release)...")
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.cpp_project_path),
                "-B",
                str(self.build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed: {result.stderr}")

    def _build_target(self, target: str):
        self.configure_build()
        print(f"Building C++ target: {target} ...")
        result = subprocess.run(
            [
                "cmake",
                "--build",
                str(self.build_dir),
                "--config",
                "Release",
                "--target",
                target,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed: {result.stderr}")
        print(f"✓ Built C++ target: {target}")

    def build_runtime_harness(self):
        self._build_target("benchmark_harness")
        if not self.harness_path.exists():
            raise RuntimeError(f"Missing built harness: {self.harness_path}")
        return self.harness_path

    def build_typed_harness(self, mode: str):
        path = self.typed_harness_paths[mode]
        self._build_target(TYPED_TARGETS[mode])
        if not path.exists():
            raise RuntimeError(f"Missing built harness: {path}")
        return path

    def run_benchmark(
        self, mode: str, pool_configs_file: str, sequences_file: str, output_file: str
    ) -> Dict:
        if mode not in SUPPORTED_MODES:
            raise ValueError("mode must be one of: i, d, ld")

        try:
            harness_bin = self.build_runtime_harness()
        except RuntimeError as exc:
            harness_bin = self.build_typed_harness(mode)
            cmd = [str(harness_bin), pool_configs_file, sequences_file, output_file]
            print(
                "Runtime benchmark_harness unavailable, using typed fallback: "
                f"{harness_bin.name}"
            )
            print(f"  runtime build error: {exc}")
        else:
            cmd = [
                str(harness_bin),
                mode,
                pool_configs_file,
                sequences_file,
                output_file,
            ]

        print(f"Running C++ harness binary: {harness_bin.name} ...")
        import time as _time

        t0 = _time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        t1 = _time.perf_counter()
        harness_time = t1 - t0
        if result.returncode != 0:
            raise RuntimeError(f"Harness execution failed: {result.stderr}")
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line:
                    print(f"  {line}")
        with open(output_file, "r") as f:
            results = json.load(f)
        total_tests = len(results.get("results", []))
        print(f"\n✓ Processed {total_tests} pool-sequence combinations")
        print(f"✓ Results written to {output_file}")
        md = results.get("metadata", {})
        md["harness_time_s"] = harness_time
        results["metadata"] = md
        return results

    def format_json_output(self, json_file: str):
        with open(json_file, "r") as f:
            data = json.load(f)
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        print("✓ Formatted output JSON")


def run_cpp_pool(
    mode: str, pool_configs_file: str, sequences_file: str, output_file: str
) -> Dict:
    repo_root = Path(__file__).resolve().parents[2]
    cpp_project_path = str(repo_root / "cpp_modular")
    runner = CppPoolRunner(cpp_project_path)
    results = runner.run_benchmark(mode, pool_configs_file, sequences_file, output_file)
    runner.format_json_output(output_file)
    return results


def main():
    if len(sys.argv) != 5 or sys.argv[1] not in SUPPORTED_MODES:
        print(
            "Usage: python cpp_pool_runner.py <i|d|ld> <pool_configs.json> <sequences.json> <output.json>"
        )
        return 1
    mode = sys.argv[1]
    pool_configs = sys.argv[2]
    sequences = sys.argv[3]
    output = sys.argv[4]
    if not os.path.exists(pool_configs):
        print(f"❌ Pool configs not found: {pool_configs}")
        return 1
    if not os.path.exists(sequences):
        print(f"❌ Sequences not found: {sequences}")
        return 1
    try:
        run_cpp_pool(mode, pool_configs, sequences, output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
