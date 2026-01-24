#!/usr/bin/env python3
"""
Cluster utility functions.

Commands:
    status  - Show blade status (reachability, binary, load)
    check   - Alias for status
    clean   - Remove job data from shared NFS
    kill    - Kill running harness processes
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from config import (
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
    DEFAULT_BLADES,
    ALL_BLADES,
    REMOTE_BUILD,
    REMOTE_BASE,
    HARNESS_BINARY,
)


def run_ssh(blade: str, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run command on blade via SSH."""
    cmd = ["ssh"] + SSH_OPTIONS + ["-i", str(SSH_KEY), f"{SSH_USER}@{blade}", command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def get_blade_status(blade: str) -> dict:
    """Get status info from a blade."""
    status = {
        "blade": blade,
        "reachable": False,
        "binary": False,
        "cores": 0,
        "load": "-",
        "mem_free_gb": 0,
        "home_owner": "-",
    }

    try:
        # Check reachable
        result = run_ssh(blade, "echo OK", timeout=15)
        if "OK" not in result.stdout:
            return status
        status["reachable"] = True

        # Check binary
        binary = f"{REMOTE_BUILD}/{HARNESS_BINARY}"
        result = run_ssh(blade, f"test -x {binary} && echo YES", timeout=10)
        status["binary"] = "YES" in result.stdout

        # System info
        result = run_ssh(
            blade,
            "nproc && uptime && free -g | grep Mem && stat -c '%U:%G' /home/heswithme",
            timeout=15,
        )
        lines = result.stdout.strip().split("\n")

        if lines:
            status["cores"] = int(lines[0])
        if len(lines) >= 2 and "load average:" in lines[1]:
            status["load"] = lines[1].split("load average:")[-1].strip()[:15]
        if len(lines) >= 3:
            parts = lines[2].split()
            if len(parts) >= 4:
                status["mem_free_gb"] = int(parts[3])
        if len(lines) >= 4:
            status["home_owner"] = lines[3].strip()
    except Exception as e:
        status["error"] = str(e)

    return status


def show_status(blades: List[str] = None):
    """Display cluster status (parallel queries)."""
    if blades is None:
        blades = ALL_BLADES

    print(
        f"\n{'Blade':<12} {'Status':<10} {'Binary':<8} {'Home':<12} {'Cores':<6} {'Load':<18} {'RAM Free':<10}"
    )
    print("-" * 84)

    # Query all blades in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=min(32, len(blades))) as executor:
        futures = {executor.submit(get_blade_status, b): b for b in blades}
        for future in as_completed(futures):
            blade = futures[future]
            results[blade] = future.result()

    # Print in original order
    for blade in blades:
        s = results[blade]
        print(
            f"{blade:<12} "
            f"{'OK' if s['reachable'] else 'DOWN':<10} "
            f"{'YES' if s['binary'] else 'NO':<8} "
            f"{s['home_owner']:<12} "
            f"{s['cores'] or '-':<6} "
            f"{s['load']:<18} "
            f"{s['mem_free_gb']}G"
            if s["mem_free_gb"]
            else "-"
        )
    print()


def clean(blades: List[str] = None, confirm: bool = True):
    """Clean job data from shared NFS."""
    if blades is None:
        blades = DEFAULT_BLADES

    blade = blades[0]  # Only need one blade for shared NFS

    if confirm:
        print(f"This will delete {REMOTE_BASE} on shared NFS.")
        if input("Continue? [y/N] ").lower() != "y":
            print("Aborted.")
            return

    print(f"Cleaning {REMOTE_BASE} via {blade}...")
    try:
        # Keep the base dir but clean contents
        run_ssh(
            blade,
            f"rm -rf {REMOTE_BASE}/jobs/* {REMOTE_BASE}/results/* {REMOTE_BASE}/data/*",
            timeout=60,
        )
        print("Done.")
    except Exception as e:
        print(f"Failed: {e}")


def _kill_on_blade(blade: str) -> tuple:
    """Kill harness on a single blade. Returns (blade, success, message)."""
    try:
        run_ssh(blade, "pkill -f arb_harness || true", timeout=10)
        return (blade, True, "Killed")
    except Exception as e:
        return (blade, False, f"Error: {e}")


def kill_jobs(blades: List[str] = None, confirm: bool = True):
    """Kill harness processes on all blades (parallel)."""
    if blades is None:
        blades = ALL_BLADES

    if confirm:
        print(f"This will kill arb_harness on {len(blades)} blades.")
        if input("Continue? [y/N] ").lower() != "y":
            print("Aborted.")
            return

    results = {}
    with ThreadPoolExecutor(max_workers=min(32, len(blades))) as executor:
        futures = {executor.submit(_kill_on_blade, b): b for b in blades}
        for future in as_completed(futures):
            blade, success, msg = future.result()
            results[blade] = (success, msg)

    # Print in original order
    for blade in blades:
        success, msg = results[blade]
        print(f"[{blade}] {msg}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cluster utilities")
    sub = parser.add_subparsers(dest="cmd")

    # status
    p = sub.add_parser("status", help="Show cluster status")
    p.add_argument("--blades", nargs="+")

    # check (alias for status)
    p = sub.add_parser("check", help="Check blades (alias for status)")
    p.add_argument("--blades", nargs="+")

    # clean
    p = sub.add_parser("clean", help="Clean job data")
    p.add_argument("-y", "--yes", action="store_true")

    # kill
    p = sub.add_parser("kill", help="Kill running jobs")
    p.add_argument("--blades", nargs="+")
    p.add_argument("-y", "--yes", action="store_true")

    args = parser.parse_args()

    if args.cmd in {"status", "check"}:
        show_status(args.blades)
    elif args.cmd == "clean":
        clean(confirm=not args.yes)
    elif args.cmd == "kill":
        kill_jobs(args.blades, confirm=not args.yes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
