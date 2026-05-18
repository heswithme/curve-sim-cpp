#!/usr/bin/env python3
"""Live cluster monitor using one persistent SSH session per blade."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from config import DEFAULT_BLADES, SSH_KEY, SSH_OPTIONS, SSH_USER


@dataclass
class BladeSample:
    status: str = "starting"
    cpu_pct: float | None = None
    load1: float | None = None
    load5: float | None = None
    load15: float | None = None
    mem_avail_gb: int | None = None
    mem_total_gb: int | None = None
    harness_procs: int | None = None
    harness_threads: int | None = None
    remote_time: str = "-"
    updated_at: float = 0.0
    error: str = ""


REMOTE_SCRIPT = r"""
interval="${MONITOR_INTERVAL:-2}"
prev="$(awk '/^cpu /{for (i=2; i<=NF; ++i) printf "%s ", $i; print ""}' /proc/stat)"
while true; do
    sleep "$interval"
    cur="$(awk '/^cpu /{for (i=2; i<=NF; ++i) printf "%s ", $i; print ""}' /proc/stat)"
    cpu="$(
        awk -v prev="$prev" -v cur="$cur" '
            BEGIN {
                split(prev, p, " ");
                split(cur, c, " ");
                pt = 0; ct = 0;
                for (i = 1; i <= length(c); ++i) {
                    pt += p[i];
                    ct += c[i];
                }
                pidle = p[4] + p[5];
                cidle = c[4] + c[5];
                dt = ct - pt;
                di = cidle - pidle;
                busy = dt > 0 ? 100.0 * (dt - di) / dt : 0.0;
                printf "%.1f", busy;
            }
        '
    )"
    prev="$cur"

    set -- $(cut -d' ' -f1-3 /proc/loadavg)
    load1="$1"; load5="$2"; load15="$3"
    read -r mem_avail mem_total < <(free -g | awk '/^Mem:/ {print $7, $2}')
    procs="$(pgrep -fc '[a]rb_harness' || true)"
    threads="$(ps -eo comm,nlwp | awk '$1 ~ /^arb_harness/ {sum += $2} END {print sum + 0}')"
    now="$(date +%H:%M:%S)"
    printf 'OK|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$cpu" "$load1" "$load5" "$load15" "$mem_avail" "$mem_total" "$procs" "$threads" "$now"
done
"""


def ssh_monitor_cmd(blade: str, interval: float) -> list[str]:
    ssh_options = list(SSH_OPTIONS)
    ssh_options.extend(["-o", "BatchMode=yes"])
    remote = (
        f"MONITOR_INTERVAL={shlex.quote(str(interval))} "
        f"bash -lc {shlex.quote(REMOTE_SCRIPT)}"
    )
    return [
        "ssh",
        *ssh_options,
        "-i",
        str(SSH_KEY),
        f"{SSH_USER}@{blade}",
        remote,
    ]


def parse_sample(line: str) -> BladeSample:
    parts = line.rstrip("\n").split("|")
    if len(parts) != 10 or parts[0] != "OK":
        return BladeSample(status="bad-line", error=line.strip(), updated_at=time.time())
    return BladeSample(
        status="ok",
        cpu_pct=float(parts[1]),
        load1=float(parts[2]),
        load5=float(parts[3]),
        load15=float(parts[4]),
        mem_avail_gb=int(parts[5]),
        mem_total_gb=int(parts[6]),
        harness_procs=int(parts[7]),
        harness_threads=int(parts[8]),
        remote_time=parts[9],
        updated_at=time.time(),
    )


async def monitor_blade(
    blade: str,
    interval: float,
    reconnect_delay: float,
    samples: dict[str, BladeSample],
) -> None:
    while True:
        proc = await asyncio.create_subprocess_exec(
            *ssh_monitor_cmd(blade, interval),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        samples[blade] = BladeSample(status="connected", updated_at=time.time())

        async def read_stderr() -> None:
            assert proc.stderr is not None
            async for raw in proc.stderr:
                text = raw.decode(errors="replace").strip()
                if text:
                    cur = samples.get(blade, BladeSample())
                    cur.error = text[-160:]
                    samples[blade] = cur

        stderr_task = asyncio.create_task(read_stderr())
        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                samples[blade] = parse_sample(raw.decode(errors="replace"))
        except asyncio.CancelledError:
            stderr_task.cancel()
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
            raise
        finally:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        rc = await proc.wait()
        samples[blade] = BladeSample(
            status=f"reconnect rc={rc}",
            updated_at=time.time(),
            error=samples.get(blade, BladeSample()).error,
        )
        await asyncio.sleep(reconnect_delay)


def bar(cpu_pct: float | None, width: int = 18) -> str:
    if cpu_pct is None:
        return "-" * width
    filled = max(0, min(width, round(width * cpu_pct / 100.0)))
    return "#" * filled + "." * (width - filled)


def fmt_float(value: float | None, width: int = 5, precision: int = 1) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_int(value: int | None, width: int = 5) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}d}"


def render(blades: Iterable[str], samples: dict[str, BladeSample], no_clear: bool) -> None:
    if not no_clear:
        print("\033[H\033[J", end="")

    now = datetime.now().strftime("%H:%M:%S")
    print(f"cluster monitor {now}  persistent ssh sessions: {len(list(blades))}")
    print(
        f"{'blade':<11} {'cpu%':>6} {'cpu':<18} {'load 1/5/15':<22} "
        f"{'mem avail/total':<16} {'procs':>5} {'thr':>5} {'age':>5} {'remote':>8}  status"
    )
    print("-" * 116)

    total_cpu = 0.0
    cpu_count = 0
    total_threads = 0
    active_blades = 0
    ts = time.time()

    for blade in blades:
        s = samples.get(blade, BladeSample())
        age = ts - s.updated_at if s.updated_at else 0.0
        if s.cpu_pct is not None:
            total_cpu += s.cpu_pct
            cpu_count += 1
        if s.harness_threads:
            total_threads += s.harness_threads
        if s.harness_procs:
            active_blades += 1

        loads = (
            f"{fmt_float(s.load1)}/{fmt_float(s.load5)}/{fmt_float(s.load15)}"
            if s.load1 is not None
            else "-"
        )
        mem = (
            f"{s.mem_avail_gb}/{s.mem_total_gb}G"
            if s.mem_avail_gb is not None and s.mem_total_gb is not None
            else "-"
        )
        status = s.status if not s.error else f"{s.status}: {s.error}"
        print(
            f"{blade:<11} {fmt_float(s.cpu_pct, 6, 1)} {bar(s.cpu_pct):<18} "
            f"{loads:<22} {mem:<16} {fmt_int(s.harness_procs):>5} "
            f"{fmt_int(s.harness_threads):>5} {age:5.1f} {s.remote_time:>8}  {status}"
        )

    avg_cpu = total_cpu / cpu_count if cpu_count else 0.0
    print("-" * 116)
    print(
        f"active blades: {active_blades}/{len(list(blades))}  "
        f"avg cpu: {avg_cpu:.1f}%  harness threads: {total_threads}"
    )
    print("Ctrl-C to stop")
    sys.stdout.flush()


async def run_monitor(args: argparse.Namespace) -> None:
    samples: dict[str, BladeSample] = {}
    tasks = [
        asyncio.create_task(
            monitor_blade(blade, args.interval, args.reconnect_delay, samples)
        )
        for blade in args.blades
    ]
    try:
        frames = 0
        while True:
            render(args.blades, samples, args.no_clear)
            frames += 1
            if args.frames is not None and frames >= args.frames:
                return
            await asyncio.sleep(args.refresh)
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blades", nargs="+", default=DEFAULT_BLADES)
    parser.add_argument("--interval", type=float, default=2.0, help="Remote sample interval")
    parser.add_argument("--refresh", type=float, default=1.0, help="Local redraw interval")
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=15.0,
        help="Seconds to wait before reconnecting a failed SSH session",
    )
    parser.add_argument("--no-clear", action="store_true", help="Append frames instead of clearing")
    parser.add_argument("--frames", type=int, help="Render N frames and exit")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.refresh <= 0:
        parser.error("--refresh must be positive")
    if args.reconnect_delay <= 0:
        parser.error("--reconnect-delay must be positive")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")

    try:
        asyncio.run(run_monitor(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
