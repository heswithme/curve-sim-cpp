#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Iterable, Tuple, List
import json
from datetime import datetime, timezone
import statistics as stats
from collections import deque


# Inline helpers (was common_io)
def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def parse_date_arg(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    if s.isdigit():
        # Accept unix seconds directly
        if len(s) >= 10:
            return int(s)
        # Accept compact dates without separators
        # Prefer YYYYMMDD when first 4 look like a year, else DDMMYYYY
        if len(s) == 8:
            try:
                y = int(s[0:4]); m = int(s[4:6]); d = int(s[6:8])
                if 1900 <= y <= 2100:
                    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())
            except Exception:
                pass
            # Try DDMMYYYY
            try:
                d = int(s[0:2]); m = int(s[2:4]); y = int(s[4:8])
                return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())
            except Exception:
                pass
        # Fallback: treat as seconds if short
        return int(s)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            pass
    raise ValueError(f"Unrecognized date/time: {s}")


def cut_rows(rows: list, t0: Optional[int], t1: Optional[int]) -> list:
    if t0 is None and t1 is None:
        return rows
    out = []
    for r in rows:
        ts = int(r[0])
        if (t0 is not None and ts < t0) or (t1 is not None and ts > t1):
            continue
        out.append(r)
    return out


def candle_abs_length(r: list) -> float:
    o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
    return max(o, h, l, c) - min(o, h, l, c)


def moving_average(series: Iterable[Tuple[int, float]], window_seconds: int) -> Iterable[Tuple[int, float]]:
    buf: deque[Tuple[int, float]] = deque()
    s = 0.0
    for ts, val in series:
        buf.append((ts, val))
        s += val
        while buf and ts - buf[0][0] > window_seconds:
            _, v = buf.popleft()
            s -= v
        yield ts, (s / len(buf)) if buf else 0.0


def three_sigma_threshold(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = float(stats.mean(values))
    sd = float(stats.pstdev(values)) if len(values) > 1 else 0.0
    return mu + 3.0 * sd


def flip_row(r: list) -> Optional[list]:
    ts = int(r[0])
    o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
    v = float(r[5]) if len(r) > 5 else 0.0
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return None
    o1 = 1.0 / o
    c1 = 1.0 / c
    hi = max(o, h, l, c)
    lo = min(o, h, l, c)
    h1 = 1.0 / lo
    l1 = 1.0 / hi
    h1 = max(h1, o1, c1)
    l1 = min(l1, o1, c1)
    return [ts, o1, h1, l1, c1, v]


def op_cut(path: Path, start: Optional[str], end: Optional[str]) -> Path:
    data = load_json(path)
    t0 = parse_date_arg(start)
    t1 = parse_date_arg(end)
    out_rows = cut_rows(data, t0, t1)
    # Build output as <stem>.cut.json keeping directory and dropping only final extension
    out = path.with_name(path.stem + ".cut.json")
    save_json(out, out_rows)
    return out


def op_filter(path: Path) -> Path:
    data = load_json(path)
    # 1) Three sigmas on candle height (absolute length)
    lengths = [max(0.0, candle_abs_length(r)) for r in data]
    thr_len = three_sigma_threshold(lengths)
    keep1 = [i for i, L in enumerate(lengths) if L <= thr_len]

    # 2) Three sigmas on 7-day MA deviations (Close vs trailing 7d MA)
    window_seconds = 7 * 86400
    series = [(int(data[i][0]), float(data[i][4])) for i in keep1]
    ma = list(moving_average(series, window_seconds))
    idx_by_ts = {int(data[i][0]): i for i in keep1}
    def rel_dev(i: int, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return abs(float(data[i][4]) - baseline) / baseline
    devs = [rel_dev(idx_by_ts.get(ts, -1), baseline) if idx_by_ts.get(ts, -1) != -1 else 0.0 for ts, baseline in ma]
    thr_dev = three_sigma_threshold(devs)
    keep2 = [i for i, (ts, baseline) in zip(keep1, ma) if (baseline <= 0 or rel_dev(i, baseline) <= thr_dev)]

    # 3) Three sigmas on C->O distance (abs difference in price units)
    co = [abs(float(data[i][4]) - float(data[i][1])) for i in keep2]
    thr_co = three_sigma_threshold(co)
    keep3 = [i for i in keep2 if abs(float(data[i][4]) - float(data[i][1])) <= thr_co]

    out_rows = [data[i] for i in keep3]
    out = path.with_name(path.stem + ".filtered.json")
    save_json(out, out_rows)
    return out


def op_flip(path: Path) -> Path:
    data = load_json(path)
    out_rows = []
    skipped = 0
    for r in data:
        fr = flip_row(r)
        if fr is None:
            skipped += 1
            continue
        out_rows.append(fr)
    out = path.with_name(path.stem + ".flipped.json")
    save_json(out, out_rows)
    print(f"Skipped {skipped} rows with non-positive prices")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Process OHLC series: cut, filter, flip")
    ap.add_argument("file", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cut", action="store_true")
    g.add_argument("--filter", action="store_true")
    g.add_argument("--flip", action="store_true")
    ap.add_argument("--start", type=str, default=None, help="Start for cut (YYYY-MM-DD, YYYYMMDD, DDMMYYYY, or unix seconds)")
    ap.add_argument("--end", type=str, default=None, help="End for cut (YYYY-MM-DD, YYYYMMDD, DDMMYYYY, or unix seconds)")
    args = ap.parse_args()

    if args.cut:
        out = op_cut(args.file, args.start, args.end)
        print(f"Wrote {out}")
    elif args.filter:
        out = op_filter(args.file)
        print(f"Wrote {out}")
    elif args.flip:
        out = op_flip(args.file)
        print(f"Wrote {out}")
    else:
        ap.error("Specify one of --cut, --filter, or --flip")


if __name__ == "__main__":
    main()
