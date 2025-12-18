from typing import Any, List
import json


def _first_candle_ts(path: str) -> int:
    """Return first timestamp (seconds) from candles or events file.

    Supports:
    - Candles: [[ts, o, h, l, c, v], ...]
    - Events:  [[ts, price, volume], ...]
    - Object roots with 'data' or 'events' arrays
    - Dict rows with 'ts' or ISO8601 'time'

    Raises ValueError if file cannot be read or parsed.
    """

    def to_ts(v: Any) -> int | None:
        try:
            t = int(v)
            if t > 10_000_000_000:
                t //= 1000
            return t
        except Exception:
            # try ISO
            try:
                from datetime import datetime, timezone

                s = str(v)
                if s.endswith("Z"):
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except Exception:
                return None

    try:
        with open(path, "r") as f:
            root = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Data file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    arr: List[Any] | None = None
    if isinstance(root, list):
        arr = root
    elif isinstance(root, dict):
        for k in ("data", "events", "candles"):
            if isinstance(root.get(k), list):
                arr = root[k]
                break
    if not arr or not isinstance(arr, list):
        raise ValueError(f"No data array found in {path}")

    first = arr[0]
    if isinstance(first, list) and first:
        tsv = first[0]
        ts = to_ts(tsv)
        if ts is not None:
            return ts
        raise ValueError(f"Cannot parse timestamp from first element in {path}: {tsv}")
    if isinstance(first, dict):
        ts = to_ts(first.get("ts") or first.get("timestamp") or first.get("time"))
        if ts is not None:
            return ts
        raise ValueError(f"Cannot parse timestamp from first dict in {path}: {first}")
    raise ValueError(f"Unsupported data format in {path}")


def _initial_price_from_file(path: str) -> float:
    """Extract initial price from candles/events file.

    - Candles: use close (index 4) of first candle.
    - Events:  use price (index 1) of first event.
    - Object roots with 'data'/'events' arrays handled.

    Raises ValueError if file cannot be read or price cannot be extracted.
    """
    try:
        with open(path, "r") as f:
            root = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Data file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    arr: List[Any] | None = None
    if isinstance(root, list):
        arr = root
    elif isinstance(root, dict):
        for k in ("data", "events", "candles"):
            if isinstance(root.get(k), list):
                arr = root[k]
                break
    if not arr or not isinstance(arr, list) or not arr:
        raise ValueError(f"No data array found in {path}")

    first = arr[0]
    try:
        if isinstance(first, list):
            # Candles: [ts,o,h,l,c,v] or Events: [ts,price,volume]
            if len(first) >= 5:  # likely candles
                return float(first[4])
            elif len(first) >= 2:  # likely events
                return float(first[1])
        elif isinstance(first, dict):
            # Try common keys (including "p" for Binance trade format)
            for k in ("close", "c", "price", "p"):
                if k in first:
                    return float(first[k])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert price to float in {path}: {e}")

    raise ValueError(f"Cannot extract price from first element in {path}: {first}")


# -------------------- Helpers --------------------
def strify_pool(pool: dict) -> dict:
    """Convert pool values to strings for JSON.

    - Lists are treated as integer arrays and stringified element-wise.
    - Integers are stringified as ints.
    - Floats are preserved as decimal strings (no int cast), e.g., donation_apy=0.05 -> "0.05".
    """
    out = {}
    for k, v in pool.items():
        if isinstance(v, list):
            out[k] = [str(int(x)) for x in v]
        elif isinstance(v, float):
            out[k] = str(v)
        else:
            out[k] = str(int(v))
    return out
