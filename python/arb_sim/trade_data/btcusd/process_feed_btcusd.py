#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


name = "btcusd-2023-2026"
window_size = 10
folder = Path(__file__).resolve().parent
processed_data = []
touched_rows = 0
field_counts = [0, 0, 0, 0]

with open(folder / f"{name}.json", "r") as f:
    data = json.load(f)

for row in data:
    r = [int(row[0])] + [float(x) for x in row[1:6]]
    before = r[:]

    i = len(processed_data)
    window = data[max(i - window_size // 2, 0): min(i + window_size // 2 + 1, len(data))]
    window = np.array([[float(x) for x in d[1:5]] for d in window if d[0] != r[0]])
    min_window = window.min()
    max_window = window.max()

    for j in range(1, 5):
        if r[j] > max_window:
            r[j] = max_window
        if r[j] < min_window:
            r[j] = min_window

    if r[1:5] != before[1:5]:
        touched_rows += 1
        for j in range(1, 5):
            if r[j] != before[j]:
                field_counts[j - 1] += 1

    processed_data.append(r)

out = folder / f"{name}-filtered.json"
with open(out, "w") as f:
    json.dump(processed_data, f)

print(f"read {len(data)} rows")
print(f"wrote {len(processed_data)} rows -> {out}")
print(f"touched_rows={touched_rows}")
print(f"field_counts_OHLC={field_counts}")
