from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def is_npz_run_path(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists()


def load_json_or_npz(path: Path) -> dict[str, Any]:
    if is_npz_run_path(path):
        run = load_npz_run(path)
        return {"metadata": run.metadata, "_npz_run": run}

    try:
        import orjson
    except Exception:
        orjson = None

    if orjson is not None:
        return orjson.loads(path.read_bytes())
    with path.open("r") as f:
        return json.load(f)


def ordered_grid(metadata: dict[str, Any]) -> tuple[list[str], list[list[float]], list[str]]:
    grid = metadata.get("grid", {})
    if not isinstance(grid, dict):
        return [], [], []
    keys = sorted(
        (k for k in grid if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )
    names: list[str] = []
    values: list[list[float]] = []
    for key in keys:
        axis = grid.get(key)
        if not isinstance(axis, dict) or "name" not in axis or "values" not in axis:
            continue
        names.append(str(axis["name"]))
        values.append([float(v) for v in axis["values"]])
    return names, values, keys[: len(names)]


def grid_shape(metadata: dict[str, Any], n_pools: int) -> tuple[int, ...]:
    _, values, _ = ordered_grid(metadata)
    if not values:
        return (n_pools,)
    shape = tuple(len(axis) for axis in values)
    count = 1
    for n in shape:
        count *= n
    if count != n_pools:
        return (n_pools,)
    return shape


@dataclass(frozen=True)
class NpzShard:
    root: Path
    metrics_path: Path
    pool_start: int
    pool_end: int
    n_pools: int


class NpzRun:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.manifest = manifest
        self.metadata = manifest.get("metadata", {})
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        self.n_pools = int(
            manifest.get(
                "n_pools",
                self.metadata.get("n_pools", self.metadata.get("total_pools", 0)),
            )
        )
        self.shards = self._parse_shards()

    def _parse_shards(self) -> list[NpzShard]:
        raw_shards = self.manifest.get("shards")
        if isinstance(raw_shards, list):
            shards: list[NpzShard] = []
            for item in raw_shards:
                if not isinstance(item, dict):
                    continue
                rel = item.get("path")
                if rel is None:
                    continue
                shard_root = Path(str(rel))
                if not shard_root.is_absolute():
                    shard_root = self.root / shard_root
                metrics_file = item.get("metrics_file", "metrics.npz")
                metrics_path = Path(str(metrics_file))
                if not metrics_path.is_absolute():
                    metrics_path = shard_root / metrics_path
                start = int(item.get("pool_start", 0))
                end = int(item.get("pool_end", start + int(item.get("n_pools", 0))))
                shards.append(
                    NpzShard(
                        root=shard_root,
                        metrics_path=metrics_path,
                        pool_start=start,
                        pool_end=end,
                        n_pools=max(0, end - start),
                    )
                )
            return shards

        metrics_file = self.manifest.get("metrics_file", "metrics.npz")
        metrics_path = Path(str(metrics_file))
        if not metrics_path.is_absolute():
            metrics_path = self.root / metrics_path
        start = int(self.manifest.get("pool_start", self.metadata.get("pool_start", 0)))
        end_raw = self.manifest.get("pool_end", self.metadata.get("pool_end"))
        end = int(end_raw) if end_raw is not None else start + self.n_pools
        return [
            NpzShard(
                root=self.root,
                metrics_path=metrics_path,
                pool_start=start,
                pool_end=end,
                n_pools=max(0, end - start),
            )
        ]

    def metric_names(self) -> set[str]:
        names: set[str] = set()
        for shard in self.shards:
            with np.load(shard.metrics_path) as data:
                names.update(data.files)
        return names

    def load_array(self, name: str) -> np.ndarray:
        if len(self.shards) == 1 and self.shards[0].pool_start == 0 and self.shards[0].n_pools == self.n_pools:
            with np.load(self.shards[0].metrics_path) as data:
                if name not in data.files:
                    raise KeyError(name)
                return np.asarray(data[name])

        out: np.ndarray | None = None
        for shard in self.shards:
            with np.load(shard.metrics_path) as data:
                if name not in data.files:
                    continue
                arr = np.asarray(data[name])
                pool_index = np.asarray(data["pool_index"]) if "pool_index" in data.files else None
            if out is None:
                fill = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
                out = np.full(self.n_pools, fill, dtype=arr.dtype)
            if pool_index is not None:
                out[pool_index.astype(np.intp, copy=False)] = arr
            else:
                rel_start = shard.pool_start
                rel_end = rel_start + len(arr)
                out[rel_start:rel_end] = arr
        if out is None:
            raise KeyError(name)
        return out


def load_npz_run(path: Path) -> NpzRun:
    manifest_path = path / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    return NpzRun(path, manifest)
