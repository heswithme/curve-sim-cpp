import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

from arb_run_npz import load_json_or_npz, load_npz_run, ordered_grid  # noqa: E402


def test_load_single_npz_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "arb_run"
    run_dir.mkdir()
    manifest = {
        "format": "arb_npz_v1",
        "metrics_file": "metrics.npz",
        "n_pools": 4,
        "metadata": {
            "n_pools": 4,
            "grid": {
                "x1": {"name": "A", "values": [10000, 20000]},
                "x2": {"name": "mid_fee", "values": [10, 20]},
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(run_dir / "metrics.npz", vp=np.array([1.0, 1.1, 1.2, 1.3]))

    loaded = load_json_or_npz(run_dir)
    npz_run = loaded["_npz_run"]

    assert ordered_grid(loaded["metadata"])[0] == ["A", "mid_fee"]
    assert npz_run.load_array("vp").tolist() == [1.0, 1.1, 1.2, 1.3]


def test_load_sharded_npz_run(tmp_path: Path) -> None:
    root = tmp_path / "cluster_run"
    shard0 = root / "shards" / "result_a"
    shard1 = root / "shards" / "result_b"
    shard0.mkdir(parents=True)
    shard1.mkdir(parents=True)
    np.savez(shard0 / "metrics.npz", vp=np.array([1.0, 1.1]))
    np.savez(shard1 / "metrics.npz", vp=np.array([1.2, 1.3]))
    manifest = {
        "format": "arb_npz_v1_sharded",
        "n_pools": 4,
        "metadata": {"n_pools": 4},
        "shards": [
            {"path": "shards/result_a", "pool_start": 0, "pool_end": 2},
            {"path": "shards/result_b", "pool_start": 2, "pool_end": 4},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    npz_run = load_npz_run(root)

    assert npz_run.load_array("vp").tolist() == [1.0, 1.1, 1.2, 1.3]


def test_load_sharded_npz_run_uses_pool_index_when_present(tmp_path: Path) -> None:
    root = tmp_path / "cluster_run"
    shard0 = root / "shards" / "result_a"
    shard1 = root / "shards" / "result_b"
    shard0.mkdir(parents=True)
    shard1.mkdir(parents=True)
    np.savez(
        shard0 / "metrics.npz",
        vp=np.array([1.0, 1.2]),
        pool_index=np.array([0, 2], dtype=np.uint64),
    )
    np.savez(
        shard1 / "metrics.npz",
        vp=np.array([1.1, 1.3]),
        pool_index=np.array([1, 3], dtype=np.uint64),
    )
    manifest = {
        "format": "arb_npz_v1_sharded",
        "n_pools": 4,
        "metadata": {"n_pools": 4},
        "shards": [
            {"path": "shards/result_a", "pool_start": 0, "pool_end": 3},
            {"path": "shards/result_b", "pool_start": 1, "pool_end": 4},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    npz_run = load_npz_run(root)

    assert npz_run.load_array("vp").tolist() == [1.0, 1.1, 1.2, 1.3]
