import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CLUSTER_ROOT = ROOT / "python" / "arb_sim" / "cluster_orchestration"
sys.path.insert(0, str(CLUSTER_ROOT))

import run as cluster_run  # noqa: E402
import build as cluster_build  # noqa: E402
import collect as cluster_collect  # noqa: E402
import distribute as cluster_distribute  # noqa: E402


def test_block_cyclic_assignment_covers_every_pool_once() -> None:
    ranges_by_blade = cluster_distribute.compute_block_cyclic_ranges(
        n_pools=37,
        n_blades=4,
        chunk_size=5,
    )

    covered = [
        pool_index
        for ranges in ranges_by_blade
        for start, end in ranges
        for pool_index in range(start, end)
    ]
    counts = [cluster_distribute.ranges_pool_count(ranges) for ranges in ranges_by_blade]

    assert sorted(covered) == list(range(37))
    assert len(covered) == len(set(covered))
    assert max(counts) - min(counts) <= 5
    assert ranges_by_blade[0][:2] == [(0, 5), (20, 25)]


def test_extract_grid_values_supports_nested_policy_axes() -> None:
    run = {
        "params": {
            "pool": {
                "A": "10000",
                "policy": {"fee_bps": 25.0},
                "donation_apy": "0.2",
            }
        }
    }
    grid = {
        "x1": {"name": "A"},
        "x2": {"name": "policy.fee_bps"},
        "x3": {"name": "donation_apy"},
    }

    assert cluster_collect.extract_grid_values(run, grid) == {
        "A": 10000.0,
        "policy.fee_bps": 25.0,
        "donation_apy": 0.2,
    }


def test_cluster_cmake_template_links_pool_config_source() -> None:
    assert "src/pool_config_source.cpp" in cluster_build.CMAKE_TEMPLATE


def test_run_blade_job_forwards_start_time(monkeypatch) -> None:
    commands: list[str] = []

    def fake_run_ssh(blade: str, command: str, timeout: int = 0) -> subprocess.CompletedProcess:
        commands.append(command)
        if command.startswith("test -f "):
            return subprocess.CompletedProcess(["ssh"], 0, stdout="123\n", stderr="")
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    monkeypatch.setattr(cluster_run, "run_ssh", fake_run_ssh)

    result = cluster_run.run_blade_job(
        blade="blade-test",
        remote_pools="/remote/pools.json",
        remote_candles="/remote/candles.json",
        job_id="unit",
        pool_start=0,
        pool_end=8,
        threads=4,
        dustswap_freq=3600,
        candle_filter=None,
        start_time="1709638320",
        disable_slippage_probes=True,
        quiet_harness=True,
        retries=1,
    )

    assert result.success
    harness_commands = [command for command in commands if "/remote/pools.json" in command]
    assert len(harness_commands) == 1
    assert "--start-time 1709638320" in harness_commands[0]
    assert "--disable-slippage-probes" in harness_commands[0]
    assert "--quiet" in harness_commands[0]


def test_run_blade_job_forwards_pool_ranges(monkeypatch) -> None:
    commands: list[str] = []

    def fake_run_ssh(blade: str, command: str, timeout: int = 0) -> subprocess.CompletedProcess:
        commands.append(command)
        if command.startswith("test -f "):
            return subprocess.CompletedProcess(["ssh"], 0, stdout="123\n", stderr="")
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    monkeypatch.setattr(cluster_run, "run_ssh", fake_run_ssh)

    result = cluster_run.run_blade_job(
        blade="blade-test",
        remote_pools="/remote/pools.json",
        remote_candles="/remote/candles.json",
        job_id="unit",
        pool_start=0,
        pool_end=8,
        threads=4,
        dustswap_freq=3600,
        candle_filter=None,
        n_pools=4,
        pool_ranges_remote="/remote/ranges.txt",
        range_count=2,
        retries=1,
    )

    assert result.success
    harness_commands = [command for command in commands if "/remote/pools.json" in command]
    assert len(harness_commands) == 1
    assert "--pool-ranges /remote/ranges.txt" in harness_commands[0]
    assert "--pool-start" not in harness_commands[0]
    assert "--pool-end" not in harness_commands[0]


def test_collect_preserves_fast_run_flags_in_metadata(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "job_id": "unit",
        "local_job_dir": str(tmp_path),
        "remote_candles": "/remote/candles.json",
        "grid": {},
        "config": {
            "dustswap_freq": 3600,
            "candle_filter": 99.0,
            "start_time": "1709638320",
            "disable_slippage_probes": True,
            "quiet_harness": True,
        },
        "run_status": {
            "blade-test": {
                "success": True,
                "remote_output": "/remote/result.json",
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fake_download(_remote_path: str, local_path: Path, _blade: str, **_kwargs) -> bool:
        local_path.write_text(json.dumps({"runs": [{"result": {"vp": 1.0, "apy": 0.1}}]}))
        return True

    monkeypatch.setattr(cluster_collect, "rsync_from_cluster", fake_download)

    output_path = tmp_path / "merged.json"
    merged = cluster_collect.collect(manifest_path, output_path)

    assert merged is not None
    assert merged["metadata"]["harness_args"]["disable_slippage_probes"] is True
    assert merged["metadata"]["harness_args"]["quiet_harness"] is True


def test_collect_npz_shards_writes_merged_root_npz(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "job_id": "unit",
        "local_job_dir": str(tmp_path),
        "remote_candles": "/remote/candles.json",
        "grid": {"x1": {"name": "A", "values": [10000, 20000]}},
        "base_pool": {"A": "10000"},
        "base_costs": {"arb_fee_bps": 2},
        "config": {
            "dustswap_freq": 3600,
            "start_time": "1704067200",
            "disable_slippage_probes": True,
            "quiet_harness": True,
        },
        "run_status": {
            "blade-a": {
                "success": True,
                "remote_output": "/remote/result_blade_a",
            },
            "blade-b": {
                "success": True,
                "remote_output": "/remote/result_blade_b",
            }
        },
        "total_pools": 4,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fake_download(
        _remote_path: str, local_path: Path, _blade: str, **kwargs
    ) -> bool:
        assert kwargs.get("remote_is_dir") is True
        local_path.mkdir(parents=True)
        start = 0 if local_path.name == "result_blade-a" else 2
        values = np.array([1.0, 1.1]) if start == 0 else np.array([1.2, 1.3])
        apys = np.array([0.1, 0.2]) if start == 0 else np.array([0.3, 0.4])
        (local_path / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "arb_npz_v1",
                    "n_pools": 2,
                    "pool_start": start,
                    "pool_end": start + 2,
                    "metadata": {"n_pools": 2, "pool_start": start, "pool_end": start + 2},
                    "metrics_schema": [
                        {"name": "vp", "dtype": "float64"},
                        {"name": "apy", "dtype": "float64"},
                    ],
                }
            )
        )
        np.savez(local_path / "metrics.npz", vp=values, apy=apys)
        return True

    monkeypatch.setattr(cluster_collect, "rsync_from_cluster", fake_download)

    output_path = tmp_path / "merged_npz"
    merged = cluster_collect.collect(manifest_path, output_path)

    assert merged is not None
    assert (output_path / "manifest.json").exists()
    assert (output_path / "metrics.npz").exists()
    assert not (output_path / "shards").exists()
    assert "_shards" not in [p.name for p in output_path.iterdir()]
    assert merged["format"] == "arb_npz_v1"
    assert merged["metadata"]["start_time"] == "1704067200"
    assert merged["metadata"]["harness_args"]["start_time"] == "1704067200"
    assert merged["metadata"]["grid"]["x1"]["name"] == "A"
    assert merged["metadata"]["base_pool"] == {"A": "10000"}
    assert "shards" not in merged
    assert merged["summary"]["total_runs"] == 4
    with np.load(output_path / "metrics.npz") as arrays:
        assert arrays["vp"].tolist() == [1.0, 1.1, 1.2, 1.3]
        assert arrays["apy"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_collect_npz_shards_merges_by_pool_index(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "job_id": "unit",
        "local_job_dir": str(tmp_path),
        "remote_candles": "/remote/candles.json",
        "grid": {},
        "config": {},
        "run_status": {
            "blade-a": {"success": True, "remote_output": "/remote/result_blade_a"},
            "blade-b": {"success": True, "remote_output": "/remote/result_blade_b"},
        },
        "total_pools": 6,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fake_download(
        _remote_path: str, local_path: Path, _blade: str, **kwargs
    ) -> bool:
        assert kwargs.get("remote_is_dir") is True
        local_path.mkdir(parents=True)
        if local_path.name == "result_blade-a":
            pool_index = np.array([0, 2, 4], dtype=np.uint64)
            vp = np.array([10.0, 12.0, 14.0])
        else:
            pool_index = np.array([1, 3, 5], dtype=np.uint64)
            vp = np.array([11.0, 13.0, 15.0])
        (local_path / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "arb_npz_v1",
                    "n_pools": 3,
                    "pool_start": int(pool_index.min()),
                    "pool_end": int(pool_index.max()) + 1,
                    "metadata": {"n_pools": 3},
                    "metrics_schema": [
                        {"name": "vp", "dtype": "float64"},
                        {"name": "pool_index", "dtype": "uint64"},
                    ],
                }
            )
        )
        np.savez(local_path / "metrics.npz", vp=vp, pool_index=pool_index)
        return True

    monkeypatch.setattr(cluster_collect, "rsync_from_cluster", fake_download)

    output_path = tmp_path / "merged_npz"
    merged = cluster_collect.collect(manifest_path, output_path)

    assert merged is not None
    with np.load(output_path / "metrics.npz") as arrays:
        assert arrays["vp"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        assert arrays["pool_index"].tolist() == [0, 1, 2, 3, 4, 5]


def test_distribute_writes_fast_run_flags_to_manifest(monkeypatch, tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    candles.write_text("[]")
    pools = tmp_path / "pools.json"
    pools.write_text(
        json.dumps(
            {
                "meta": {"datafile": str(candles), "start_time": "1709638320"},
                "pools": [{"pool": {}, "costs": {}}],
            }
        )
    )

    monkeypatch.setattr(cluster_distribute, "__file__", str(tmp_path / "distribute.py"))
    monkeypatch.setattr(
        cluster_distribute,
        "run_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(cluster_distribute, "get_remote_file_size", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(cluster_distribute, "rsync_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_distribute, "rsync_many_to_cluster", lambda *_args, **_kwargs: None)

    manifest = cluster_distribute.distribute(
        pools_file=pools,
        blades=["blade-test"],
        job_id="unit",
        dustswap_freq=3600,
        candle_filter=98.5,
        disable_slippage_probes=True,
        quiet_harness=True,
        output_prefix="fast",
    )

    cfg = manifest["config"]
    assert cfg["dustswap_freq"] == 3600
    assert cfg["candle_filter"] == 98.5
    assert cfg["disable_slippage_probes"] is True
    assert cfg["quiet_harness"] is True
    assert cfg["output_prefix"] == "fast"
    assert manifest["assignment_mode"] == "block-cyclic"
    assert manifest["chunk_size"] == 2048
    assert manifest["blades"]["blade-test"]["ranges"] == [[0, 1]]
    assert "pool_ranges_remote" in manifest["blades"]["blade-test"]


def test_distribute_always_uploads_pools_even_when_remote_size_matches(
    monkeypatch, tmp_path: Path
) -> None:
    candles = tmp_path / "candles.json"
    candles.write_text("[]")
    pools = tmp_path / "pools.json"
    pools.write_text(
        json.dumps(
            {
                "meta": {"datafile": str(candles), "start_time": "1709638320"},
                "pools": [{"pool": {}, "costs": {}}],
            }
        )
    )
    uploads: list[str] = []

    monkeypatch.setattr(cluster_distribute, "__file__", str(tmp_path / "distribute.py"))
    monkeypatch.setattr(
        cluster_distribute,
        "run_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        cluster_distribute,
        "get_remote_file_size",
        lambda *_args, **_kwargs: pools.stat().st_size,
    )
    monkeypatch.setattr(
        cluster_distribute,
        "rsync_to_cluster",
        lambda local, *_args, **_kwargs: uploads.append(Path(local).name),
    )
    monkeypatch.setattr(cluster_distribute, "rsync_many_to_cluster", lambda *_args, **_kwargs: None)

    cluster_distribute.distribute(
        pools_file=pools,
        blades=["blade-test"],
        job_id="unit",
    )

    assert "pools.json" in uploads


def test_load_pools_counts_compact_grid(tmp_path: Path) -> None:
    candles = tmp_path / "candles.json"
    candles.write_text("[]")
    pools = tmp_path / "pools.json"
    pools.write_text(
        json.dumps(
            {
                "meta": {
                    "datafile": str(candles),
                    "base_pool": {},
                    "pool_count": 256,
                    "grid": {
                        "x1": {"name": "A", "values": list(range(16))},
                        "x2": {"name": "mid_fee", "values": list(range(16))},
                    },
                }
            }
        )
    )

    n_pools, meta = cluster_distribute.load_pools(pools)

    assert n_pools == 256
    assert meta["grid"]["x1"]["name"] == "A"
