import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLUSTER_ROOT = ROOT / "python" / "arb_sim" / "cluster_orchestration"
sys.path.insert(0, str(CLUSTER_ROOT))

import run as cluster_run  # noqa: E402
import collect as cluster_collect  # noqa: E402
import distribute as cluster_distribute  # noqa: E402


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
        dustswap_freq=600,
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


def test_collect_preserves_fast_run_flags_in_metadata(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "job_id": "unit",
        "local_job_dir": str(tmp_path),
        "remote_candles": "/remote/candles.json",
        "grid": {},
        "config": {
            "dustswap_freq": 600,
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

    manifest = cluster_distribute.distribute(
        pools_file=pools,
        blades=["blade-test"],
        job_id="unit",
        dustswap_freq=600,
        candle_filter=98.5,
        disable_slippage_probes=True,
        quiet_harness=True,
        output_prefix="fast",
    )

    cfg = manifest["config"]
    assert cfg["dustswap_freq"] == 600
    assert cfg["candle_filter"] == 98.5
    assert cfg["disable_slippage_probes"] is True
    assert cfg["quiet_harness"] is True
    assert cfg["output_prefix"] == "fast"
