import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARB_SIM_ROOT = ROOT / "python" / "arb_sim"
sys.path.insert(0, str(ARB_SIM_ROOT))

import arb_sim  # noqa: E402


def test_parse_start_time_accepts_json_numeric_timestamp() -> None:
    assert arb_sim.parse_start_time(1_758_672_000) == 1_758_672_000


def test_runner_forwards_quiet_harness_flag(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(list(cmd))
        out_path = Path(cmd[3])
        out_path.write_text(json.dumps({"metadata": {}, "runs": []}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(arb_sim.subprocess, "run", fake_run)

    runner = arb_sim.ArbHarnessRunner(ROOT, exe_path="/bin/echo")
    out_path = tmp_path / "out.json"
    result = runner.run(
        tmp_path / "pools.json",
        tmp_path / "candles.json",
        out_path,
        threads=2,
        disable_slippage_probes=True,
        quiet_harness=True,
    )

    assert result == {"metadata": {}, "runs": []}
    assert commands
    cmd = commands[0]
    assert "--threads" in cmd
    assert "--disable-slippage-probes" in cmd
    assert "--quiet" in cmd
