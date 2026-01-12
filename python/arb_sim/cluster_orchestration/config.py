"""
Cluster configuration for NixOS HPC cluster with shared NFS home.

Architecture:
- All blades share /home/heswithme via NFS
- Build once, binary visible to all blades
- Upload data once, all blades read from same location
- Each blade writes results to shared location

Blades:
- A-series: blade-a5 through blade-a10 (a1-a4 reserved, a9 may be down)
- B-series: blade-b1 through blade-b10 (b3 may be down)
- Each blade: 64 physical cores (128 logical), 3.9TB RAM
- CPU: Intel Xeon Platinum 8352Y (Ice Lake Server)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# =============================================================================
# SSH Configuration
# =============================================================================

SSH_USER = "heswithme"
SSH_KEY = Path.home() / ".ssh" / "id_rsa2"

# ProxyJump configured in ~/.ssh/config:
#   Local -> cluster-jumphost -> gateway -> blade
# So we SSH directly to blade-* hostnames

SSH_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=60",
    "-o",
    "ServerAliveCountMax=3",
]

# =============================================================================
# Blade Configuration
# =============================================================================

# Blades with CORRECT /home/heswithme ownership (can read/write home dir)
BLADES_WORKING = [
    "blade-a1",
    "blade-a2",
    "blade-a3",
    "blade-a4",
    "blade-a5",
    "blade-a6",
    "blade-a7",
    "blade-a8",
    "blade-a10",
    "blade-b1",
    "blade-b2",
    "blade-b4",
    "blade-b5",
    "blade-b6",
    "blade-b7",
    "blade-b8",
    "blade-b9",
    "blade-b10",
]

# Blades with broken ownership (george/heswithme/michwill rotated)
# These need admin fix before they can access /home/heswithme
BLADES_BROKEN_PERMS = []

# Unreachable blades
BLADES_DOWN = ["blade-a9", "blade-b3"]

ALL_BLADES = BLADES_WORKING + BLADES_BROKEN_PERMS
DEFAULT_BLADES = BLADES_WORKING  # Use all reachable blades

# Per-blade resources
CORES_PER_BLADE = 128  # Logical cores (64 physical with HT)

# =============================================================================
# Remote Paths (Shared NFS)
# =============================================================================

REMOTE_HOME = Path("/home/heswithme")
REMOTE_BASE = REMOTE_HOME / "arb"

# Subdirectories (all shared across blades)
REMOTE_SRC = REMOTE_BASE / "cpp_modular"
REMOTE_BUILD = REMOTE_BASE / "build"
REMOTE_DATA = REMOTE_BASE / "data"
REMOTE_JOBS = REMOTE_BASE / "jobs"
REMOTE_RESULTS = REMOTE_BASE / "results"

# Binary name
HARNESS_BINARY = "arb_harness_ld"

# =============================================================================
# Local Paths
# =============================================================================

LOCAL_CLUSTER_DIR = Path(__file__).resolve().parent
LOCAL_PROJECT_ROOT = LOCAL_CLUSTER_DIR.parents[2]
LOCAL_CPP_DIR = LOCAL_PROJECT_ROOT / "cpp_modular"
LOCAL_RESULTS_DIR = LOCAL_CLUSTER_DIR / "results"

# =============================================================================
# Compiler Flags for Ice Lake Server
# =============================================================================

CLUSTER_CXX_FLAGS = [
    "-O3",
    "-march=icelake-server",
    "-mtune=icelake-server",
    "-flto",
    "-funroll-loops",
    "-DNDEBUG",
]

# =============================================================================
# Nix Shell Configuration
# =============================================================================

NIX_BUILD_PACKAGES = ["gcc", "cmake", "boost", "gnumake"]
NIX_SHELL_CMD = f"nix-shell -p {' '.join(NIX_BUILD_PACKAGES)} --run"

# =============================================================================
# Timeouts
# =============================================================================

BUILD_TIMEOUT = 600  # 10 minutes
JOB_TIMEOUT = 7200  # 2 hours per blade
COLLECT_TIMEOUT = 300  # 5 minutes

# =============================================================================
# Job Configuration
# =============================================================================


@dataclass
class JobConfig:
    """Configuration for a simulation job."""

    pools_file: Path
    candles_file: Path
    blades: List[str] = field(default_factory=lambda: DEFAULT_BLADES.copy())

    # Harness arguments
    threads_per_blade: int = CORES_PER_BLADE
    dustswap_freq: int = 600
    apy_period_days: float = 1.0
    apy_period_cap: int = 20
    candle_filter: Optional[float] = None

    # Output
    output_prefix: str = "cluster_sweep"

    def __post_init__(self):
        self.pools_file = Path(self.pools_file)
        self.candles_file = Path(self.candles_file)


# =============================================================================
# SSH Helpers
# =============================================================================


def ssh_cmd(blade: str, command: str) -> List[str]:
    """Build SSH command list."""
    cmd = ["ssh"]
    cmd.extend(SSH_OPTIONS)
    cmd.extend(["-i", str(SSH_KEY)])
    cmd.append(f"{SSH_USER}@{blade}")
    cmd.append(command)
    return cmd


def scp_cmd(local_path: Path, remote_path: Path, to_remote: bool = True) -> List[str]:
    """Build SCP command list. Uses first available blade."""
    cmd = ["scp", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new"]
    blade = DEFAULT_BLADES[0]

    if to_remote:
        cmd.extend([str(local_path), f"{SSH_USER}@{blade}:{remote_path}"])
    else:
        cmd.extend([f"{SSH_USER}@{blade}:{remote_path}", str(local_path)])

    return cmd


def rsync_cmd(local_path: Path, remote_path: Path, to_remote: bool = True) -> List[str]:
    """Build rsync command list."""
    ssh_opts = f"ssh -i {SSH_KEY} " + " ".join(SSH_OPTIONS)
    blade = DEFAULT_BLADES[0]

    cmd = ["rsync", "-avz", "--delete", "-e", ssh_opts]
    cmd.extend(["--exclude", "build/", "--exclude", ".git/", "--exclude", "*.o"])

    if to_remote:
        cmd.extend([str(local_path) + "/", f"{SSH_USER}@{blade}:{remote_path}/"])
    else:
        cmd.extend([f"{SSH_USER}@{blade}:{remote_path}/", str(local_path) + "/"])

    return cmd
