"""
Cluster Orchestration for TwoCrypto Pool Sweeps

This package provides tools to distribute and run pool parameter sweeps
across a NixOS HPC cluster.

Modules:
    config      - Cluster configuration (hosts, paths, compiler flags)
    build       - Compile harness on cluster with optimal flags
    distribute  - Split pools across blades and upload data
    run         - Execute jobs on blades in parallel
    collect     - Download and merge results
    orchestrate - Main entry point combining all steps
    utils       - Utility functions (status, clean, kill)

Quick Start:
    # Full sweep
    python -m cluster_orchestration.orchestrate \\
        --pools path/to/pools.json \\
        --candles path/to/candles.json
    
    # Check cluster status
    python -m cluster_orchestration.utils status
"""

from .config import (
    DEFAULT_BLADES,
    BLADES_A_SERIES,
    BLADES_B_SERIES,
    JobConfig,
)

__version__ = "0.1.0"
