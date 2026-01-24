#!/usr/bin/env python3
"""
Build the C++ harness on the cluster (once, shared via NFS).

Since all blades share /home/heswithme, we only need to:
1. Upload source to shared NFS
2. Build on any one blade
3. Binary is immediately available to all blades
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from config import (
    REMOTE_SRC,
    REMOTE_BUILD,
    LOCAL_CPP_DIR,
    SSH_USER,
    SSH_KEY,
    SSH_OPTIONS,
    HARNESS_BINARY,
    CLUSTER_CXX_FLAGS,
    NIX_BUILD_PACKAGES,
    BUILD_TIMEOUT,
    DEFAULT_BLADES,
    REMOTE_BASE,
)


CMAKE_TEMPLATE = """cmake_minimum_required(VERSION 3.14)
project(arb_harness_modular CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.75 REQUIRED COMPONENTS json)
find_package(Threads REQUIRED)

set(COMMON_SOURCES
    src/main.cpp
    src/events.cpp
    src/cli.cpp
)

set(COMMON_FLAGS -Wall -Wextra)
set(RELEASE_FLAGS {release_flags})
 
function(arb_target_defaults TARGET_NAME)
    target_include_directories(${{TARGET_NAME}} PRIVATE
        ${{CMAKE_CURRENT_SOURCE_DIR}}/include
        ${{Boost_INCLUDE_DIRS}}
    )
    target_compile_options(${{TARGET_NAME}} PRIVATE
        ${{COMMON_FLAGS}}
        $<$<CONFIG:Release>:${{RELEASE_FLAGS}}>
    )
    target_link_options(${{TARGET_NAME}} PRIVATE
        $<$<CONFIG:Release>:-flto>
    )
endfunction()

add_executable(arb_harness ${{COMMON_SOURCES}})
target_link_libraries(arb_harness Boost::json Threads::Threads)
arb_target_defaults(arb_harness)

add_executable(arb_harness_ld ${{COMMON_SOURCES}})
target_compile_definitions(arb_harness_ld PRIVATE ARB_MODE_LD)
target_link_libraries(arb_harness_ld Boost::json Threads::Threads)
arb_target_defaults(arb_harness_ld)
"""


def run_ssh(blade: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run command on blade via SSH."""
    cmd = ["ssh"] + SSH_OPTIONS + ["-i", str(SSH_KEY), f"{SSH_USER}@{blade}", command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_rsync(local_path: Path, remote_path: str, blade: str) -> None:
    """Sync only source files (include/ and src/) to cluster."""
    ssh_opts = f"ssh -i {SSH_KEY} " + " ".join(SSH_OPTIONS)

    # Only sync what we need: include/ and src/
    # Exclude all build artifacts, cmake cache, PGO builds, object files
    cmd = [
        "rsync",
        "-avz",
        "--delete",
        "-e",
        ssh_opts,
        "--include",
        "include/",
        "--include",
        "include/**",
        "--include",
        "src/",
        "--include",
        "src/**",
        "--exclude",
        "build/",
        "--exclude",
        "build_pgo/",
        "--exclude",
        "CMakeFiles/",
        "--exclude",
        "CMakeCache.txt",
        "--exclude",
        "cmake_install.cmake",
        "--exclude",
        "Makefile",
        "--exclude",
        ".git/",
        "--exclude",
        "*.o",
        "--exclude",
        "*.a",
        "--exclude",
        "scripts/",
        "--exclude",
        "CMakeLists.txt",  # We generate our own optimized one
        str(local_path) + "/",
        f"{SSH_USER}@{blade}:{remote_path}/",
    ]
    print(f"Syncing source (include/, src/) -> {remote_path}")
    subprocess.run(cmd, check=True, timeout=120)


def setup_dirs(blade: str) -> None:
    """Create directory structure on shared NFS."""
    dirs = [
        REMOTE_BASE,
        REMOTE_SRC,
        REMOTE_BUILD,
        REMOTE_BASE / "data",
        REMOTE_BASE / "jobs",
        REMOTE_BASE / "results",
    ]
    run_ssh(blade, f"mkdir -p {' '.join(str(d) for d in dirs)}")


def upload_source(blade: str) -> None:
    """Upload C++ source to shared NFS."""
    run_rsync(LOCAL_CPP_DIR, str(REMOTE_SRC), blade)


def compile_harness(blade: str) -> bool:
    """Compile harness using nix-shell."""
    nix_packages = " ".join(NIX_BUILD_PACKAGES)

    cmake_content = CMAKE_TEMPLATE.format(release_flags=" ".join(CLUSTER_CXX_FLAGS))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(cmake_content)
        temp_cmake = Path(f.name)

    try:
        scp_cmd = [
            "scp",
            "-i",
            str(SSH_KEY),
            "-o",
            "StrictHostKeyChecking=accept-new",
            str(temp_cmake),
            f"{SSH_USER}@{blade}:{REMOTE_SRC}/CMakeLists.txt",
        ]
        subprocess.run(scp_cmd, check=True, timeout=60)
    finally:
        temp_cmake.unlink()

    build_script = f"""
cd {REMOTE_SRC}
rm -rf {REMOTE_BUILD}
mkdir -p {REMOTE_BUILD}
cd {REMOTE_BUILD}
cmake {REMOTE_SRC} -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) arb_harness_ld arb_harness
"""

    full_cmd = f"nix-shell -p {nix_packages} --run '{build_script}'"

    print(f"Compiling on {blade} (this may take a few minutes)...")
    result = run_ssh(blade, full_cmd, timeout=BUILD_TIMEOUT)

    if result.returncode != 0:
        print(f"Build failed!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        return False

    print(f"Build output (last 500 chars): {result.stdout[-500:]}")
    return True


def verify_binary(blade: str) -> bool:
    """Check that binary exists and runs."""
    binary = REMOTE_BUILD / HARNESS_BINARY

    result = run_ssh(blade, f"test -x {binary} && {binary} 2>&1 | head -3")

    if result.returncode != 0 and "usage" not in result.stdout.lower():
        print(f"Binary verification failed: {result.stderr}")
        return False

    print(f"Binary OK: {result.stdout.strip()}")
    return True


def build(blade: str = None) -> bool:
    """
    Full build process. Uploads source and compiles once.
    Binary is available to all blades via shared NFS.
    """
    if blade is None:
        blade = DEFAULT_BLADES[0]

    print(f"\n{'=' * 60}")
    print(f"Building on {blade} (shared NFS - all blades will have access)")
    print(f"{'=' * 60}\n")

    try:
        print("Setting up directories...")
        setup_dirs(blade)

        print("Uploading source...")
        upload_source(blade)

        print("Compiling...")
        if not compile_harness(blade):
            return False

        print("Verifying binary...")
        if not verify_binary(blade):
            return False

        print(f"\nBuild successful! Binary at: {REMOTE_BUILD}/{HARNESS_BINARY}")
        return True

    except Exception as e:
        print(f"Build failed: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build harness on cluster")
    parser.add_argument("--blade", default=DEFAULT_BLADES[0], help="Blade to build on")
    args = parser.parse_args()

    success = build(args.blade)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
