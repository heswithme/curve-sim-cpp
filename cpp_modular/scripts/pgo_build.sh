#!/usr/bin/env bash
set -euo pipefail

# Safe PGO build pipeline for arb_harness.
#
# The default output directories live under /tmp and the normal
# cpp_modular/build tree is left untouched. Use the printed --harness-exe path
# with python/arb_sim/arb_sim.py or python/arb_sim/thread_sweep.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

TARGET="arb_harness"
POOLS=""
CANDLES=""
OUT_DIR="/tmp/cpp_modular_pgo_use"
GEN_DIR="/tmp/cpp_modular_pgo_gen"
PROFILE_DIR="/tmp/arb_pgo_profile"
THREADS="12"
DUSTSWAPFREQ="3600"
START_TIME=""
N_CANDLES=""
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)"

usage() {
    cat <<EOF
Usage: $0 --pools POOLS.json --candles CANDLES.json [options]

Options:
  --target NAME           CMake target to build (default: arb_harness)
  --out-dir DIR           Profile-use build dir (default: /tmp/cpp_modular_pgo_use)
  --gen-dir DIR           Instrumented build dir (default: /tmp/cpp_modular_pgo_gen)
  --profile-dir DIR       Profile raw-data dir (default: /tmp/arb_pgo_profile)
  --threads N             Harness threads for training (default: 12)
  --dustswapfreq S        Idle tick cadence for training (default: 3600)
  --start-time TS         Optional harness --start-time
  --n-candles N           Optional harness --n-candles
  --jobs N                Build parallelism (default: host CPU count)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --pools) POOLS="$2"; shift 2 ;;
        --candles) CANDLES="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --gen-dir) GEN_DIR="$2"; shift 2 ;;
        --profile-dir) PROFILE_DIR="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --dustswapfreq) DUSTSWAPFREQ="$2"; shift 2 ;;
        --start-time) START_TIME="$2"; shift 2 ;;
        --n-candles) N_CANDLES="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$POOLS" || -z "$CANDLES" ]]; then
    usage >&2
    exit 2
fi

abs_dir_path() {
    local path="$1"
    local dir
    local base
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    mkdir -p "$dir"
    (cd "$dir" && printf "%s/%s\n" "$(pwd -P)" "$base")
}

NORMAL_BUILD_DIR="$(abs_dir_path "$PROJECT_DIR/build")"
GEN_DIR="$(abs_dir_path "$GEN_DIR")"
OUT_DIR="$(abs_dir_path "$OUT_DIR")"
PROFILE_DIR="$(abs_dir_path "$PROFILE_DIR")"

if [[ "$GEN_DIR" == "$NORMAL_BUILD_DIR" || "$OUT_DIR" == "$NORMAL_BUILD_DIR" || "$PROFILE_DIR" == "$NORMAL_BUILD_DIR" ]]; then
    echo "Error: PGO script refuses to overwrite normal build dir: $NORMAL_BUILD_DIR" >&2
    exit 2
fi

rm -rf "$GEN_DIR" "$OUT_DIR" "$PROFILE_DIR"
mkdir -p "$GEN_DIR" "$OUT_DIR" "$PROFILE_DIR"

PROFILE_PATTERN="$PROFILE_DIR/arb_%p.profraw"
PROFILE_DATA="$GEN_DIR/merged.profdata"

echo "=== PGO Build Pipeline ==="
echo "Project:      $PROJECT_DIR"
echo "Target:       $TARGET"
echo "Pools:        $POOLS"
echo "Candles:      $CANDLES"
echo "Threads:      $THREADS"
echo "Dust cadence: $DUSTSWAPFREQ"
if [[ -n "$START_TIME" ]]; then echo "Start time:   $START_TIME"; fi
if [[ -n "$N_CANDLES" ]]; then echo "N candles:    $N_CANDLES"; fi
echo "Output dir:   $OUT_DIR"
echo

echo "=== Step 1/4: Building instrumented binary ==="
cmake -S "$PROJECT_DIR" -B "$GEN_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fprofile-instr-generate=$PROFILE_PATTERN" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate=$PROFILE_PATTERN"
cmake --build "$GEN_DIR" --target "$TARGET" -j "$JOBS"

TRAINING_OUT="/tmp/pgo_training_${TARGET}.json"
TRAIN_CMD=(
    "$GEN_DIR/$TARGET"
    "$POOLS"
    "$CANDLES"
    "$TRAINING_OUT"
    --threads "$THREADS"
    --dustswapfreq "$DUSTSWAPFREQ"
    --disable-slippage-probes
)
if [[ -n "$START_TIME" ]]; then
    TRAIN_CMD+=(--start-time "$START_TIME")
fi
if [[ -n "$N_CANDLES" ]]; then
    TRAIN_CMD+=(--n-candles "$N_CANDLES")
fi

echo
echo "=== Step 2/4: Training run ==="
LLVM_PROFILE_FILE="$PROFILE_PATTERN" "${TRAIN_CMD[@]}"

PROFRAW_COUNT="$(find "$PROFILE_DIR" -name '*.profraw' -type f | wc -l | tr -d ' ')"
if [[ "$PROFRAW_COUNT" == "0" ]]; then
    echo "Error: no profile data generated in $PROFILE_DIR" >&2
    exit 1
fi
echo "Generated $PROFRAW_COUNT profile file(s)"

echo
echo "=== Step 3/4: Merging profiles ==="
PROFDATA="$(xcrun --find llvm-profdata 2>/dev/null || command -v llvm-profdata 2>/dev/null || true)"
if [[ -z "$PROFDATA" ]]; then
    echo "Error: llvm-profdata not found" >&2
    exit 1
fi
"$PROFDATA" merge -output="$PROFILE_DATA" "$PROFILE_DIR"/*.profraw

echo
echo "=== Step 4/4: Building profile-use binary ==="
cmake -S "$PROJECT_DIR" -B "$OUT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fprofile-instr-use=$PROFILE_DATA" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-use=$PROFILE_DATA"
cmake --build "$OUT_DIR" --target "$TARGET" -j "$JOBS"

echo
echo "=== PGO Build Complete ==="
echo "Binary: $OUT_DIR/$TARGET"
echo
echo "Use with:"
echo "  uv --project python run python python/arb_sim/arb_sim.py --harness-exe $OUT_DIR/$TARGET --skip-build --real double --dustswapfreq $DUSTSWAPFREQ -n $THREADS --disable-slippage-probes"
