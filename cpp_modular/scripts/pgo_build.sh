#!/bin/bash
set -e

# PGO (Profile-Guided Optimization) build script
# Usage: ./scripts/pgo_build.sh
#
# Trains on the standard arb_sim workload for optimal production performance.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"
BUILD_DIR="$PROJECT_DIR/build_pgo"
PROFILE_DIR="$BUILD_DIR/profiles"
FINAL_BUILD_DIR="$PROJECT_DIR/build"
ARB_SIM_DIR="$ROOT_DIR/python/arb_sim"

echo "=== PGO Build Pipeline ==="
echo "Project: $PROJECT_DIR"
echo ""

# Clean previous PGO artifacts
rm -rf "$BUILD_DIR" "$PROFILE_DIR"
mkdir -p "$BUILD_DIR" "$PROFILE_DIR"

# Step 1: Instrumented build
echo "=== Step 1/4: Building instrumented binary ==="
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fprofile-generate=$PROFILE_DIR"

cmake --build "$BUILD_DIR" --config Release -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo ""
echo "=== Step 2/4: Training run ==="

# Generate pools for training
echo "Generating pools..."
cd "$ARB_SIM_DIR"
uv run python generate_pools_generic.py 2>/dev/null || python3 generate_pools_generic.py

# Find training data
TRAINING_DATA=$(find "$ARB_SIM_DIR/trade_data" -name "*.json" -type f 2>/dev/null | head -1)
if [[ -z "$TRAINING_DATA" ]]; then
    echo "Error: No training data found in $ARB_SIM_DIR/trade_data"
    exit 1
fi
echo "Using training data: $TRAINING_DATA"

# Training run with production-like workload
# This matches: uv run python/arb_sim/arb_sim.py --real double --dustswapfreq 600 --apy-period-days 1 --apy-period-cap 30 -n 10
echo "Running training workload (double precision)..."
ARB_HARNESS="$BUILD_DIR/arb_harness"
POOLS_FILE="$ARB_SIM_DIR/run_data/pool_config.json"

if [[ -f "$POOLS_FILE" ]]; then
    # Run the actual workload that will be used in production
    "$ARB_HARNESS" "$POOLS_FILE" "$TRAINING_DATA" /tmp/pgo_training_out.json \
        --dustswapfreq 600 \
        --apy-period-days 1 \
        --apy-period-cap 30 \
        -n 10 \
        2>&1 | tail -10 || true
    
    # Also train float and long double variants
    echo "Training float variant..."
    "$BUILD_DIR/arb_harness_f" "$POOLS_FILE" "$TRAINING_DATA" /tmp/pgo_training_f.json \
        --dustswapfreq 600 -n 4 --n-candles 50000 2>/dev/null || true
    
    echo "Training long double variant..."
    "$BUILD_DIR/arb_harness_ld" "$POOLS_FILE" "$TRAINING_DATA" /tmp/pgo_training_ld.json \
        --dustswapfreq 600 -n 4 --n-candles 50000 2>/dev/null || true
else
    echo "Warning: pools_grid.json not found, using fallback training"
    "$ARB_HARNESS" --help 2>/dev/null || true
fi

# Cleanup training outputs
rm -f /tmp/pgo_training_*.json

# Check if profile data was generated
PROFRAW_COUNT=$(find "$PROFILE_DIR" -name "*.profraw" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$PROFRAW_COUNT" -eq 0 ]]; then
    echo "Error: No profile data generated. Check if training run completed."
    exit 1
fi
echo "Generated $PROFRAW_COUNT profile file(s)"

echo ""
echo "=== Step 3/4: Merging profiles ==="

# Find llvm-profdata (macOS location varies)
PROFDATA=$(xcrun -find llvm-profdata 2>/dev/null || which llvm-profdata 2>/dev/null || echo "")
if [[ -z "$PROFDATA" ]]; then
    echo "Error: llvm-profdata not found. Install Xcode command line tools."
    exit 1
fi

"$PROFDATA" merge -output="$BUILD_DIR/merged.profdata" "$PROFILE_DIR"/*.profraw
echo "Merged profile: $BUILD_DIR/merged.profdata"

echo ""
echo "=== Step 4/4: Building optimized binary ==="

# Clean and rebuild with profile data
rm -rf "$FINAL_BUILD_DIR"
cmake -S "$PROJECT_DIR" -B "$FINAL_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fprofile-use=$BUILD_DIR/merged.profdata"

cmake --build "$FINAL_BUILD_DIR" --config Release -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo ""
echo "=== PGO Build Complete ==="
echo "Optimized binaries in: $FINAL_BUILD_DIR"
echo ""
echo "Binaries:"
ls -la "$FINAL_BUILD_DIR"/arb_harness* "$FINAL_BUILD_DIR"/benchmark_harness* 2>/dev/null | grep -v '\.o$'
