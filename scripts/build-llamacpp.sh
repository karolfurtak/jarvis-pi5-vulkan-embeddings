#!/usr/bin/env bash
#
# build-llamacpp.sh
#
# Builds two llama.cpp `llama-embedding` binaries on Raspberry Pi 5 / Debian 13
# (Trixie), aarch64: one CPU-only (ARM-native, Cortex-A76 + dotprod), one with
# the Vulkan backend enabled (targets the V3D GPU via Mesa's V3DV driver).
#
# Why two separate builds instead of one binary with a runtime flag: llama.cpp
# compiles backends in at build time (CMake -DGGML_VULKAN=ON/OFF). The CPU
# build is the baseline this repository benchmarks against; the Vulkan build
# is the GPU path under test. Both link against the SAME unmodified GGUF
# (GPT-Generated Unified Format) model file fetched by fetch-model.sh — the
# backend, not the weights, is what differs between the two binaries.
#
# Tested on: Debian 13 (Trixie) aarch64, kernel 6.18.39+rpt-rpi-2712,
# gcc 14.2.0, cmake 3.31.6, Raspberry Pi 5 (BCM2712).
#
# Usage:
#   ./build-llamacpp.sh [--jobs N] [--dir DIR]
#
# --jobs N   Parallel compile jobs (default: 2). Deliberately conservative —
#            this is a 4-core board that may be running other unrelated
#            services; leaving 2 cores free avoids starving them. Raise to
#            -j4 only if the board is otherwise idle.
# --dir DIR  Working directory (default: ~/gpu-bench).
#
# Exit codes: non-zero on any failure (missing package, configure error,
# compile/link failure). Does not continue silently past an error
# (set -euo pipefail).

set -euo pipefail

JOBS=2
BENCH_DIR="$HOME/gpu-bench"

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        --dir) BENCH_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_DIR="$BENCH_DIR/llama.cpp"
# Pinned to the exact commit this repository's results/results.csv was
# measured against. llama.cpp moves fast; override with
# LLAMA_CPP_COMMIT=HEAD (or any other ref) to build current upstream instead —
# useful for checking whether upstream has fixed the pinned-memory allocation
# issue described in README.md, but numbers won't be directly comparable to
# results/ if you do.
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-153d324bcf86d220b235ca010eeb11213f32b5d1}"

echo "== Installing build + Vulkan dependencies (apt, requires sudo) =="
# vulkan-tools, libvulkan-dev: vulkaninfo + Vulkan headers/loader for the build
# clinfo, mesa-opencl-icd: OpenCL (rusticl) inspection, not required for the
#   build itself but used by this repo's setup-check step
# ninja-build: faster builds than plain make
# glslang-tools: provides glslangValidator (NOT glslc — see below)
# glslc: the actual GLSL->SPIR-V compiler ggml-vulkan's CMake looks for
#   (separate package from glslang-tools; easy to miss — CMake fails late,
#   at the ggml-vulkan step, with "Could NOT find Vulkan (missing: glslc)")
# spirv-headers: required by ggml-vulkan's CMakeLists.txt (find_package
#   (SPIRV-Headers)); NOT pulled in automatically by libvulkan-dev on
#   Debian 13 — CMake fails with "Could not find ... SPIRV-HeadersConfig.cmake"
#   if this is missing, even after glslc is installed
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake ninja-build git \
    vulkan-tools libvulkan-dev clinfo mesa-opencl-icd \
    glslang-tools glslc spirv-headers

echo "== Fetching llama.cpp @ $LLAMA_CPP_COMMIT =="
mkdir -p "$BENCH_DIR"
if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
    git init -q "$LLAMA_CPP_DIR"
    git -C "$LLAMA_CPP_DIR" remote add origin "$LLAMA_CPP_URL"
fi
if [ "$LLAMA_CPP_COMMIT" = "HEAD" ]; then
    git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin
    git -C "$LLAMA_CPP_DIR" checkout -q FETCH_HEAD
else
    git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_COMMIT"
    git -C "$LLAMA_CPP_DIR" checkout -q FETCH_HEAD
fi
echo "  commit: $(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"

echo "== Building CPU variant (ARM-native, no GPU backend) =="
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build-cpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=OFF \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$LLAMA_CPP_DIR/build-cpu" -j"$JOBS" --target llama-embedding

echo "== Building Vulkan variant (GPU backend, targets V3D via V3DV) =="
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build-vulkan" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$LLAMA_CPP_DIR/build-vulkan" -j"$JOBS" --target llama-embedding

echo "== Done =="
echo "  CPU binary:    $LLAMA_CPP_DIR/build-cpu/bin/llama-embedding"
echo "  Vulkan binary: $LLAMA_CPP_DIR/build-vulkan/bin/llama-embedding"
