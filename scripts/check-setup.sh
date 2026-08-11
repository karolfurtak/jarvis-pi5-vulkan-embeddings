#!/usr/bin/env bash
#
# check-setup.sh — prints the GPU (Graphics Processing Unit) driver stack
# state this repository's benchmark depends on: Vulkan driver/version, GPU
# memory heap as seen by Vulkan, OpenCL (rusticl) availability, and the CMA
# (Contiguous Memory Allocator) reservation. Read-only, no changes made.
#
# Requires: vulkan-tools, clinfo, mesa-opencl-icd (installed by
# build-llamacpp.sh).

set -uo pipefail

echo "=== OS / kernel ==="
grep PRETTY_NAME /etc/os-release
uname -r

echo
echo "=== Vulkan driver ==="
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|driverInfo|apiVersion|deviceType"

echo
echo "=== GPU memory heap (as reported by Vulkan) ==="
vulkaninfo 2>&1 | grep -A6 "memoryHeaps\[0\]" | head -7

echo
echo "=== CMA (Contiguous Memory Allocator) reservation ==="
grep -i cma /proc/meminfo
echo "Note: on Raspberry Pi 5, the V3D GPU driver allocates compute buffers"
echo "via generic DRM/GEM (Direct Rendering Manager / Graphics Execution"
echo "Manager) from system RAM, not from this legacy CMA pool — the Vulkan"
echo "heap budget above is the number that actually matters for this"
echo "benchmark. See README.md for what we observed running the benchmark."

echo
echo "=== OpenCL / rusticl ==="
echo "-- without RUSTICL_ENABLE (default) --"
clinfo 2>&1 | grep -E "Number of devices|Platform Name" | head -4
echo "-- with RUSTICL_ENABLE=v3d --"
RUSTICL_ENABLE=v3d clinfo 2>&1 | grep -E "Device Name|Platform Name" | head -4

echo
echo "=== Thermal / throttling state right now ==="
vcgencmd measure_temp
vcgencmd get_throttled
awk '{printf "RP1 temp: %.1f C\n", $1/1000}' /sys/class/hwmon/hwmon1/temp1_input 2>/dev/null || true
