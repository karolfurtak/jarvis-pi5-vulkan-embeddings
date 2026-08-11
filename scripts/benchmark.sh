#!/usr/bin/env bash
#
# benchmark.sh — CPU vs. Vulkan/GPU wall-clock benchmark for llama-embedding
# on a Raspberry Pi 5. Run on the Pi itself, after build-llamacpp.sh and
# fetch-model.sh have produced both binaries and the model file.
#
# Every repetition starts from an idle machine (1-minute load average below
# 0.3 on this 4-core board, plus a fixed cooldown) so results aren't inflated
# by residual heat or load from the previous repetition. Records: wall time,
# average CPU utilization across all 4 cores during the run (0-100%, derived
# from /proc/stat deltas — not a per-core figure), SoC (System on Chip) and
# RP1 (the Pi 5's I/O controller chip) temperature at start/end/peak,
# `vcgencmd get_throttled` (0x0 = no thermal/voltage throttling), and peak
# resident memory (RSS).
#
# Usage:
#   ./benchmark.sh                          # 5 CPU reps, 5 Vulkan reps
#   REPS_VULKAN=3 ./benchmark.sh            # override repetition count
#
# See ../README.md for why this repository's own run used 5 CPU repetitions
# but only 3 Vulkan repetitions.

set -euo pipefail

BENCH_DIR="$HOME/gpu-bench"
MODEL="$BENCH_DIR/model/paraphrase-multilingual-MiniLM-L12-v2-F16.gguf"
CORPUS="$BENCH_DIR/corpus/sample_600.txt"
OUT_CSV="$BENCH_DIR/results.csv"
REPS_CPU="${REPS_CPU:-5}"
REPS_VULKAN="${REPS_VULKAN:-5}"
COOLDOWN_S=90

BIN_CPU="$BENCH_DIR/llama.cpp/build-cpu/bin/llama-embedding"
BIN_VK="$BENCH_DIR/llama.cpp/build-vulkan/bin/llama-embedding"

SOC_TEMP_PATH="/sys/class/hwmon/hwmon0/temp1_input"   # cpu_thermal (SoC / core)
RP1_TEMP_PATH="/sys/class/hwmon/hwmon1/temp1_input"   # rp1_adc (RP1 I/O chip)

echo "variant,rep,seconds,cpu_pct_avg,soc_start_C,soc_end_C,soc_max_C,rp1_start_C,rp1_end_C,rp1_max_C,throttled_hex,rss_max_kB" > "$OUT_CSV"

read_soc()  { awk '{printf "%.1f", $1/1000}' "$SOC_TEMP_PATH"; }
read_rp1()  { awk '{printf "%.1f", $1/1000}' "$RP1_TEMP_PATH"; }

read_throttled() {
    vcgencmd get_throttled | cut -d= -f2
}

# Aggregate CPU ticks across all cores from /proc/stat's "cpu " line.
cpu_ticks() {
    awk '/^cpu / {busy=$2+$3+$4+$6+$7+$8+$9; total=busy+$5; print busy, total}' /proc/stat
}

wait_idle() {
    # Idle gate: 1-minute load average below 0.3 on this 4-core board,
    # then a fixed cooldown so the previous repetition's heat has time to
    # dissipate before the next one starts.
    while true; do
        la=$(awk '{print $1}' /proc/loadavg)
        below=$(awk -v la="$la" 'BEGIN{print (la<0.3)?1:0}')
        if [ "$below" = "1" ]; then break; fi
        sleep 5
    done
    sleep "$COOLDOWN_S"
}

run_variant() {
    local variant="$1"
    local bin="$2"
    local extra_args="$3"
    local reps="$4"

    for i in $(seq 1 "$reps"); do
        echo "== $variant rep $i/$reps: starting from idle =="
        wait_idle
        soc_start=$(read_soc)
        rp1_start=$(read_rp1)

        temp_log="$BENCH_DIR/.temp_${variant}_${i}.log"
        : > "$temp_log"
        ( while true; do echo "$(read_soc) $(read_rp1)" >> "$temp_log"; sleep 1; done ) &
        monitor_pid=$!

        read cpu_busy_0 cpu_total_0 <<< "$(cpu_ticks)"
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        "$bin" -m "$MODEL" -f "$CORPUS" --pooling mean $extra_args \
            > "$BENCH_DIR/.out_${variant}_${i}.log" 2>&1 &
        run_pid=$!
        rss_peak=0
        while kill -0 "$run_pid" 2>/dev/null; do
            rss=$(grep -oE '^VmRSS:\s+[0-9]+' /proc/"$run_pid"/status 2>/dev/null | awk '{print $2}')
            if [ -n "${rss:-}" ] && [ "$rss" -gt "$rss_peak" ]; then rss_peak=$rss; fi
            sleep 0.5
        done
        wait "$run_pid" || true
        t1=$(date +%s.%N)
        read cpu_busy_1 cpu_total_1 <<< "$(cpu_ticks)"

        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true

        seconds=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
        soc_end=$(read_soc)
        rp1_end=$(read_rp1)
        soc_max=$(awk '{print $1}' "$temp_log" 2>/dev/null | sort -n | tail -1)
        rp1_max=$(awk '{print $2}' "$temp_log" 2>/dev/null | sort -n | tail -1)
        # % of total 4-core capacity used during the run window (0-100%,
        # where 100% = all 4 cores fully busy for the whole window).
        cpu_pct=$(awk -v b0="$cpu_busy_0" -v t0="$cpu_total_0" -v b1="$cpu_busy_1" -v t1="$cpu_total_1" \
            'BEGIN{db=b1-b0; dt=t1-t0; if(dt>0) printf "%.1f", 100*db/dt; else print "NA"}')
        thr=$(read_throttled)

        echo "$variant,$i,$seconds,$cpu_pct,$soc_start,$soc_end,$soc_max,$rp1_start,$rp1_end,$rp1_max,$thr,$rss_peak" >> "$OUT_CSV"
        echo "   seconds=${seconds} cpu=${cpu_pct}% SoC ${soc_start}->${soc_end} (max ${soc_max}) RP1 ${rp1_start}->${rp1_end} (max ${rp1_max}) throttled=$thr rss_max=${rss_peak}kB"
    done
}

echo "=== BASELINE: CPU (4 cores, ARM-native build, Cortex-A76+dotprod) ==="
run_variant "cpu" "$BIN_CPU" "-t 4" "$REPS_CPU"

echo "=== GPU: Vulkan (V3D 7.1.7.0, all layers offloaded, -ngl 99) ==="
run_variant "vulkan" "$BIN_VK" "-ngl 99 -t 4" "$REPS_VULKAN"

echo "Done. Results: $OUT_CSV"
