# Raspberry Pi 5: is the GPU worth using for embeddings?

A measured answer to one question: on a Raspberry Pi 5 with no NPU (Neural
Processing Unit), does routing sentence-embedding inference through the
integrated GPU (V3D, via Vulkan) beat plain CPU inference?

Short answer: **no, by two orders of magnitude** — and the reason turned out
to be more interesting than the question.

## Why this repository exists

This is the companion piece to
[`radxa-zero-3w-npu`](https://github.com/karolfurtak/radxa-zero-3w-npu), which
benchmarks a *real* NPU on a different board for the same class of workload.
The Raspberry Pi 5 has no NPU — its only compute option besides the CPU
(Central Processing Unit) is the VideoCore V3D GPU (Graphics Processing Unit),
accessible through Vulkan via Mesa's V3DV driver. The question was whether
that GPU path is worth building at all for a real, recurring workload (batch
embedding of a text library) on this specific board — decided by measurement,
not by benchmarks published for different (usually discrete, usually much
larger) GPUs.

Same house rule as the sibling repos: failed and negative results are
documented as thoroughly as positive ones. A 44x slowdown that gets explained
is worth more than a fast number nobody can reproduce.

## The criterion — decided before running anything, unchanged after seeing results

The GPU path is worth keeping if it meets **at least one** of:

- **(a)** it is **≥30% faster** than 4-core CPU inference on the same input, **or**
- **(b)** it matches CPU wall-clock time while using **under 50%** of CPU
  capacity (i.e., it frees up the cores for other work at no time cost).

If neither holds, the verdict is **"not worth it"** — the GPU path is dropped,
CPU stays.

## Result

**Neither condition holds — the GPU path is dropped.** The Vulkan/GPU variant
was **25-44x slower** than the CPU baseline on the same input, while using only
6-17% of CPU capacity. Condition (b) fails on its first clause (time parity)
before the CPU-usage clause even matters — freeing up cores is worthless when
the job takes 25-44x longer to do it.

| Variant | Mean time | vs. CPU | CPU usage |
|---|---|---|---|
| CPU (4 cores, baseline) | 21.21 s | — | 80.6% |
| GPU, cold start (1st run ever) | 942.27 s | **44.4x slower** | 16.6% |
| GPU, steady state (cache warm) | 537.99 s | **25.4x slower** | 6.1% |

Full numbers, spread, and methodology: [`results/`](results/) and the
[Method](#method) section below.

## Why so slow — the actual finding

The starting hypothesis was that CPU and GPU on this board share the same
LPDDR4X memory and memory bus (on the order of 17 GB/s combined, per the Pi 5's
published memory configuration), so routing compute through the GPU wouldn't
add bandwidth — no gain expected, but not a catastrophe either. A 25-44x
slowdown is a different scale of problem. Digging into the `llama.cpp` Vulkan
backend's log, batch by batch, found two concrete, unrelated causes — neither
of which is "shared memory bandwidth":

**1. A one-time shader compilation tax (~355-400 seconds), paid once per
machine, not per run.** The very first batch on the very first run took
411.93 s; every batch after that (same run and every subsequent run) took
56.6-58.2 s. Mesa's V3DV driver compiles compute shaders (SPIR-V →
GPU machine code) lazily on first use and caches the result to disk. Run #1
pays the compile tax; runs #2 and #3 don't — which is exactly why the
measured time dropped from 942 s to 538 s with the identical binary and
identical input.

**2. Even past the warm-up: a hard Vulkan allocation-size limit forces a slow
transfer path for every batch.** The log shows, three times per run:

```
W ggml_vulkan: Failed to allocate pinned memory
  (Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory)
```

The V3D device advertises `maxMemoryAllocationSize = 1 GiB` — a cap on any
*single* allocation, separate from the 4 GiB total heap budget. The driver
tries to allocate a "pinned" (host-visible, non-pageable) staging buffer for
fast CPU↔GPU transfer, that request exceeds the 1 GiB per-allocation cap, and
the driver silently falls back to a slower, unpinned transfer path for the
rest of the run. The compute buffers themselves are tiny (5.5-26.4 MiB per the
log) — this isn't a real out-of-memory condition, it's a mismatched allocation
request against a fixed device limit.

**No operator fell back to the CPU** — the log shows zero "not supported" or
"fallback" messages; all 12 transformer layers genuinely ran on the GPU per
`-ngl 99`. The slowness isn't the GPU failing to compute — it's the
communication overhead between CPU and GPU dominating, for a workload made of
many small batches (~2000 tokens, ~384-dimensional vectors) on an integrated
GPU without a working fast-transfer path.

**Verdict on the original hypothesis:** not disproven outright — CPU and GPU
still share the same memory bus, and that's still a ceiling on any future
gain — but it is not the dominant explanation for what was measured. The
dominant costs are the one-time shader compile and the pinned-memory
allocation fallback. Measured, not assumed, per the brief.

## A side note on temperature

- The SoC (System on Chip) ran **cooler** under GPU load (55.1-59.5°C peak)
  than under CPU load (62.2-63.4°C peak) — despite running 25-44x longer.
  Consistent with the physics: CPU load is short and intense (~81% of 4 cores
  for ~21 s = high instantaneous power), GPU load is long and CPU-light
  (~6-17% of 4 cores for 9-16 minutes = low instantaneous power, spread out).
- The **RP1** (the Pi 5's separate I/O controller chip — USB, Ethernet, GPIO)
  ran **warmer** under GPU load (51.4-53.7°C peak) than under CPU load
  (48.5-49.7°C peak). RP1 has no role in V3D compute — the most likely
  explanation is test *duration*, not workload type: a longer run gives more
  time for heat to soak across the board from the adjacent SoC, which RP1
  picks up as a slow ambient rise. This is a physical reading, not a causal
  claim — there's no evidence RP1 is "doing work" during GPU inference, only
  that it sits near a heat source that ran longer.
- **`throttled=0x0` on every one of the 8 runs, no exceptions** — no thermal
  or voltage throttling anywhere in this benchmark. Important methodological
  caveat: the 25-44x gap is real, not a throttling artifact.

## Method

- **Model:** [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  (118M parameters, 384-dim embeddings) — the same model used for embeddings on
  the companion NPU project. GGUF (GPT-Generated Unified Format) F16 export,
  231 MiB. Full attribution and checksum: [`model-attribution.md`](model-attribution.md).
- **Corpus:** 600 English sentences (`corpus/sample_600.jsonl`), the same
  sample used for embedding-quality diagnostics on the companion NPU project —
  identical input on both benchmark arms.
- **Backends:** two separate `llama.cpp` builds (commit `153d324b`, 2026-08-11)
  from the same source tree — one CPU-only (`-DGGML_VULKAN=OFF`, ARM-native,
  detected Cortex-A76 + dotprod), one with Vulkan enabled
  (`-DGGML_VULKAN=ON`, `-ngl 99` at run time to offload all layers). Same
  unmodified GGUF file loaded by both.
- **Repetitions:** 5 for CPU, **3 for Vulkan (not 5 — see below)**. Every
  repetition starts from an idle machine (1-minute load average below 0.3 on
  this 4-core board, plus a 90 s cooldown) so results aren't inflated by heat
  or load carried over from the previous repetition.
- **Why 3 Vulkan repetitions, not 5:** each Vulkan repetition took 9-16
  minutes. After 3 (one cold-start, two steady-state — see the finding above),
  the steady-state pair agreed to within 0.34 s on 538 s (0.06% spread) and the
  CPU-vs-GPU gap was already two orders of magnitude. Two more repetitions
  (~32 more minutes of a shared machine) would not have changed the
  conclusion. Stated here explicitly so this reads as a documented,
  reasoned stop — not a silently shortened sample.
- **Recorded per run:** wall-clock time, average CPU utilization across all 4
  cores (0-100%, from `/proc/stat` deltas), SoC and RP1 temperature at
  start/end/peak, `vcgencmd get_throttled` (thermal/voltage throttle state),
  peak resident memory.

## Setup

| Item | Value |
|---|---|
| Board | Raspberry Pi 5 (BCM2712) |
| OS | Debian 13 (Trixie), kernel `6.18.39+rpt-rpi-2712`, aarch64 |
| GPU | VideoCore **V3D 7.1.7.0** (integrated, no dedicated VRAM) |
| Vulkan driver | **V3DV** (Mesa 26.2.0), Vulkan API 1.4.309 (loader) / 1.3.354 (device) |
| GPU memory heap (as seen by Vulkan) | 4096 MiB, dynamically shared from system LPDDR4X |
| `maxMemoryAllocationSize` (device) | 1 GiB — see the finding above |
| OpenCL / rusticl | present but reports 0 devices unless `RUSTICL_ENABLE=v3d` is set; not used by this benchmark (`llama.cpp` uses the Vulkan backend) |

Two apt packages are easy to miss when building `llama.cpp` with
`-DGGML_VULKAN=ON` on Debian 13 — both cause a CMake *configure*-time failure,
not a compile error, so they surface late: `glslc` (the actual GLSL→SPIR-V
compiler; a separate package from `glslang-tools`, which only provides
`glslangValidator`) and `spirv-headers` (`find_package(SPIRV-Headers)` in
ggml-vulkan's `CMakeLists.txt`). Both are in `scripts/build-llamacpp.sh`.

## Reproducing this

```bash
git clone https://github.com/karolfurtak/jarvis-pi5-vulkan-embeddings
cd jarvis-pi5-vulkan-embeddings

./scripts/check-setup.sh          # inspect the Vulkan/OpenCL/CMA state first
./scripts/build-llamacpp.sh       # ~15 min CPU build + ~16 min Vulkan build on a Pi 5
./scripts/fetch-model.sh          # downloads + checksums the GGUF file (231 MiB)
cp corpus/sample_600.jsonl <extract to plain text, see below> ~/gpu-bench/corpus/sample_600.txt
./scripts/benchmark.sh            # ~2 min CPU + 45-90 min Vulkan (5+5 reps by default)
```

`benchmark.sh` reads `~/gpu-bench/corpus/sample_600.txt` — one sentence per
line, not JSON Lines. Extract it from the committed `.jsonl`:

```bash
python3 -c "
import json
with open('corpus/sample_600.jsonl') as f, open('sample_600.txt', 'w') as out:
    for line in f:
        out.write(json.loads(line)['text'].replace(chr(10), ' ').strip() + chr(10))
"
```

Override repetition counts with `REPS_CPU` / `REPS_VULKAN` environment
variables — see the header of `scripts/benchmark.sh`.

The build script was re-run from a clean clone and fresh build directories
before this repository was published, to confirm it reproduces without any
undocumented manual step.

## Repository contents

- [`scripts/check-setup.sh`](scripts/check-setup.sh) — read-only inspection of
  the Vulkan/OpenCL/CMA state
- [`scripts/build-llamacpp.sh`](scripts/build-llamacpp.sh) — builds both
  `llama-embedding` binaries
- [`scripts/fetch-model.sh`](scripts/fetch-model.sh) — downloads + checksums
  the GGUF model file
- [`scripts/benchmark.sh`](scripts/benchmark.sh) — the benchmark itself
- [`corpus/sample_600.jsonl`](corpus/sample_600.jsonl) — the 600-sentence test set
- [`results/results.csv`](results/results.csv) — raw output of the run this
  README reports
- [`model-attribution.md`](model-attribution.md) — model source and license

## License

[MIT](LICENSE) for the scripts and documentation in this repository. The
embedding model is a separate artifact under Apache 2.0 — see
[`model-attribution.md`](model-attribution.md).
