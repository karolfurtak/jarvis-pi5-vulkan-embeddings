# Model attribution and license

## Source model

**[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)**
— 118M parameters, 384-dimensional sentence embeddings, 50+ languages including
Polish and English. Published by the [sentence-transformers](https://www.sbert.net/)
project (UKP Lab, Technical University of Darmstadt).

**License: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**
— permits use, modification, and redistribution of derivative works (including
format conversions such as the GGUF (GPT-Generated Unified Format) export this
repository benchmarks), provided the license text is retained and changes are
stated. Both conditions are met by this document.

This is the same base model used for embeddings on the companion project's
NPU (Neural Processing Unit) — see
[`radxa-zero-3w-npu/model-attribution.md`](https://github.com/karolfurtak/radxa-zero-3w-npu/blob/main/model-attribution.md).
Same weights, two different runtimes (RKNN int8/fp16 on the Radxa's NPU vs.
GGUF F16 on the Raspberry Pi 5's CPU/GPU) — comparable at the level of "same
model family, fp16 precision," not bit-identical output.

## GGUF export used as the benchmark artifact

**[`mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf`](https://huggingface.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf)**
(file: `paraphrase-multilingual-MiniLM-L12-118M-v2-F16.gguf`, 231 MiB) — the
same weights as the source model above, pre-converted to GGUF F16 (16-bit
floating point, full precision of the original checkpoint, no quantization)
for use with `llama.cpp`. Used here instead of running our own
`convert_hf_to_gguf.py` conversion, since the weights are identical and this
conversion is directly loadable by both the CPU and Vulkan backends tested.
No separate license is declared on that repository; it inherits the
Apache-2.0 terms of the weights it re-exports.

## What this repository adds

Nothing to the model itself — this repository only benchmarks two `llama.cpp`
inference backends (CPU vs. Vulkan/GPU) against the unmodified F16 GGUF file
above. No fine-tuning, no re-quantization, no architecture changes.

## Why the GGUF file itself is not committed here

231 MiB — over plain git's 100 MB per-file limit. `scripts/fetch-model.sh`
downloads the exact same file (checksummed) instead of vendoring it.
