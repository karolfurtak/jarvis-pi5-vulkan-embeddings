#!/usr/bin/env bash
#
# fetch-model.sh — downloads the exact GGUF (GPT-Generated Unified Format)
# embedding model file this repository benchmarks. See ../model-attribution.md
# for what this file is, its source, and its license.
#
# Not vendored in this repo: 231 MiB, over plain git's 100 MB per-file limit.

set -euo pipefail

BENCH_DIR="${1:-$HOME/gpu-bench}"
MODEL_DIR="$BENCH_DIR/model"
MODEL_FILE="$MODEL_DIR/paraphrase-multilingual-MiniLM-L12-v2-F16.gguf"
MODEL_URL="https://huggingface.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf/resolve/main/paraphrase-multilingual-MiniLM-L12-118M-v2-F16.gguf"
EXPECTED_SHA256="fa47907f8732d52ba31796d74bc87c4642b76da465d7d1f74df04ca89f3a9aab"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    actual="$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)"
    if [ "$actual" = "$EXPECTED_SHA256" ]; then
        echo "Model already present and checksum matches: $MODEL_FILE"
        exit 0
    fi
    echo "Existing file checksum mismatch, re-downloading..." >&2
fi

curl -L -o "$MODEL_FILE" "$MODEL_URL"

actual="$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)"
if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: checksum mismatch after download." >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  actual:   $actual" >&2
    exit 1
fi

echo "Downloaded and verified: $MODEL_FILE"
