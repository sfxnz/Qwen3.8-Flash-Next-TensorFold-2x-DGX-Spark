#!/usr/bin/env bash
# T3 long-context needles against the live serve. TensorFold has no /tokenize, so the needle documents
# are sized with the checkpoint's own tokenizer.json inside the recipe image (QLIB_TOKENIZER).
#   tools/run_t3.sh OUT_DIR [extra t3_needles.py args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
IMAGE="${IMAGE:-tf-qwen38-flashnext:0.6.2}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAP=/cache/huggingface/hub/models--TensorFold--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/${SNAPSHOT_SHA:-2b170fa6309d5d1ee380b35636075fac7945f286}
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$ROOT:/recipe:ro" -v "$OUT:/out" -v "$HF_CACHE:/cache/huggingface:ro" \
  -e QLIB_TOKENIZER="$SNAP/tokenizer.json" -e HOME=/tmp \
  --entrypoint python "$IMAGE" /recipe/quality/t3_needles.py --url "http://127.0.0.1:${PORT:-8000}" --out /out "$@"
