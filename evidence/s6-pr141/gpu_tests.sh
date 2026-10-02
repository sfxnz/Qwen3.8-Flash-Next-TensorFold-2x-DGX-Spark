#!/usr/bin/env bash
# TensorFold's own CUDA test suites for Flash Next concurrency, run on one GB10 inside an image
# (two ranks as threads on one GPU, synthetic checkpoints; no serve may be up).
#   evidence/s6-pr141/gpu_tests.sh IMAGE OUTFILE
set -uo pipefail
IMAGE="$1"; OUT="$2"
TESTS="tests/cuda/test_flashnext_tp.py tests/cuda/test_flashnext_tp_multi.py tests/cuda/test_flashnext_multi.py tests/cuda/test_flashnext_prompt_cache.py tests/cuda/test_flashnext_fork_lanes.py tests/cuda/test_flashnext_fill_arrivals.py tests/cuda/test_flashnext_large_pieces.py"
docker run --rm --gpus all --ipc host --ulimit memlock=-1:-1 \
  -v /home/sfxnz/projects/data/tensorfold-qwen38/cache/56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21:/cache/tf \
  -e TORCH_EXTENSIONS_DIR=/cache/tf/torch_extensions -e TRITON_CACHE_DIR=/cache/tf/triton \
  --entrypoint bash "$IMAGE" -c "cd /opt/tensorfold-src && pip install -q pytest >/dev/null 2>&1; \
    for t in $TESTS; do [ -f \$t ] || { echo \"MISSING \$t\"; continue; }; echo \"=== \$t\"; \
    python -m pytest -q -p no:cacheprovider \$t 2>&1 | tail -15; done" >"$OUT" 2>&1
grep -E "^=== |passed|failed|error|MISSING" "$OUT"
