#!/usr/bin/env bash
# TODO-model on 2x DGX Spark (GB10) — vLLM TP=2
# Derived from sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark run.sh @ 025544a with the
# model-specific pieces removed (DSpark spec, B12X env, the refusals). Fill the TODOs through
# recipe.yaml, then add the refusals for what this model cannot do. The shared kit replaces this later.
set -euo pipefail

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-TODO-org/TODO-model}"
SERVED_NAME="${SERVED_NAME:-TODO-org/TODO-model}"
IMAGE="${IMAGE:-TODO-local-image-tag}"
CONTAINER_NAME="${CONTAINER_NAME:-TODO-container-name}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1}"
TP="${TP:-2}"
NNODES="${NNODES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
UTIL="${UTIL:-0.80}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
SPEC_CONFIG="${SPEC_CONFIG:-}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-TODO-40-hex-snapshot-sha}"
SNAPSHOT="${HF_CACHE}/hub/models--${MODEL//\//--}/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--${MODEL//\//--}/snapshots/${SNAPSHOT_SHA}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# END generated

# Refusals go here. Encode what this model cannot do on GB10 so a bad boot fails before docker runs
# (GLM refuses a 1M window on its fp8 pin; DeepSeek refuses MAX_NUM_SEQS above 2 on its 12 GiB pin).
# Give each one a FORCE_UNSAFE_* override and a VALIDATE_ONLY=1 test in .github/workflows/ci.yml.

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only image=%s snapshot=%s ctx=%s seqs=%s spec=%s eager=%s compilation=%s\n' \
    "$IMAGE" "$SNAPSHOT_SHA" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "${SPEC_CONFIG:-none}" "$ENFORCE_EAGER" "${COMPILATION_CONFIG:-default}"
  exit 0
fi

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

token_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    tr -d '[:space:]' <"$HOME/.cache/huggingface/token"
  fi
}

resolve_model() {
  printf '%s\n' "$SNAPSHOT_IN_CONTAINER"
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  fi
}

ensure_image() {
  log "Ensuring image $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build it from docker/ first (see README). Do not use stock vllm/vllm-openai on sm_121." >&2
    exit 1
  fi
}

ensure_weights() {
  if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    local HF=""
    HF="$(hf_bin || true)"
    if [[ -d "$SNAPSHOT" ]]; then
      log "Using pinned snapshot $SNAPSHOT"
    elif [[ -n "$HF" ]]; then
      export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
      log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE)"
      "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA"
    else
      echo "No hf CLI on PATH and snapshot $SNAPSHOT is missing" >&2
      exit 1
    fi
  fi
  if [[ ! -d "$SNAPSHOT" ]]; then
    echo "Pinned snapshot missing: $SNAPSHOT" >&2
    exit 1
  fi
}

refuse_busy_port() {
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    echo "Port $PORT is already in use" >&2
    exit 1
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE"
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found" >&2
    exit 1
  fi
  maybe_drop_caches
  stop_local
  ensure_image
  ensure_weights

  local serve_model
  serve_model="$(resolve_model)"

  local tok
  tok="$(token_env || true)"
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "GLOO_SOCKET_IFNAME=$IFACE"
    -e "TP_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_NVLS_ENABLE=0"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=WARN"
  )
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")
  if [[ -n "$tok" ]]; then
    env_args+=(-e "HF_TOKEN=$tok" -e "HUGGING_FACE_HUB_TOKEN=$tok")
  fi

  local rank_args=()
  if [[ "$rank" == "0" ]]; then
    rank_args+=(--host 0.0.0.0 --port "$PORT")
  else
    rank_args+=(--headless)
  fi

  local eager_args=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  elif [[ -n "$COMPILATION_CONFIG" ]]; then
    eager_args+=(--compilation-config "$COMPILATION_CONFIG")
  fi

  local vol_args=(-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}")
  local opt_args=()
  if [[ -n "$KV_CACHE_MEMORY" ]]; then
    opt_args+=(--kv-cache-memory "$KV_CACHE_MEMORY")
  fi
  if [[ -n "$BLOCK_SIZE" ]]; then
    opt_args+=(--block-size "$BLOCK_SIZE")
  fi
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    opt_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ -n "$SPEC_CONFIG" ]]; then
    opt_args+=(--speculative-config "$SPEC_CONFIG")
  fi

  # The image ENTRYPOINT is `vllm serve`; the first argument is the model path.
  log "Starting $CONTAINER_NAME rank=$rank model=$serve_model ctx=$MAX_MODEL_LEN kv=${KV_CACHE_MEMORY:-auto} eager=$ENFORCE_EAGER"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart no \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    "${vol_args[@]}" \
    "${env_args[@]}" \
    "$IMAGE" \
    "$serve_model" \
    --tensor-parallel-size "$TP" \
    --nnodes "$NNODES" \
    --node-rank "$rank" \
    --distributed-executor-backend mp \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    "${rank_args[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${opt_args[@]}" \
    "${eager_args[@]}" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    $EXTRA_ARGS
}

wait_ready() {
  log "Waiting for http://127.0.0.1:${PORT}/v1/models"
  local i body
  for i in $(seq 1 480); do
    body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
    if [[ -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
      log "Ready → http://127.0.0.1:${PORT}/v1  (context=$MAX_MODEL_LEN)"
      printf '%s\n' "$body"
      echo
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  refuse_busy_port
  if [[ "$NNODES" -gt 1 ]]; then
    if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
      echo "Cannot SSH to $WORKER_HOST. Refusing to start a TP=$TP head rank alone (NNODES=$NNODES)." >&2
      exit 1
    fi
    log "Starting worker on $WORKER_HOST first"
    mkdir -p "${PWD}/.run-state"
    printf '%s\n' "$WORKER_HOST" >"${PWD}/.run-state/worker_host"
    scp -q "$0" "${WORKER_HOST}:/tmp/${CONTAINER_NAME}-run.sh"
    ssh "$WORKER_HOST" \
      "ROLE=worker ORCHESTRATE=0 MODEL='$MODEL' SERVED_NAME='$SERVED_NAME' IMAGE='$IMAGE' CONTAINER_NAME='$CONTAINER_NAME' PORT='$PORT' MASTER_PORT='$MASTER_PORT' HEAD_IP='$HEAD_IP' IFACE='$IFACE' HCA='$HCA' TP='$TP' NNODES='$NNODES' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' UTIL='$UTIL' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' KV_CACHE_MEMORY='$KV_CACHE_MEMORY' BLOCK_SIZE='$BLOCK_SIZE' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' SPEC_CONFIG='$SPEC_CONFIG' COMPILATION_CONFIG='$COMPILATION_CONFIG' ENFORCE_EAGER='$ENFORCE_EAGER' HF_CACHE='$HF_CACHE' SNAPSHOT_SHA='$SNAPSHOT_SHA' SKIP_DOWNLOAD='$SKIP_DOWNLOAD' EXTRA_ARGS='$EXTRA_ARGS' bash /tmp/${CONTAINER_NAME}-run.sh"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
  fi
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
fi
