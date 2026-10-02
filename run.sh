#!/usr/bin/env bash
# Qwen3.8-Flash-Next (MLX 4-bit + MTP) on 2x DGX Spark (GB10) with TensorFold, TP=2.
# Guards and orchestration follow sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark run.sh @ d66a95e.
# The head (spark1) starts rank 1 on WORKER_HOST over ssh, then rank 0, which serves HTTP.
set -euo pipefail

# PROFILE=concurrent: the opt-in two-rank --parallel image (README "Concurrency"). It only fills the
# variables below when they are unset, so an explicit IMAGE / TF_PATCH / PARALLEL still wins.
PROFILE="${PROFILE:-serial}"
case "$PROFILE" in
  serial) ;;
  concurrent)
    IMAGE="${IMAGE:-tf-qwen38-flashnext:0.6.2-pr141}"
    TF_PATCH="${TF_PATCH:-patches/pr141-on-0.6.2.patch}"
    PARALLEL="${PARALLEL:-8}"
    ;;
  *) echo "PROFILE=$PROFILE must be serial or concurrent." >&2; exit 1 ;;
esac

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
SERVED_NAME="${SERVED_NAME:-TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
IMAGE="${IMAGE:-tf-qwen38-flashnext:0.6.2}"
TF_SHA="${TF_SHA:-56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21}"
TF_PATCH="${TF_PATCH:-none}"
CONTAINER_NAME="${CONTAINER_NAME:-tf-qwen38-flashnext}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29551}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1,roceP2p1s0f1}"
TP="${TP:-2}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-bf16}"
MTP_DRAFTS="${MTP_DRAFTS:-15}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.70}"
PARALLEL="${PARALLEL:-1}"
THINKING="${THINKING:-0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-2b170fa6309d5d1ee380b35636075fac7945f286}"
SNAPSHOT="${HF_CACHE}/hub/models--TensorFold--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--TensorFold--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/${SNAPSHOT_SHA}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
TF_CACHE="${TF_CACHE:-$HOME/.cache/tensorfold-qwen38}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-1000}"
MEMGUARD="${MEMGUARD:-1}"
MEMGUARD_MIN_AVAIL_MB="${MEMGUARD_MIN_AVAIL_MB:-3072}"
MEMGUARD_MIN_SWAP_FREE_MB="${MEMGUARD_MIN_SWAP_FREE_MB:-2048}"
BENCH_ONLY="${BENCH_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
# END generated
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.run-state"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# TensorFold's native window for this checkpoint (config.json max_position_embeddings). It refuses more.
NATIVE_CONTEXT=262144
# The engine's draft cap (families/qwen4_exp/cuda/engine.py MAX_DEPTH).
MAX_MTP_DRAFTS=15
# The only patch that serves --parallel on two ranks (docker/patches/; README "Concurrency").
PARALLEL_PATCH=patches/pr141-on-0.6.2.patch

die() {
  echo "$*" >&2
  exit 1
}

# --- integers and flags. Leading zeros are refused: bash reads 010 as octal 8.
for name in PORT MASTER_PORT TP CONTEXT PARALLEL MAX_TOKENS MEMGUARD_MIN_AVAIL_MB MEMGUARD_MIN_SWAP_FREE_MB; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name=${!name} is not a positive decimal integer."
done
[[ "$MTP_DRAFTS" =~ ^(0|[1-9][0-9]*)$ ]] || die "MTP_DRAFTS=$MTP_DRAFTS is not a non-negative decimal integer."
for name in THINKING SKIP_DOWNLOAD HF_HUB_DISABLE_XET MEMGUARD BENCH_ONLY; do
  [[ "${!name}" =~ ^[01]$ ]] || die "$name=${!name} must be 0 or 1."
done
[[ "$OOM_SCORE_ADJ" =~ ^(-?[1-9][0-9]*|0)$ ]] && (( OOM_SCORE_ADJ >= -1000 && OOM_SCORE_ADJ <= 1000 )) || die "OOM_SCORE_ADJ=$OOM_SCORE_ADJ must be an integer in [-1000, 1000]."
[[ "$MTP_CONFIDENCE" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || die "MTP_CONFIDENCE=$MTP_CONFIDENCE must be a decimal in [0, 1], e.g. 0.70."
[[ -z "$MEMORY_RESERVE_GIB" || "$MEMORY_RESERVE_GIB" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]] || die "MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB is not empty or a positive decimal."
[[ "$TF_SHA" =~ ^[0-9a-f]{40}$ ]] || die "TF_SHA=$TF_SHA is not a 40-hex TensorFold commit."
[[ "$SNAPSHOT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "SNAPSHOT_SHA=$SNAPSHOT_SHA is not a 40-hex snapshot revision."
[[ "$TF_CACHE" == /* ]] || die "TF_CACHE=$TF_CACHE must be an absolute path."
# The head hashes the patch file; the worker's /tmp copy of this script has no docker/ dir and gets the
# head's hash forwarded instead (its image must carry the same patch_sha label either way).
if [[ "$TF_PATCH" == none ]]; then
  TF_PATCH_SHA=none
else
  [[ "$TF_PATCH" =~ ^patches/[A-Za-z0-9._-]+\.patch$ ]] || die "TF_PATCH=$TF_PATCH must be none or a file under docker/patches/."
  if [[ -f "$SCRIPT_DIR/docker/$TF_PATCH" ]]; then
    TF_PATCH_SHA="$(sha256sum "$SCRIPT_DIR/docker/$TF_PATCH" | cut -d' ' -f1)"
  elif [[ "${ROLE:-}" == worker && "${TF_PATCH_SHA:-}" =~ ^[0-9a-f]{64}$ ]]; then
    :
  else
    die "TF_PATCH=$TF_PATCH must be none or a file under docker/patches/."
  fi
fi

# --- topology. TensorFold runs one rank per machine, at most two (cli_args.py --tp choices 1/2).
case "$TP" in
  1 | 2) ;;
  *) die "TP=$TP: TensorFold serves Flash Next on one or two ranks only." ;;
esac
case "$KV_DTYPE" in
  bf16 | int8 | int4) ;;
  *) die "KV_DTYPE=$KV_DTYPE must be bf16, int8 or int4." ;;
esac
(( CONTEXT <= NATIVE_CONTEXT )) || die "CONTEXT=$CONTEXT exceeds the native window $NATIVE_CONTEXT; TensorFold refuses it and serves no YaRN."
(( MTP_DRAFTS <= MAX_MTP_DRAFTS )) || die "MTP_DRAFTS=$MTP_DRAFTS exceeds the engine cap $MAX_MTP_DRAFTS."
# 0.6.2 refuses --parallel with --tp 2 (engine.py:64-66). The PR-141 port serves it.
if (( PARALLEL > 1 && TP == 2 )) && [[ "$TF_PATCH" != "$PARALLEL_PATCH" ]]; then
  die "PARALLEL=$PARALLEL with TP=2 needs TF_PATCH=$PARALLEL_PATCH (TensorFold 0.6.2 serves one request at a time on two ranks; README 'Concurrency')."
fi

# --- EXTRA_ENV: KEY=VALUE words, no spaces inside a value.
EXTRA_ENV_ARGS=()
for kv in $EXTRA_ENV; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "EXTRA_ENV entry '$kv' is not KEY=VALUE."
  EXTRA_ENV_ARGS+=(-e "$kv")
done

# --- EXTRA_ARGS must not re-set a flag run.sh builds; argparse keeps the last value, so a duplicate
# would bypass the guard on its variable or desynchronise the two ranks.
for w in $EXTRA_ARGS; do
  case "${w%%=*}" in
    --tp | --rank | --master | --master-port | --host | --port | --name | --context | --kv-dtype | --mtp-drafts | --mtp-confidence | --parallel | --thinking | --no-thinking | --max-tokens | --no-update-check | --no-drafts)
      die "EXTRA_ARGS sets $w, which run.sh passes itself. Use TP, PORT, MASTER_PORT, SERVED_NAME, CONTEXT, KV_DTYPE, MTP_DRAFTS (0 = no drafts), MTP_CONFIDENCE, PARALLEL, THINKING or MAX_TOKENS instead."
      ;;
    --vision | --vision-urls | --vision-max-images)
      die "EXTRA_ARGS sets $w: TensorFold serves Flash Next images on one GPU with --parallel 2 or more only (engine.py:50-51)."
      ;;
    --prefill-fp8 | --drafter | --ple-on-ssd | --ssd-experts)
      die "EXTRA_ARGS sets $w: refused or unused for the MLX 4-bit checkpoint on CUDA (README 'Not supported')."
      ;;
  esac
done

API_HOST=0.0.0.0
[[ "$BENCH_ONLY" == 1 ]] && API_HOST=127.0.0.1

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only profile=%s image=%s tf=%s patch=%s snapshot=%s tp=%s ctx=%s kv=%s mtp=%s@%s parallel=%s thinking=%s max_tokens=%s hca=%s host=%s\n' \
    "$PROFILE" "$IMAGE" "$TF_SHA" "$TF_PATCH" "$SNAPSHOT_SHA" "$TP" "$CONTEXT" "$KV_DTYPE" "$MTP_DRAFTS" "$MTP_CONFIDENCE" "$PARALLEL" "$THINKING" "$MAX_TOKENS" "$HCA" "$API_HOST"
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

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    log "drop_caches: ran"
  else
    log "drop_caches: skipped (no passwordless sudo)"
  fi
}

image_sha() {
  # "<tensorfold sha> <patch sha>" from the image labels, empty when the image is missing.
  docker image inspect -f '{{ index .Config.Labels "tensorfold.sha" }} {{ index .Config.Labels "tensorfold.patch_sha" }}' "$IMAGE" 2>/dev/null || true
}

ensure_image() {
  local have
  have="$(image_sha)"
  if [[ -z "$have" ]]; then
    [[ -f "$SCRIPT_DIR/docker/Dockerfile" ]] || die "Image $IMAGE is missing and $SCRIPT_DIR/docker/Dockerfile is not here. Build it on this node (README 'Image')."
    log "Building $IMAGE (TensorFold $TF_SHA, patch $TF_PATCH) from docker/Dockerfile"
    docker build --build-arg "TF_SHA=$TF_SHA" --build-arg "TF_PATCH=$TF_PATCH" --build-arg "TF_PATCH_SHA=$TF_PATCH_SHA" \
      -t "$IMAGE" "$SCRIPT_DIR/docker"
    have="$(image_sha)"
  fi
  [[ "$have" == "$TF_SHA $TF_PATCH_SHA" ]] || die "Image $IMAGE carries TensorFold/patch '${have:-unknown}', not '$TF_SHA $TF_PATCH_SHA' (TF_SHA, sha256 of TF_PATCH). Rebuild it or set IMAGE to the matching tag."
  log "Image $IMAGE (TensorFold $TF_SHA, patch $TF_PATCH)"
}

snapshot_complete() {
  # Every shard in the index, config.json and the tokenizer exist, and each shard's safetensors
  # header ends exactly at its file size (a truncated download fails here, not mid-load).
  python3 - "$SNAPSHOT" <<'PY'
import json
import struct
import sys
from pathlib import Path

snap = Path(sys.argv[1])
bad = []
for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "generation_config.json"):
    p = snap / name
    if not p.is_file() or p.stat().st_size == 0:
        bad.append(name)
index = snap / "model.safetensors.index.json"
try:
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"snapshot {snap} incomplete: {index.name} ({exc})")
for shard in shards:
    p = snap / shard
    try:
        with p.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if 8 + n + end != p.stat().st_size:
            bad.append(f"{shard} (truncated)")
    except (OSError, ValueError, struct.error) as exc:
        bad.append(f"{shard} ({exc.__class__.__name__})")
if bad:
    more = " ..." if len(bad) > 8 else ""
    sys.exit(f"snapshot {snap} incomplete: {', '.join(bad[:8])}{more}")
PY
}

ensure_weights() {
  if snapshot_complete 2>/dev/null; then
    log "Using pinned snapshot $SNAPSHOT (index shards, headers and tokenizer checked)"
    return
  fi
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    snapshot_complete || true
    die "SKIP_DOWNLOAD=1 and the pinned snapshot is incomplete."
  fi
  local HF=""
  HF="$(hf_bin || true)"
  [[ -n "$HF" ]] || die "No hf CLI on PATH and snapshot $SNAPSHOT is incomplete."
  export HF_HUB_DISABLE_XET
  log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE; about 113 GB)"
  "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA"
  snapshot_complete || die "Snapshot still incomplete after download."
}

refuse_foreign_serve() {
  local name devices
  while IFS= read -r name; do
    [[ -z "$name" || "$name" == "$CONTAINER_NAME" ]] && continue
    devices="$(docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}} {{json .Config.Env}}' "$name" 2>/dev/null || true)"
    if printf '%s' "$devices" | grep -Eqi 'gpu|nvidia|infiniband'; then
      die "$name is using GPUs or InfiniBand. This recipe needs exclusive GPUs on both Sparks. Stop that serve first (its own stop script). Do not docker rm it from this script."
    fi
  done < <(docker ps --format '{{.Names}}')
}

refuse_busy_port() {
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    die "Port $PORT is already in use"
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Stopping existing container $CONTAINER_NAME"
    docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

# Host memory watchdog. GB10 is unified memory: when the serve plus host load exhausts RAM and swap,
# the host thrashes for minutes before the kernel OOM killer acts, and it then picks small services
# first. Kill the serve instead once available RAM and free swap are both low for 3 samples (6 s).
# Exits when the container is gone.
memguard_loop() {
  local name="$1" min_avail_kb=$(( $2 * 1024 )) min_swap_kb=$(( $3 * 1024 )) hits=0 n=0 avail swapfree
  while :; do
    avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swapfree="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    if (( avail < min_avail_kb && swapfree < min_swap_kb )); then
      hits=$(( hits + 1 ))
    else
      hits=0
    fi
    if (( hits >= 3 )); then
      logger -t tf-qwen38-memguard "MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      echo "$(date -Is) MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      docker kill "$name" >/dev/null 2>&1 || true
      return 0
    fi
    n=$(( n + 1 ))
    if (( n % 8 == 0 )) && [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]]; then
      return 0
    fi
    sleep 2
  done
}

start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || { log "memguard off (MEMGUARD=$MEMGUARD)"; return 0; }
  local logf="$STATE_DIR/memguard.log"
  mkdir -p "$STATE_DIR"
  setsid bash -c "$(declare -f memguard_loop); memguard_loop \"\$@\"" memguard \
    "$CONTAINER_NAME" "$MEMGUARD_MIN_AVAIL_MB" "$MEMGUARD_MIN_SWAP_FREE_MB" >>"$logf" 2>&1 </dev/null &
  disown || true
  log "memguard pid=$! min_avail=${MEMGUARD_MIN_AVAIL_MB}MB min_swap_free=${MEMGUARD_MIN_SWAP_FREE_MB}MB log=$logf"
}

serve_args() {
  # The tensorfold serve argv after the model directory, for one rank.
  local rank="$1"
  local args=(--context "$CONTEXT" --kv-dtype "$KV_DTYPE" --no-update-check)
  if [[ "$MTP_DRAFTS" == 0 ]]; then
    args+=(--no-drafts)
  else
    args+=(--mtp-drafts "$MTP_DRAFTS" --mtp-confidence "$MTP_CONFIDENCE")
  fi
  (( PARALLEL > 1 )) && args+=(--parallel "$PARALLEL")
  if [[ "$TP" == 2 ]]; then
    args+=(--tp 2 --rank "$rank" --master "$HEAD_IP" --master-port "$MASTER_PORT")
  fi
  if [[ "$rank" == 0 ]]; then
    args+=(--name "$SERVED_NAME" --host "$API_HOST" --port "$PORT" --max-tokens "$MAX_TOKENS")
    if [[ "$THINKING" == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
  fi
  printf '%s\n' "${args[@]}"
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE" "$TF_CACHE/$TF_SHA"
  command -v docker >/dev/null 2>&1 || die "docker not found"
  refuse_foreign_serve
  stop_local
  maybe_drop_caches
  ensure_image
  ensure_weights

  # The snapshot is complete (ensure_weights), so the container never needs the Hub or a token.
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "HF_HUB_OFFLINE=1"
    -e "TENSORFOLD_NO_UPDATE_CHECK=1"
    -e "TORCH_EXTENSIONS_DIR=/cache/tf/torch_extensions"
    -e "TRITON_CACHE_DIR=/cache/tf/triton"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_DEBUG=WARN"
  )
  [[ -n "$MEMORY_RESERVE_GIB" ]] && env_args+=(-e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB")
  env_args+=("${EXTRA_ENV_ARGS[@]}")

  local args=()
  mapfile -t args < <(serve_args "$rank")

  # --init: rank 1 installs no SIGTERM handler and would ignore it as PID 1, so `docker stop` would wait
  # out its timeout. --ulimit memlock + IPC_LOCK: the 29.8 GiB n-gram tables are mlocked; without them
  # the lock fails silently and lookups can page. --ulimit core=1: no multi-GiB core dumps in host RAM.
  log "Starting $CONTAINER_NAME rank=$rank tp=$TP ctx=$CONTEXT kv=$KV_DTYPE mtp=$MTP_DRAFTS@$MTP_CONFIDENCE parallel=$PARALLEL hca=$HCA"
  # shellcheck disable=SC2086 # EXTRA_ARGS is word-split on purpose
  docker run -d \
    --name "$CONTAINER_NAME" \
    --init \
    --restart no \
    --oom-score-adj "$OOM_SCORE_ADJ" \
    --ulimit core=1 \
    --gpus all \
    --network host \
    --ipc host \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    -v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}:ro" \
    -v "${TF_CACHE}/${TF_SHA}:/cache/tf" \
    "${env_args[@]}" \
    "$IMAGE" \
    tensorfold serve "$SNAPSHOT_IN_CONTAINER" "${args[@]}" $EXTRA_ARGS >/dev/null
  start_memguard
}

# Variables the worker rank needs, forwarded shell-quoted over ssh.
FORWARD_VARS=(
  MODEL SERVED_NAME IMAGE TF_SHA TF_PATCH CONTAINER_NAME PORT MASTER_PORT HEAD_IP IFACE HCA TP CONTEXT KV_DTYPE
  TF_PATCH_SHA MTP_DRAFTS MTP_CONFIDENCE PARALLEL THINKING MAX_TOKENS MEMORY_RESERVE_GIB HF_CACHE SNAPSHOT_SHA
  SKIP_DOWNLOAD HF_HUB_DISABLE_XET TF_CACHE OOM_SCORE_ADJ MEMGUARD MEMGUARD_MIN_AVAIL_MB
  MEMGUARD_MIN_SWAP_FREE_MB BENCH_ONLY EXTRA_ARGS EXTRA_ENV
)

worker_env() {
  local v out="ROLE=worker ORCHESTRATE=0"
  for v in "${FORWARD_VARS[@]}"; do
    out+=" $v=$(printf '%q' "${!v}")"
  done
  printf '%s\n' "$out"
}

worker_state() {
  # Prints true, false or missing; prints nothing when ssh itself fails. docker inspect prints an
  # empty line before failing on a missing container, so strip whitespace.
  { ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
    "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null || echo missing" 2>/dev/null || true; } |
    tr -d '[:space:]'
}

abort_worker_dead() {
  echo "Worker $CONTAINER_NAME on $WORKER_HOST is not running ($1). Worker logs:" >&2
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker logs --tail 120 '$CONTAINER_NAME'" >&2 2>&1 || true
  echo "Stop the head with ./stop.sh" >&2
  exit 1
}

sync_worker_image() {
  # The worker runs the head's exact image: same image ID, or the head's copy is sent over the link
  # (no second nvcr.io pull, and no rank pair built from two different builds).
  local local_id remote_id
  ensure_image
  local_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
  remote_id="$(ssh -o BatchMode=yes "$WORKER_HOST" "docker image inspect -f '{{.Id}}' '$IMAGE' 2>/dev/null || true" | tr -d '[:space:]')"
  [[ "$remote_id" == "$local_id" ]] && return 0
  log "Copying $IMAGE to $WORKER_HOST (worker has ${remote_id:-none}, head has $local_id)"
  docker save "$IMAGE" | ssh "$WORKER_HOST" docker load >/dev/null
}

wait_ready() {
  local watch_worker="${1:-0}"
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models"
  local i health body state
  for i in $(seq 1 720); do
    health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
    if [[ "$health" == "200" && -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
      log "Ready → http://${API_HOST}:${PORT}/v1  (context=$CONTEXT)"
      printf '%s\n' "$body"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    # Every ~10 s, so a dead worker aborts the head within a minute instead of a 600 s rendezvous wait.
    if [[ "$watch_worker" == 1 ]] && (( i % 2 == 0 )); then
      state="$(worker_state)"
      [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
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
  refuse_foreign_serve
  refuse_busy_port
  watch_worker=0
  if [[ "$TP" == 2 ]]; then
    if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
      die "Cannot SSH to $WORKER_HOST. Refusing to start a TP=2 head rank alone."
    fi
    mkdir -p "$STATE_DIR"
    printf '%s\n' "$WORKER_HOST" >"$STATE_DIR/worker_host"
    sync_worker_image
    log "Starting rank 1 on $WORKER_HOST first"
    scp -q "$0" "${WORKER_HOST}:/tmp/${CONTAINER_NAME}-run.sh"
    ssh "$WORKER_HOST" "$(worker_env) bash /tmp/${CONTAINER_NAME}-run.sh"
    # Rank 1 waits up to 600 s on rank 0's rendezvous store; give its container a moment to come up.
    sleep 5
    state="$(worker_state)"
    [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
    watch_worker=1
  fi
  start_local 0
  wait_ready "$watch_worker"
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Rank 1 is up and waits for rank 0 at $HEAD_IP:$MASTER_PORT."
else
  start_local 0
  wait_ready 0
  log "Stop with: ./stop.sh"
fi
