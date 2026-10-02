#!/usr/bin/env bash
# Long-context A/B of the MTP draft cap, ABAB on one fixed filler (--run 100), so every boot sees the
# same text (a reboot clears TensorFold's kept prompt states, so each boot's first request is cold).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
for cfg in D6:6 D15:15 D6b:6 D15b:15; do
  id=${cfg%%:*}; d=${cfg##*:}; ev=evidence/s5-longctx/$id; mkdir -p "$ev"
  echo "$(date -u +%FT%TZ) $id MTP_DRAFTS=$d"
  ./stop.sh >"$ev/stop.log" 2>&1 </dev/null
  if ! TF_CACHE=/home/sfxnz/projects/data/tensorfold-qwen38/cache MTP_DRAFTS=$d ./run.sh >"$ev/boot.log" 2>&1 </dev/null; then
    echo "$id boot failed"; continue
  fi
  free -h >"$ev/free-before.txt"; ssh spark2 free -h >"$ev/free-before-spark2.txt" </dev/null
  python3 tools/bench_prefill.py --lengths 8192 32768 131072 --run 100 >"$ev/bench-prefill.out" 2>"$ev/bench-prefill.err" </dev/null
  free -h >"$ev/free-after.txt"; ssh spark2 free -h >"$ev/free-after-spark2.txt" </dev/null
  docker logs tf-qwen38-flashnext 2>&1 | grep -E "done req-|decode graphs|startup estimate" >"$ev/requests.log"
  echo "$(date -u +%FT%TZ) $id done"
done
