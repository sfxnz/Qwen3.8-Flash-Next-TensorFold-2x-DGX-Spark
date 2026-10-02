# AGENTS.md — Qwen3.8-Flash-Next · TensorFold · 2× DGX Spark

Serve `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` (snapshot `2b170fa`) at TP=2 with the TensorFold engine (`56e2e3e`, v0.6.2) inside `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned in `docker/Dockerfile`). Not vLLM. The vLLM recipe for the NVFP4 checkpoint is a separate repo (`sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark`); one repo per engine.

Humans read [README.md](README.md).

## Standing orders

1. **PR only.** Commit on `agent/**` branches. Open a PR against `main`. Never push `main`. Never merge. Never force-push.
2. **Evidence.** Every number in `README.md` has a file under `evidence/` that proves it. No file, no number. Never gitignore `evidence/`. `recipe.yaml` names the file per measured row; `python3 kit/render.py --check` lists the rows without one.
3. **Exclusive GPUs.** One serve per pair. Before a boot, stop any other GPU serve on both Sparks with its own stop script (gracefully: `docker stop`, not `docker rm -f`) and restore it afterwards with its smokes. `run.sh` refuses to start next to a foreign GPU container and never removes it.
4. **Validate first.** `VALIDATE_ONLY=1 ./run.sh` before any serve. It needs no Docker.
5. **Source of truth.** Values live in `recipe.yaml`. Edit it and run `python3 kit/render.py`. Never edit inside the generated markers by hand.
6. **Voice.** Short sentences. Numbers first. No marketing language. No emojis.
7. **Secrets.** Never commit tokens, Tailscale IPs, or auth headers.

## Working rules

- Change one knob at a time against the gate (`tools/session_gate.sh EVDIR`, or `tools/sweep.sh SPEC EVROOT` for a list). Compare with `tools/compare.py BASE CAND`. Session-to-session noise on this pair is about ±2.5% per cell: a lever that does not beat it, or that regresses another cell, is reverted and recorded.
- `bench_decode.py` is byte-identical to the vLLM sibling's frozen ruler (sha256 `6a9c64bd…`, enforced by `tests/`). Do not edit it. TensorFold-specific measurement lives in `tools/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. The default lists both live ones (`rocep1s0f1,roceP2p1s0f1`), as TensorFold's runbook recommends.
- Both ranks must get identical engine flags (`--context`, `--kv-dtype`, `--mtp-drafts`, `--mtp-confidence`, `--parallel`); TensorFold all-gathers them at startup and refuses a mismatch. `run.sh` builds both argvs from one `serve_args()`; keep it that way.
- Default thinking is off (`--no-thinking`), matching the vLLM sibling. Requests override it with `chat_template_kwargs.enable_thinking`.
- The image is built locally from `docker/Dockerfile`; `run.sh` refuses an image whose `tensorfold.sha` label differs from `TF_SHA`, and copies the head's image to the worker when the worker's image ID differs.

## Known engine limits (TensorFold 0.6.2, two ranks)

- One request at a time on the default image. Others queue (unbounded, not strictly FIFO). `--parallel N` with `--tp 2` is refused upstream. `PROFILE=concurrent` builds the image with `docker/patches/pr141-on-0.6.2.patch` (upstream PR #141 ported onto 0.6.2) and serves 8 at once. Never edit the patch by hand: regenerate it with `git diff v0.6.2 HEAD` from the port branch and re-run TensorFold's CUDA suites (`evidence/s6-pr141/gpu_tests.sh`) and `tools/probe_parallel.py` before shipping it.
- Each request decodes to `max_tokens` or EOS on both ranks. A client disconnect, a stop string, `tool_choice` gates and `thinking_budget` cuts do not free the engine early. Size `max_tokens` per request.
- A dead rank leaves the other in NCCL without a timeout. `/health` on rank 0 does not check rank 1. Restart both with `./stop.sh && ./run.sh`.
- No logprobs, no `n > 1`, no `/tokenize`, no vision, no NVFP4 or EXL3 checkpoints at two ranks.

## Host memory safety

- Keep `OOM_SCORE_ADJ=1000`, `MEMGUARD=1` and `--ulimit core=1`.
- Per rank at 262144 bf16: TensorFold's startup estimate is 48.9 GiB on the GPU, plus the 29.8 GiB n-gram tables mlocked in host memory. About 58 GiB (spark1) / 61 GiB (spark2) stays available after load and benches (`evidence/s7-default/free-after*.txt`).
- Before a heavy step (long-context needles, 128k prefill), run `free -h` on both nodes. Skip the step if spark1 MemAvailable < 6 GiB.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
shellcheck -S warning run.sh stop.sh tools/session_gate.sh tools/sweep.sh tools/run_t3.sh
```

After `./run.sh` is up: `GET /health` is 200, `GET /v1/models` lists `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, and `tools/session_gate.sh evidence/<id>` ends `GATE=PASS`.

## Never touch

- Live HF tokens
- An unpinned base image or an unpinned TensorFold ref
- Hand-edited generated README / `run.sh` blocks
- Advertising a window, concurrency level or speed that was not run here
