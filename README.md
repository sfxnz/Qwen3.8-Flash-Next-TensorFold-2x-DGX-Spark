# Qwen3.8-Flash-Next · TensorFold · 2× DGX Spark

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2 with the [TensorFold](https://github.com/ashhart/TensorFold) inference engine. Not vLLM. The vLLM recipe for the NVFP4 checkpoint is [Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark).

Qwen3.8-Flash-Next is a ~180B MoE (512 experts, top-10) with Gated DeltaNet, sparse attention, hyper-connections, hashed n-gram (PLE) tables and one MTP layer. Native context is 262,144 tokens; this recipe serves all of it on both ranks. The checkpoint is MLX affine 4-bit (group 32) on every linear, including the n-gram tables and the MTP head. It is the only Flash Next format TensorFold 0.6.2 serves on two ranks: NVFP4 and EXL3 exports run on one GPU only.

TensorFold `56e2e3e` (v0.6.2) runs inside `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned), built locally from [`docker/Dockerfile`](docker/Dockerfile). Its kernels JIT-compile for sm_121 on the first start and are cached on the host after that. Rank 1 runs on `spark2`, rank 0 serves HTTP on `spark1`; partials are all-gathered over NCCL on the QSFP RoCE link.

Pinned snapshot: `2b170fa6309d5d1ee380b35636075fac7945f286`. Engine credit: TensorFold contributors (Apache-2.0). Checkpoint credit: the TensorFold org's MLX 4-bit + MTP conversion of Qwen3.8-Flash-Next.

## Measured on 2× DGX Spark (L.A.I.L lab)

`python3 bench_decode.py`, byte-identical to the vLLM sibling's frozen ruler (sha256 `6a9c64bd…`). Do not copy community tok/s into this table. Full gate pack: [`evidence/s7-default/`](evidence/s7-default/).

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: streamed greedy, thinking off, max_tokens 200 (prose ends at EOS near 100), 3-run median; TensorFold 0.6.2 at TP=2, context 262144, kv bf16, MTP up to 15 drafts at confidence 0.70, 4.33 tokens per verify round; second gate on the boot (the first gate on that boot measured up to 5.6% lower; s4's first-after-boot gate did not).

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 69.6 | 69.6 | 0.05 s |
| prose (note 1) | 2 | 70.0 | 92.1 | 0.78 s |
| structured | 1 | 249.5 | 249.3 | 0.07 s |
| structured (note 2) | 2 | 246.8 | 317.5 | 0.49 s |

1. TensorFold 0.6.2 serves one request at a time on two ranks. At c=2 the second stream waits for the first, which is its TTFT, and bench_decode's aggregate (tokens / (wall − median TTFT)) is not concurrent throughput. TTFT p50 is the midpoint of the first stream's 0.05-0.06 s and the queued second stream's 1.51-1.53 s (prose) or 0.93-0.94 s (structured). The --parallel profile serves both together (Concurrency).
2. Same as note 1.
<!-- END generated measured -->

The default MTP draft cap is 15 (TensorFold's own default is 6). Each verify round now emits 4.33 tokens on the frozen ruler instead of 3.56. Against depth 6 on the same day: structured +42-46%, JSON +10% and code +4%. Prose, long prose, multilingual text and sampled chat moved within ±1%, because the 70% confidence rule still stops a chain early on uncertain text. Greedy replies were byte-identical at every depth: drafting does not change the tokens ([`evidence/s4-depth/`](evidence/s4-depth/), against a fresh baseline in the same sweep; depth 15 repeated in s7). At 8k-131k context, depth 15 and depth 6 decode at the same speed ([`evidence/s5-longctx/`](evidence/s5-longctx/)).

### Longer cells (c=1)

`tools/bench_cells.py`, the same session as the table above. 3-run medians, streamed, thinking off. Tokens per round come from TensorFold's `/health` counters.

| Cell | Reply tokens | Decode tok/s | Tokens per round |
|---|---:|---:|---:|
| code (Python module + tests) | 512 | 129.8 | 5.07 |
| JSON (15 records) | 768 | 157.1 | 7.11 |
| prose, ~400 words | 512 | 81.4 | 2.31 |
| German story + French summary | 512 | 68.3 | 1.77 |
| chat, sampled (T 0.7, top_k 20) | 504 | 71.9 | 1.88 |

### Long context

`tools/bench_prefill.py`. The filler is the system message; three requests per length; greedy; thinking off; depth 15. [`evidence/s5-longctx/`](evidence/s5-longctx/) (two boots, D15 and D15b: cold prefill and decode within 1.1%, resend TTFT within 0.01 s).

| Prompt tokens | Cold prefill | Cold TTFT | Identical resend TTFT | New question on the same system prompt, TTFT | Decode tok/s at that context (story, 256 tokens) |
|---:|---:|---:|---:|---:|---:|
| 8,224 | 2,811-2,830 tok/s | 2.9 s | 0.07-0.08 s | 0.12 s | 60.4-61.0 |
| 32,921 | 2,816-2,817 tok/s | 11.7 s | 0.11 s | 0.15 s | 65.6-66.1 |
| 131,329 | 2,377-2,378 tok/s | 55.2 s | 0.24-0.25 s | 0.28 s | 55.0-55.5 |

TensorFold keeps prompt states one token before a prompt's end and at message boundaries. An identical resend or a new user turn after the same system prompt resumes from that state instead of prefilling again: 131k tokens become 0.3 s.

### Quality

`quality/` is the vLLM sibling's harness, run against this serve (default settings, thinking off unless a task says otherwise, greedy). The comparison column is that recipe's shipped default (`nvidia/Qwen3.8-Flash-Next-NVFP4`, vLLM 0.30, session 8). Different checkpoints, so this is a report, not a pass/fail gate. Receipts: [`evidence/s8-quality/`](evidence/s8-quality/).

| Task | TensorFold, MLX 4-bit | vLLM sibling, NVFP4 |
|---|---:|---:|
| GSM8K-250 (thinking off) | 95.2% (238/250) | 95.2% |
| IFEval-120 (strict, prompt level) | 86.7% (104/120) | 88.3% |
| Tool calls, exact name and arguments (30 × non-streamed + streamed) | 52/60 | 60/60 |
| `response_format` json_schema, strict | 30/30 | 30/30 |
| Repeated word-4-grams over 64 × 512-token replies | 0.03% | 0.05% |
| `reasoning_effort` unset / none / low / medium / xhigh | 5/5 | 5/5 |
| Needles, 4k / 16k / 64k / 128k × 3 depths × 2 | 24/24 | 24/24 on a different grid: 4k / 16k / 32k / 64k (its `evidence/fp8-default/`); 128k not run |
| Needles at 246-250k prompt tokens (94-95% of the window) × 3 depths × 2 | 6/6 | not run |

The 8 tool misses are 4 prompts, each failing the same way streamed and non-streamed. In 2 the model adds an optional argument (`"formal": false`). In 1 it picks the wrong tool, and in 1 it makes a second, unrequested call. The model emitted these calls: the raw token ids decode to `<parameter=formal>\nFalse\n</parameter>`, and TensorFold's parser typed them faithfully ([`t27-raw-reply.json`](evidence/s8-quality/t27-raw-reply.json), [`t2-tools-failures.jsonl`](evidence/s8-quality/t2-tools-failures.jsonl)). Whether the 4-bit weights or TensorFold's numerics lead the model there was not tested: that needs this checkpoint on another engine, or TensorFold on another checkpoint. If exact tool arguments matter more than speed, the vLLM NVFP4 recipe scored 60/60.


## Concurrency (opt-in)

TensorFold 0.6.2 serves one request at a time on two ranks. `PROFILE=concurrent ./run.sh` serves up to 8 at once. It builds `tf-qwen38-flashnext:0.6.2-pr141` with [`docker/patches/pr141-on-0.6.2.patch`](docker/patches/pr141-on-0.6.2.patch): upstream PR #141 (`--parallel N` on two ranks; open, not in a release) ported onto 0.6.2, with review fixes. Each round, rank 0 plans the admission and the batch, rank 1 replays the plan, and both ranks check a digest of it before any collective. [`evidence/s6-pr141/`](evidence/s6-pr141/) has the port history, TensorFold's CUDA test suites (160 passed) and the failure probes.

`tools/vendor/bench_concurrent.py` (TensorFold's), 256-token replies, 3 reps. Aggregate tok/s, with the slowest first token in brackets:

| Users | Code, greedy: serial default | Code, greedy: `PROFILE=concurrent` | Chat, T 1.0: serial default | Chat, T 1.0: `PROFILE=concurrent` |
|---:|---:|---:|---:|---:|
| 1 | 110.8 (0.06 s) | 105.3 (0.06 s) | 81.2 (0.06 s) | 74.7 (0.07 s) |
| 2 | 111.7 (2.40 s) | 190.4 (0.07 s) | 80.9 (3.29 s) | 109.6 (0.07 s) |
| 4 | 109.3 (7.12 s) | 323.0 (0.09 s) | 81.6 (9.65 s) | 162.2 (0.10 s) |
| 8 | 108.5 (16.58 s) | 519.6 (0.16 s) | 81.7 (22.02 s) | 228.2 (0.18 s) |

Every concurrent reply had the same token sha as the same request alone and as a `"draft": false` run, in every cell (0 unequal). A client that leaves mid-stream does not disturb the others. A `response_format` request gets HTTP 400 in this profile: two-rank concurrency serves no grammars, logprobs or images.

At 8 users the aggregate is 2.3-4.8x the serial default. The cost is a lone request, 5-29% slower depending on the traffic: 5-8% on repeated prompts (the table's 1-user row), 8-16% on the frozen ruler's prose and the longer cells, and 52.2 against 73.9 tok/s with a different prompt every time (`tools/bench_distinct.py`). The cause is not isolated. Repeated prompts speed up run to run (frozen prose c=1: 42.0, 59.3, 66.0 tok/s), so part of it is per-prompt warm-up in the concurrent decoder. Every two-rank round also adds rank 0's host planning, a TCP message and a digest all-gather, but that cost was not measured on its own. Keep the serial default for one user at a time; use the profile when several clients share the pair.

The profile keeps 15 drafts, where exactness was checked. `MTP_DRAFTS=6` measured 4-10% more aggregate at 8 users (`evidence/s6-pr141/P8-D6-v2/`, throughput only, without the exactness flags).


## Against the vLLM recipe, same day, same harness

The sibling [vLLM recipe](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark) was restored with its own `run.sh` after these tests (`d66a95e`: v0.30.0 + overlays, `nvidia/Qwen3.8-Flash-Next-NVFP4`, MTP-3, FP8 dense). Its image, env, argv and mount points match the pre-test serve on both nodes ([`restore-config-diff.txt`](evidence/s9-vllm-same-harness/restore-config-diff.txt)). This recipe's tools were then run against it. The weights differ: MLX 4-bit on every linear here, NVFP4 experts with FP8/bf16 elsewhere there. See Quality for what that costs. Receipts: [`evidence/s9-vllm-same-harness/`](evidence/s9-vllm-same-harness/).

One stream, decode tok/s:

| Cell | TensorFold (this recipe) | vLLM sibling, same day | Ratio |
|---|---:|---:|---:|
| frozen `bench_decode.py` prose c=1 | 69.6 | 46.5 | 1.50x |
| frozen `bench_decode.py` structured c=1 | 249.5 | 82.4 | 3.03x |
| code, 512 tokens | 129.8 | 70.0 | 1.85x |
| prose, 512 tokens | 81.4 | 51.4 | 1.58x |
| JSON, 768 tokens | 157.1 | 82.5 | 1.90x |
| German + French, 512 tokens (vLLM ended at EOS at 488) | 68.3 | 47.0 | 1.45x |
| chat, sampled, ~500 tokens | 71.9 | 44.9 | 1.60x |
| 12 distinct prompts, 256 tokens | 73.9 | 46.5 | 1.59x |

The vLLM recipe's README states prose c=1 56.9 tok/s (its session 8). Today, on a fresh boot of the same config, its frozen ruler measured 40.8 and then 46.5 (prose acceptance 2.69, the same as its README). The ratio above uses 46.5.

Long context (`tools/bench_prefill.py`; depth 15 here, MTP-3 there). vLLM's row is the run with the tool's warm-up request and a fresh filler (`--run 101`). Its first run, without the warm-up, measured 1,758 tok/s at 8k (`bench-prefill-nowarmup.out`). TensorFold's rows ran before the warm-up existed; its 8k and 33k cold rates are flat (2,830 / 2,816), which suggests no first-use cost there.

| Prompt tokens (TensorFold / vLLM) | Cold prefill tok/s | Identical resend TTFT | New question on the same system prompt, TTFT | Decode tok/s at that context |
|---:|---:|---:|---:|---:|
| 8,224 / 8,193 | 2,830 / 2,977 | 0.08 s / 0.64 s | 0.12 s / 0.66 s | 61.0 / 49.4 |
| 32,921 / 32,841 | 2,816 / 2,754 | 0.11 s / 1.03 s | 0.15 s / 1.03 s | 66.1 / 45.1 |
| 131,329 / 131,559 | 2,377 / 2,033 | 0.25 s / 1.58 s | 0.28 s / 1.59 s | 55.5 / 46.2 |

Several users (`tools/vendor/bench_concurrent.py`). The TensorFold runs used `--alone`, which sends each request alone before its wave, so their prompts were warm. vLLM's ran without it (vLLM returns no token sha to check), so its slowest-first-token column includes cold first reps. Compare aggregates, not first tokens:

| 8 users, 256 tokens: aggregate tok/s (slowest first token) | TensorFold serial default | TensorFold `PROFILE=concurrent` | vLLM sibling (`--max-num-seqs 8`) |
|---|---:|---:|---:|
| code, greedy | 108.5 (16.58 s) | 519.6 (0.16 s) | 295.4 (0.16 s) |
| chat, greedy | 88.1 (20.49 s) | 424.8 (0.18 s) | 242.1 (0.31 s) |
| code, T 1.0 | 99.9 (18.18 s) | 234.3 (0.2 s) | 247.5 (0.34 s) |
| chat, T 1.0 | 81.7 (22.02 s) | 228.2 (0.18 s) | 206.1 (0.4 s) |

For one user at a time, TensorFold's default decodes 1.45-3.0x vLLM's rate. A resend or new question after a long prompt starts 5-9x sooner. Cold prefill is level with vLLM to 33k tokens (vLLM +5% at 8k, TensorFold +2% at 33k) and 17% faster at 131k. With several users, the serial default falls behind vLLM by 2.5-2.7x and queues requests. `PROFILE=concurrent` closes that: 1.75x vLLM greedy and 0.95-1.1x sampled at 8 users.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes
- About 115 GB free disk per node for the weights (113.2 GB snapshot), plus 24.5 GB for the image ([`footprint.txt`](evidence/s7-default/footprint.txt))
- SSH from the head node to the worker (`spark2` in this lab)
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up; `run.sh` refuses to.

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Image

`run.sh` builds `tf-qwen38-flashnext:0.6.2` from [`docker/Dockerfile`](docker/Dockerfile) on the head when it is missing, and copies it to the worker over the link when the worker's image ID differs. To build by hand:

```bash
docker build -t tf-qwen38-flashnext:0.6.2 docker/
```

The build clones TensorFold at `TF_SHA`, applies an optional patch from `docker/patches/` (pinned by sha256 in `run.sh` and `recipe.yaml`), and pip-installs it with `xgrammar` 0.2.8 under [`docker/constraints.txt`](docker/constraints.txt). Those are the package versions the measured images resolved ([`pip-freeze-measured.txt`](evidence/s7-default/pip-freeze-measured.txt)). The build fails if pip replaces the base image's torch or triton. `xgrammar` is needed on both ranks: rank 1 recompiles every `response_format` grammar, and without `xgrammar` it would exit and leave rank 0 waiting in NCCL. `run.sh` refuses an image whose `tensorfold.sha` / `tensorfold.patch_sha` labels differ from `TF_SHA` / `TF_PATCH`.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
VALIDATE_ONLY=1 ./run.sh   # checks the defaults, no Docker
./run.sh
```

The head checks that the weights are complete on its disk: every shard's safetensors header must end at its file size. It builds or copies the image, then starts rank 1 on `spark2`, where `run.sh` makes the same weight check. Then it starts rank 0 and waits for `/health` and `/v1/models`. It checks the worker container every ~10 s while it waits. If SSH to `WORKER_HOST` fails, it exits 1 instead of starting a TP=2 head alone. Every serve setting is forwarded to the worker shell-quoted; both ranks get their engine flags from one function, because TensorFold refuses ranks with different settings.

A first start downloads 113 GB per node and JIT-compiles the CUDA kernels (TensorFold reported loaded in 120-133 s). Later starts load in 32-42 s (the `loaded in` lines in `evidence/*/startup.txt`). The containers run with `HF_HUB_OFFLINE=1` and no HF token.

If SSH is not set up, start the worker yourself, then the head without orchestration (the same settings on both):

```bash
# spark2
ROLE=worker ./run.sh

# spark1
ORCHESTRATE=0 ./run.sh
```

With `PROFILE=concurrent`, set it on both nodes. The worker then needs `docker/patches/` next to its `run.sh`, as in a full checkout.

Smoke test (thinking is off by default):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Correctness probes and the gate pack against the live API:

```bash
python3 smoke_thinking.py --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP
python3 smoke_tools.py --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP
python3 smoke_count.py --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP
python3 bench_decode.py --model TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP
tools/session_gate.sh evidence/<id>      # all of the above, plus bench_cells, acceptance and receipts
```

Stop both ranks from the head:

```bash
./stop.sh
```

`stop.sh` stops rank 0 first with SIGTERM: it closes the HTTP server and the rendezvous store. Rank 1 then exits by itself, and the script waits for it (up to `STOP_TIMEOUT`, 30 s) before stopping and removing its container.

## Defaults

`recipe.yaml` is the source of truth. Edit it, then run `python3 kit/render.py`. CI fails when this table or the `run.sh` block drifts from it.

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Engine | TensorFold `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21` (v0.6.2), built into `tf-qwen38-flashnext:0.6.2` from `docker/Dockerfile` |
| Model | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` at `2b170fa6309d5d1ee380b35636075fac7945f286` |
| Ranks | `--tp 2`: rank 1 on `spark2` first, then rank 0 (HTTP) on the head; rendezvous `10.100.8.1:29551` |
| `--context` | 262144 (the native window, on both ranks) |
| `--kv-dtype` | `bf16` |
| MTP drafts | `--mtp-drafts 15 --mtp-confidence 0.70` |
| `--parallel` | 1 (0.6.2 serves one request at a time on two ranks) |
| Default thinking | off (`THINKING=0`); requests override with `chat_template_kwargs.enable_thinking` |
| `--max-tokens` | 4096 (the reply cap when a request sets none) |
| NCCL | `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1` |
| API | `http://<head>:8000/v1`, served as `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` |
| Container | `tf-qwen38-flashnext` |
<!-- END generated defaults -->

`run.sh` refuses, before `VALIDATE_ONLY` exits:

- non-decimal or zero-padded integers, `MTP_CONFIDENCE` outside [0, 1], a `TF_SHA` / `SNAPSHOT_SHA` that is not 40 hex
- `TP` other than 1 or 2, `CONTEXT` above the native 262144 (TensorFold serves no YaRN), `MTP_DRAFTS` above the engine cap 15, `KV_DTYPE` other than bf16 / int8 / int4
- `PARALLEL` above 1 at `TP=2` unless `TF_PATCH=patches/pr141-on-0.6.2.patch` (see Concurrency)
- `EXTRA_ARGS` that re-sets a flag `run.sh` builds (`--tp`, `--rank`, `--master*`, `--host`, `--port`, `--name`, `--context`, `--kv-dtype`, `--mtp-*`, `--parallel`, `--thinking`, `--max-tokens`, `--no-drafts`), `--vision*` (one GPU only), and `--prefill-fp8` / `--drafter` / `--ple-on-ssd` / `--ssd-experts` (refused or unused for this checkpoint)
- an `EXTRA_ENV` word that is not `KEY=VALUE`

At start it also refuses another running GPU container on either node, a busy port, an image whose labels do not match, and an incomplete snapshot when `SKIP_DOWNLOAD=1`. `BENCH_ONLY=1` binds the API to 127.0.0.1. `MTP_DRAFTS=0` serves without drafts.

## Not supported (TensorFold 0.6.2 on two ranks)

- One request at a time on the default image. Others queue in an unbounded, not strictly FIFO wait; a queued streaming client has its `200` and role chunk and then waits. See Concurrency for the opt-in `--parallel` image.
- Each request decodes to `max_tokens` or EOS on both ranks. A client disconnect, a stop string, a forced `tool_choice` and a `thinking_budget` cut stop what is sent, not the GPU work, so the next request waits. Set `max_tokens` per request. `MAX_TOKENS` (default 4096) applies when a request sets none.
- A rank that dies mid-request leaves the other waiting in NCCL with no timeout, and `/health` on rank 0 does not check rank 1. Restart with `./stop.sh && ./run.sh`.
- No logprobs, no `n > 1`, no `/tokenize`, no images, no presence/frequency penalties (ignored).
- The reasoning field is `reasoning_content` (not `reasoning`). Usage arrives on the final stream chunk, whether or not `stream_options` asks for it. The default seed is a hash of the prompt, so identical sampled requests repeat unless they carry a `seed`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1,roceP2p1s0f1
export PORT=8000
export CONTEXT=262144
```

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN; unpinned NCCL can pick a dead one. Both live HCAs (`rocep1s0f1,roceP2p1s0f1`) are TensorFold's recommendation; one HCA decoded within noise here ([`evidence/s3-levers/H1`](evidence/s3-levers/H1/)). `NCCL_PROTO=LL` / `LL128` did not help decode (−3.6% to +1.0% per cell, [`evidence/s3-levers/`](evidence/s3-levers/)). `TF_CACHE` holds the kernel builds (default `~/.cache/tensorfold-qwen38`). `MEMORY_RESERVE_GIB` sets TensorFold's startup reserve (default max(4 GiB, 10% of RAM)).

Memory per rank at the full window: TensorFold's startup estimate is 48.9 GiB, plus the 29.8 GiB n-gram tables, which it mlocks in host memory (`--ulimit memlock` and `IPC_LOCK` are passed for that). About 58-61 GiB stays available per node after load and benches ([`evidence/s7-default/`](evidence/s7-default/)).

## Logs

```bash
docker logs -f tf-qwen38-flashnext
ssh spark2 docker logs -f tf-qwen38-flashnext
```

TensorFold prints one `done req-…` line per request: tokens, tok/s, TTFT, prefill time, verify rounds, accepted drafts and the reply's token sha. `GET /health` carries cumulative counters; `GET /metrics` is Prometheus with the `tensorfold:` prefix.

## Evidence

Every number above has a file under [`evidence/`](evidence/). `recipe.yaml` names the file per measured row. `python3 kit/render.py --check` lists the rows that still have none. Sessions:

| Session | What | Verdict |
|---|---|---|
| [`s1-first-boot`](evidence/s1-first-boot/) | First TP=2 boot by hand, smokes, frozen ruler | TensorFold runs on the pair |
| [`s2-baseline`](evidence/s2-baseline/) | B0 through `run.sh` and `tools/session_gate.sh` | baseline (depth 6) |
| [`s3-levers`](evidence/s3-levers/) | One HCA, NCCL LL / LL128, depth 4/8/10, confidence 0.50/0.85, baseline repeat | depth up is the lever; the rest is noise or worse |
| [`s4-depth`](evidence/s4-depth/) | Depth 8/10/12/15 with repeats against a fresh baseline | depth 15 adopted |
| [`s5-longctx`](evidence/s5-longctx/) | Depth 6 vs 15 at 8k/33k/131k, ABAB | no long-context cost |
| [`s6-pr141`](evidence/s6-pr141/) | `--parallel` on two ranks: TensorFold's CUDA suites, gate, 1-8 users, failure probes | see Concurrency |
| [`s7-default`](evidence/s7-default/) | Final default image and settings; serial 1-8 user baseline | published numbers |
| [`s8-quality`](evidence/s8-quality/) | T2 (GSM8K, IFEval, tools, JSON, repetition, effort) and T3 needles to 250k | see Quality |
| [`s9-vllm-same-harness`](evidence/s9-vllm-same-harness/) | vLLM sibling restored and checked, then measured with this recipe's tools | see "Against the vLLM recipe" |

## Gotchas

- TensorFold opens HTTP only after the model is loaded, so "connection refused" means still loading.
- One first-after-boot gate (s7) measured up to 5.6% lower than a repeat on the same boot; another (s4 D15) did not. The published table is the repeat.
- `docker stop` on a rank without `--init` would wait out its timeout: TensorFold's rank 1 installs no SIGTERM handler. `run.sh` passes `--init`.
- The `TensorFold/` repo id prints a cosmetic "untested" note at start (TensorFold lists the checkpoint under its old `Vontra/` name).

## Credits

- Engine: [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0 from 0.6.0; earlier code MIT), tag v0.6.2. `tools/vendor/bench_concurrent.py` is TensorFold's, unmodified.
- Concurrency patch: upstream PR #141 by Bill H. (BHCC2025) and ashhart, ported onto 0.6.2 here with review fixes ([`docker/patches/`](docker/patches/)).
- Checkpoint: [TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP); base model Qwen3.8-Flash-Next.
- Harness: `bench_decode.py`, the smokes and `quality/` come from the vLLM sibling recipe.

## License

The recipe's own files are MIT ([`LICENSE`](LICENSE)). `docker/patches/pr141-on-0.6.2.patch` changes TensorFold source: Apache-2.0, provenance and changes in [`docker/patches/README.md`](docker/patches/README.md). `tools/vendor/bench_concurrent.py` is TensorFold's, unmodified: MIT and Apache-2.0, per its header. Upstream's notice is in [`NOTICE`](NOTICE) and the license texts are in [`LICENSES/`](LICENSES/). Model weights follow the source model license on Hugging Face.
