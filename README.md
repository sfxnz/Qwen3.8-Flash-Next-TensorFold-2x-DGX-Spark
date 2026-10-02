# TODO-model · vLLM · 2× DGX Spark

Serve [TODO-org/TODO-model](https://huggingface.co/TODO-org/TODO-model) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

TODO: one paragraph. Total and active parameters, quantization, native context, the context this recipe serves, the drafter if any. Numbers first.

Stock `vllm/vllm-openai` TODO: what breaks on sm_121. Build `TODO-local-image-tag` from `docker/` first.

Pinned snapshot: `TODO-40-hex-snapshot-sha`.

## Hardware

- Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes
- About TODO GiB free disk per node for the weights
- SSH from the head node to the worker (`spark2` in this lab)
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up.

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Quick start

On both nodes, from this repo:

```bash
docker build -f docker/Dockerfile.TODO -t TODO-local-image-tag docker
```

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
VALIDATE_ONLY=1 ./run.sh   # checks the defaults, no Docker
./run.sh
```

The head script copies itself to `spark2`, starts the worker, waits 25s, then starts rank 0. If SSH to `WORKER_HOST` fails it exits 1 instead of starting a TP=2 head alone.

If SSH is not set up, start the worker yourself, then the head:

```bash
# spark2
ROLE=worker ./run.sh

# spark1
ROLE=head ./run.sh
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TODO-org/TODO-model",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Stop both ranks from the head:

```bash
./stop.sh
```

## Defaults

`recipe.yaml` is the source of truth. Edit it, then run `python3 kit/render.py`. CI fails when this table or the `run.sh` block drifts from it.

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `TODO-local-image-tag` (local, built from `docker/`) |
| Model | `TODO-org/TODO-model` |
| Checkpoint | `TODO-40-hex-snapshot-sha` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 32768 |
| `--max-num-seqs` | 2 |
| `--kv-cache-dtype` | `auto` |
| `--kv-cache-memory` | not pinned yet (`KV_CACHE_MEMORY` is empty; vLLM decides) |
| `--block-size` | vLLM default (`BLOCK_SIZE` is empty) |
| CUDA graphs | on (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Speculative | none (`SPEC_CONFIG` is empty) |
| API | `http://<head>:8000/v1` |
| Container | `TODO-container-name` |
| Master port | 29500 |
<!-- END generated defaults -->

TODO: the refusals `run.sh` enforces and why (what this model cannot do on GB10).

## Measured on 2× DGX Spark (TODO lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens, 3-run median. TODO: pin, context, drafter, graphs. Prose is the low-acceptance regime. Structured (count 1→200) is the high-acceptance regime. `python3 bench_decode.py` repeats both phases at c=1,2.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | TODO | TODO | TODO s |
| prose | 2 | TODO | TODO | TODO s |
| structured | 1 | TODO | TODO | TODO s |
| structured | 2 | TODO | TODO | TODO s |
<!-- END generated measured -->

TODO: prefill and needle results with prompt token counts.

## Agent-readiness probes

TODO: one result per probe. Each has a receipt under `evidence/`.

| Probe | Result |
|---|---|
| Thinking off, no `<think>` leak in `content` | TODO |
| Tool call parsed (`get_weather`) | TODO |
| Tool follow-up (`role: tool`) answers without a think leak | TODO |
| Greedy count 1→200 consecutive | TODO |
| Unique-salt needle at TODO prompt tokens | TODO |

## Evidence

Every number above has a file under [`evidence/`](evidence/). `recipe.yaml` names the file per measured row. `python3 kit/render.py --check` lists the rows that still have none. See [`evidence/README.md`](evidence/README.md).

## Gotchas

- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`.
- Do not use stock `vllm/vllm-openai` on sm_121.
- TODO: what this model cannot do on GB10 and the `run.sh` refusal that encodes it.

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (TODO).
