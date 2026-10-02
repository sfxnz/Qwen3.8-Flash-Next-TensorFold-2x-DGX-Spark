# Session 6: two-rank --parallel (upstream PR #141 ported onto 0.6.2) (2026-10-02)

**Result.** Shipped as the opt-in `PROFILE=concurrent`, not the default. With 8 users the aggregate is 2.3-4.8x the serial default, and the slowest first token drops from 16.6-22.0 s to 0.16-0.20 s. Every concurrent reply equals its solo run and its `"draft": false` run (bench_concurrent `--alone --serial`, 0 unequal in every cell). The cost is lone-request speed: −5 to −8% on repeated prompts (bench_concurrent c=1), −8 to −16% on the frozen-ruler and longer cells, and −29% on distinct prompts (`tools/bench_distinct.py`).

## The patch

`docker/patches/pr141-on-0.6.2.patch` = `git diff v0.6.2 HEAD` of branch `port/pr141-on-0.6.2` (4 commits on 56e2e3e):
- `4d534b2` port: PR #141 (3854dc4 + cb5101d, squashed) onto 0.6.2. Three files conflicted (geometry.py, engine.py, multi.py). Upstream 0.6.1/0.6.2 prompt-pass fixes kept in `multi_fill.PromptPasses`. Per-rank timing/arrival values are planned on rank 0 or disabled under two ranks. `Shadow.copy_prefix` was added for prefix forks, which the PR did not have.
- `4b7b3e7` review fixes: rank 1 survives admission errors rank 0 survives (was: rank 0 hangs in NCCL); a failed slot change on one rank fails both before any model collective; done streams are named in "drop" so stream lists stay equal.
- `5a49305`, `dd1bda6` keep the lone-graph slot's geometry across requests and give its spare rows back under memory pressure.

Each commit went through adversarial review (rank desync, one-GPU regression, exactness, graph-pointer safety); confirmed findings are fixed in the commits above.

## Checks

- TensorFold's own CUDA suites on one GB10 (two ranks as threads, synthetic checkpoints): `gpu-tests-pr141-v2.txt` 160 passed (36 in `test_flashnext_tp_multi.py`); plain 0.6.2 on the shared suites: `gpu-tests-0.6.2.txt`, same counts.
- Real two-rank serve, depth 15, `--parallel 8`: `P8-D15-v2/` (gate PASS, `concurrent.out` exactness, `probe-parallel.out` PASS: a client leaving mid-stream does not disturb 3 others, a `response_format` request gets HTTP 400, serving continues; no OutOfStep or Traceback in either rank's log).
- `P8-D15/` is the first port (before the lone-slot commits); `P8-D6-v2/` is a throughput-only A/B at depth 6 (no exactness flags).

## Why lone requests are slower

Not isolated. The lone-slot commits did not move it (P8-D15 → P8-D15-v2 is within noise). Repeated prompts speed up run to run on the profile (frozen prose c=1 in `P8-D15-v2/bench-frozen.out`: 42.0, 59.3, 66.0 tok/s; per `requests.log` 43.3 → 30.7 → 27.5 ms per round against 26.0 serial), so part of it is per-prompt warm-up in the concurrent decoder, for example the lone-graph slot recapturing graphs when the slot holds another conversation's kept prompt (left open in the port; see `docker/patches/README.md`). Every two-rank round also adds rank 0's host planning, a TCP message and a digest all-gather; that cost was not measured on its own.

Distinct-prompt lone bench (`tools/bench_distinct.py`, 12 prompts, 256 tokens, greedy):
- serial default (`../s7-default/bench-distinct.out`): `SUMMARY {"n": 12, "median_decode_tok_s": 73.91493612918684, "median_ttft_s": 0.10516041750088334}`
- concurrent, depth 15: `SUMMARY {"n": 12, "median_decode_tok_s": 52.17831709604151, "median_ttft_s": 0.10978490102570504}`
- concurrent, depth 6: `SUMMARY {"n": 12, "median_decode_tok_s": 50.3004467718347, "median_ttft_s": 0.10916679992806166}`

## Users 1-8 (`tools/vendor/bench_concurrent.py --levels 1,2,4,8 --alone --serial --reps 3`)

Generated from `../s7-default/concurrent.out`, `P8-D15-v2/concurrent.out` and `P8-D6-v2/concurrent.out`:

| Prompt | Sampling | Users | Serial default: aggregate tok/s (worst TTFT) | PROFILE=concurrent, depth 15 | depth 6 (`MTP_DRAFTS=6`, throughput only) |
|---|---|---:|---:|---:|---:|
| chat | greedy | 1 | 89.9 (0.06 s) | 83.8 (0.06 s) | 85.0 (0.06 s) |
| chat | greedy | 2 | 88.3 (2.99 s) | 151.3 (0.07 s) | - |
| chat | greedy | 4 | 88.2 (8.88 s) | 260.4 (0.1 s) | - |
| chat | greedy | 8 | 88.1 (20.49 s) | 424.8 (0.18 s) | 440.9 (0.17 s) |
| code | greedy | 1 | 110.8 (0.06 s) | 105.3 (0.06 s) | 107.5 (0.06 s) |
| code | greedy | 2 | 111.7 (2.4 s) | 190.4 (0.07 s) | - |
| code | greedy | 4 | 109.3 (7.12 s) | 323.0 (0.09 s) | - |
| code | greedy | 8 | 108.5 (16.58 s) | 519.6 (0.16 s) | 545.6 (0.15 s) |
| chat | T 1.0, top_k 20 | 1 | 81.2 (0.06 s) | 74.7 (0.07 s) | 75.4 (0.06 s) |
| chat | T 1.0, top_k 20 | 2 | 80.9 (3.29 s) | 109.6 (0.07 s) | - |
| chat | T 1.0, top_k 20 | 4 | 81.6 (9.65 s) | 162.2 (0.1 s) | - |
| chat | T 1.0, top_k 20 | 8 | 81.7 (22.02 s) | 228.2 (0.18 s) | 236.9 (0.17 s) |
| code | T 1.0, top_k 20 | 1 | 110.4 (0.05 s) | 102.7 (0.06 s) | 110.0 (0.06 s) |
| code | T 1.0, top_k 20 | 2 | 112.8 (2.43 s) | 146.2 (0.07 s) | - |
| code | T 1.0, top_k 20 | 4 | 104.9 (7.65 s) | 168.2 (0.17 s) | - |
| code | T 1.0, top_k 20 | 8 | 99.9 (18.18 s) | 234.3 (0.2 s) | 257.0 (0.12 s) |

## Frozen ruler and longer cells, serial default vs concurrent (depth 15)

Generated with `python3 tools/compare.py evidence/s7-default/rerun evidence/s6-pr141/P8-D15-v2`:

| metric | rerun | P8-D15-v2 |
|---|---:|---:|
| prose c1 | 69.6 | 59.3 (-14.8%) |
| prose c2 | 70 | 61 (-12.8%) |
| prose c2 agg | 92.1 | 122 (+32.6%) |
| structured c1 | 250 | 227 (-8.9%) |
| structured c2 | 247 | 214 (-13.4%) |
| structured c2 agg | 318 | 431 (+35.7%) |
| code | 130 | 112 (-14.0%) = |
| prose_long | 81.4 | 68.3 (-16.1%) = |
| json | 157 | 137 (-12.7%) = |
| multilingual | 68.3 | 62.9 (-7.9%) = |
| chat_sampled | 71.9 | 66.3 (-7.8%) = |
| frozen tokens/round | 4.33 | 4.32 (-0.3%) |
