# Session 5: long context, MTP depth 6 vs 15, ABAB (2026-10-02)

**Result.** Depth 15 costs nothing at long context: cold prefill, resume TTFT and decode after the prompt match depth 6 within run-to-run spread at 8k, 33k and 131k prompt tokens. Cold prefill is about 2.8k tok/s up to 33k and 2.38k tok/s at 131k (55 s). An identical resend resumes in 0.07-0.26 s, and a new question on the same 131k system prompt starts in 0.28-0.32 s (TensorFold keeps prompt states at message boundaries).

- Driver: `evidence/s5-longctx/abab.sh` (one boot per config, one fixed filler `--run 100` so every boot sees the same text; a reboot clears kept states, so each boot's first request is cold).
- Tool: `tools/bench_prefill.py` (cold: filler as system message + a one-sentence question, 16 tokens; resend: identical; story: same system message + a 300-word story request, 256 tokens).
- An earlier attempt with a different filler per boot was discarded (it compared different texts); it is not in this repo.

Rows below are generated from each config's `bench-prefill.out`:

| Config | Prompt tokens | Cold prefill tok/s | Cold TTFT s | Resend TTFT s | Story TTFT s | Story decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| D6 | 8224 | 2818 | 2.92 | 0.07 | 0.12 | 61.5 |
| D6 | 32921 | 2814 | 11.70 | 0.10 | 0.14 | 65.2 |
| D6 | 131329 | 2375 | 55.30 | 0.26 | 0.28 | 55.7 |
| D15 | 8224 | 2830 | 2.91 | 0.08 | 0.12 | 61.0 |
| D15 | 32921 | 2816 | 11.69 | 0.11 | 0.15 | 66.1 |
| D15 | 131329 | 2377 | 55.24 | 0.25 | 0.28 | 55.5 |
| D6b | 8224 | 2825 | 2.91 | 0.07 | 0.13 | 60.6 |
| D6b | 32921 | 2818 | 11.68 | 0.11 | 0.14 | 66.0 |
| D6b | 131329 | 2377 | 55.24 | 0.22 | 0.32 | 55.1 |
| D15b | 8224 | 2811 | 2.93 | 0.07 | 0.12 | 60.4 |
| D15b | 32921 | 2817 | 11.69 | 0.11 | 0.15 | 65.6 |
| D15b | 131329 | 2378 | 55.23 | 0.24 | 0.28 | 55.0 |

These runs predate `tools/bench_prefill.py`'s warm-up request (added after the pre-PR review). The 8k and 33k cold rates are flat in every config (2,811-2,830 vs 2,814-2,818 tok/s), so no first-use cost shows in the first row.
