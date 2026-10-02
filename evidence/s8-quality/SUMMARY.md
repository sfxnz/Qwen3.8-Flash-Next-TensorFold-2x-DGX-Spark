# Session 8: quality on the shipped default (2026-10-02)

**Result.** T2 against the vLLM sibling's shipped default (session 8, `runs/session8/fp8/t2`), report-only because the checkpoints differ: GSM8K-250 95.2% vs 95.2%, IFEval-120 86.7% vs 88.3% (not significant), tools 52/60 vs 60/60 (8 losses on 4 prompts, p 0.008), JSON schema 30/30 vs 30/30, repeat-4gram 0.0003 vs 0.0005, effort 5/5. T3: 24/24 needles at 4k-128k and 6/6 at 246-250k prompt tokens.

- `t2.json`, `t2-manifest.json`: the capture (full rows in `~/projects/data/qwen38-evals/runs/tensorfold-s8/t2/`).
- `t2-vs-vllm-session8.txt`: `python3 quality/score.py t2 ~/projects/data/qwen38-evals/runs/session8/fp8/t2 <capture>` (exit 1: the tools rule fails against the reference).
- `t2-tools-failures.jsonl`: the 8 tool misses. 4 prompts, each failing the same way streamed and non-streamed. t27/t47 add the optional `formal: false`; t44 calls `book_restaurant` for a calendar event; t45 adds an unrequested `convert_currency` call.
- `t27-raw-reply.json`: the t27 reply re-requested with `return_token_ids`. Its tokens decode to `<parameter=formal>\nFalse\n</parameter>`, so the model emitted the argument and TensorFold's parser typed it as the schema's boolean. Whether the 4-bit weights or TensorFold's numerics cause these calls was not tested.
- `t3.out`, `t3.json`, `t3-250k.out`, `t3-250k.json`: needles via `tools/run_t3.sh` (needle documents sized with the checkpoint tokenizer inside the image). Cold needle wall time: 1.5 s at 4k, 6 s at 16k, 25 s at 64k, 56 s at 128k, 123 s at 246-250k.
