# Session 7: the shipped default (2026-10-02)

**Result.** Final image `tf-qwen38-flashnext:0.6.2` (git clone of 56e2e3e + xgrammar 0.2.8, labels checked by `run.sh`) with `MTP_DRAFTS=15`, everything else at the `recipe.yaml` defaults. Gate PASS twice on one boot. The README table uses `rerun/`, the second gate on the boot. The first gate on the boot (`./`) ran up to 5.6% lower on some cells (cause not isolated; s4's D15, also first after its boot, did not). `rerun/` matches session 4's D15 within 1.2%.

- `concurrent.out`: serial 1-8 user baseline (`tools/vendor/bench_concurrent.py --levels 1,2,4,8 --alone --serial --reps 3`). Aggregate stays at 81-113 tok/s and the slowest first token at 8 users is 16.6-22.0 s. All replies equal their solo and serial runs.
- `bench-distinct.out`: 12 distinct lone prompts, 256 tokens: the reference for the concurrency profile.
- Memory after load and benches: 58 GiB (spark1) / 61 GiB (spark2) available (`free-after*.txt`).

Generated with `python3 tools/compare.py evidence/s4-depth/D15 evidence/s7-default evidence/s7-default/rerun`:

| metric | D15 | s7-default | rerun |
|---|---:|---:|---:|
| prose c1 | 70.4 | 68.7 (-2.5%) | 69.6 (-1.2%) |
| prose c2 | 70.3 | 70.5 (+0.2%) | 70 (-0.5%) |
| prose c2 agg | 92.4 | 92.8 (+0.4%) | 92.1 (-0.4%) |
| structured c1 | 250 | 242 (-3.2%) | 250 (-0.1%) |
| structured c2 | 247 | 241 (-2.1%) | 247 (+0.1%) |
| structured c2 agg | 321 | 314 (-2.4%) | 318 (-1.2%) |
| code | 130 | 122 (-5.4%) = | 130 (+0.2%) = |
| prose_long | 81.2 | 81.4 (+0.2%) = | 81.4 (+0.2%) = |
| json | 156 | 155 (-0.4%) = | 157 (+1.0%) = |
| multilingual | 68.2 | 67.9 (-0.5%) = | 68.3 (+0.1%) = |
| chat_sampled | 72.2 | 71.3 (-1.1%) = | 71.9 (-0.4%) = |
| frozen tokens/round | 4.33 | 4.33 (+0.0%) | 4.33 (+0.0%) |
