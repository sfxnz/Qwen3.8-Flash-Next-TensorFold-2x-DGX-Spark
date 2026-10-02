# Session 4: MTP draft depth 8-15 against a fresh baseline (2026-10-02)

**Result.** Depth 15 (the engine cap) is adopted as the default. Against B0c (depth 6) it measured structured c1 +45.6%, c2 aggregate +41.9%, JSON +10.4%, code +4.2%; prose, long prose, multilingual and sampled chat within ±1%. D10r and D12r repeat D10 (s3) and D12 within 1-2%. Memory after load is unchanged (58 / 61 GiB available); graphs captured at boot 21 → 48. Greedy reply shas are identical at every depth.

Driver: `tools/sweep.sh evidence/s4-depth/spec.txt evidence/s4-depth`. Long-context check of the same choice: `evidence/s5-longctx/`.

Generated with `python3 tools/compare.py evidence/s4-depth/B0c evidence/s4-depth/{D8r,D10r,D12,D12r,D15}`:

| metric | B0c | D8r | D10r | D12 | D12r | D15 |
|---|---:|---:|---:|---:|---:|---:|
| prose c1 | 69.9 | 69.5 (-0.5%) | 69.2 (-0.9%) | 69.3 (-0.9%) | 70.7 (+1.2%) | 70.4 (+0.8%) |
| prose c1 ttft | 0.0565 | 0.0564 (-0.2%) | 0.0584 (+3.4%) | 0.0619 (+9.5%) | 0.0657 (+16.3%) | 0.0568 (+0.5%) |
| prose c2 | 70 | 70 (-0.0%) | 69.4 (-0.9%) | 69.6 (-0.7%) | 69.9 (-0.2%) | 70.3 (+0.4%) |
| prose c2 agg | 92.1 | 92 (-0.1%) | 91.4 (-0.7%) | 91.6 (-0.5%) | 92.1 (+0.0%) | 92.4 (+0.4%) |
| prose c2 ttft | 0.791 | 0.791 (-0.0%) | 0.793 (+0.2%) | 0.789 (-0.3%) | 0.789 (-0.2%) | 0.785 (-0.8%) |
| structured c1 | 172 | 198 (+15.6%) | 216 (+26.0%) | 234 (+36.1%) | 234 (+36.1%) | 250 (+45.6%) |
| structured c1 ttft | 0.0562 | 0.0627 (+11.5%) | 0.0623 (+10.9%) | 0.0572 (+1.7%) | 0.0604 (+7.5%) | 0.0556 (-1.0%) |
| structured c2 | 173 | 196 (+13.3%) | 217 (+25.6%) | 235 (+35.6%) | 234 (+35.3%) | 247 (+42.5%) |
| structured c2 agg | 227 | 257 (+13.4%) | 282 (+24.3%) | 305 (+34.6%) | 306 (+35.1%) | 321 (+41.9%) |
| structured c2 ttft | 0.663 | 0.589 (-11.2%) | 0.534 (-19.4%) | 0.51 (-23.1%) | 0.51 (-23.0%) | 0.487 (-26.6%) |
| code | 124 | 132 (+6.0%) = | 128 (+3.1%) = | 128 (+3.1%) = | 130 (+4.2%) = | 130 (+4.2%) = |
| prose_long | 81.9 | 81.8 (-0.1%) = | 81.6 (-0.4%) = | 80.9 (-1.3%) = | 81.2 (-0.8%) = | 81.2 (-0.9%) = |
| json | 141 | 150 (+6.6%) = | 158 (+12.1%) = | 157 (+11.5%) = | 156 (+10.6%) = | 156 (+10.4%) = |
| multilingual | 68.2 | 68.4 (+0.2%) = | 69.1 (+1.2%) = | 68 (-0.3%) = | 68.3 (+0.0%) = | 68.2 (-0.0%) = |
| chat_sampled | 71.5 | 71.8 (+0.4%) = | 72.4 (+1.2%) = | 71.6 (+0.2%) = | 71.8 (+0.4%) = | 72.2 (+0.9%) = |
| frozen tokens/round | 3.56 | 3.83 (+7.7%) | 4.04 (+13.5%) | 4.21 (+18.3%) | 4.21 (+18.3%) | 4.33 (+21.7%) |
