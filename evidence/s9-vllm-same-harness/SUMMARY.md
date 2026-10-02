# Session 9: the vLLM sibling restored, then measured with this recipe's harness (2026-10-02)

**Result.** The displaced vLLM serve was restored with its own `run.sh` (main `d66a95e`, see `vllm-recipe-head.txt`). Its container image, env, argv and mount points on both nodes are identical to the inspect snapshot taken before the TensorFold window. Its smokes pass (`restore-smoke_*.out`: thinking, tools, vision, count). Its frozen ruler measured prose c1 40.8 then 46.5 tok/s (`restore-bench-frozen*.out`; its README states 56.9 from its session 8, with the same prose acceptance, 2.69) and structured c1 82.8 / 82.4 (README 84.3).

Then `tools/bench_cells.py`, `tools/bench_distinct.py`, `tools/bench_prefill.py --lengths 8192 32768 131072 --run 100` and `tools/vendor/bench_concurrent.py --levels 1,2,4,8 --reps 3` ran against it, with the model name `nvidia/Qwen3.8-Flash-Next-NVFP4` (`harness.sha256`, `serve-argv.txt`). `bench_concurrent.py` ran without `--alone --serial`: those checks compare TensorFold's per-reply token sha, which vLLM does not return.

The vLLM serve was left running afterwards, as found.

**Prefill re-run.** The first `bench_prefill.py` run against vLLM (`bench-prefill-nowarmup.out`) had no warm-up, and its 8k row (1,758 tok/s) was vLLM's first long prompt after boot. The pre-PR review flagged it. The tool now sends a throwaway long prompt first (`harness-prefill-warmup.sha256`). Re-run with a fresh filler (`--run 101`, because vLLM's prefix cache held `--run 100`): `bench-prefill.out` gives 2,977 / 2,754 / 2,033 tok/s cold at 8k / 33k / 131k. The README uses the re-run.
