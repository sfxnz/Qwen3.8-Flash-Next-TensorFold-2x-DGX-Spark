# Session 1: first TP=2 boot of TensorFold on the pair (2026-10-02)

**Result.** TensorFold 0.6.2 serves `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` at `--tp 2` across spark1/spark2. Loaded in 133 s including the first CUDA kernel JIT; full 262,144 window on both ranks; n-gram tables mlocked; 21 decode graphs. Smokes thinking/tools/count pass. Frozen `bench_decode.py`: prose c1 69.4, structured c1 171.6 tok/s (TensorFold default depth 6).

- Booted by hand (before `run.sh` existed): rank 1 on spark2 first, then rank 0. The exact argv is in `docker-head.log` / `docker-spark2.log`.
- The displaced vLLM serve (`qwen38-flash-next-nvfp4`) was stopped gracefully first (`docker stop -t 120`, head then worker, both exit 0) and restored at the end of the work.
- MTP acceptance over the frozen run (from `/health` deltas): 3.56 tokens per round, draft hit rate 0.853.
