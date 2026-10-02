# Session 2: baseline B0 through run.sh and the gate pack (2026-10-02)

**Result.** `run.sh` boots both ranks end to end (worker image copy, snapshot header check on both nodes, rank 1 then rank 0, worker watch) and `tools/session_gate.sh` passes. B0 (TensorFold defaults: depth 6, confidence 0.70, bf16 KV, both HCAs) is the reference for session 3: prose c1 70.4, structured c1 175.9 tok/s, 3.56 tokens per round. Every greedy cell of `tools/bench_cells.py` repeated byte for byte across its 3 runs.
