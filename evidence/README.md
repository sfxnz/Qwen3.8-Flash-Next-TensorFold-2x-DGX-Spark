# evidence/

Receipts for every number in `README.md`. `recipe.yaml` points each measured row at a file here. `python3 kit/render.py --check` lists the rows that have none.

What belongs here:

- `bench-<UTC stamp>.txt`: stdout of `python3 bench_decode.py`, including the `SUMMARY` JSON. One file per frozen wave.
- `run-<UTC stamp>.log`, `doctor-<UTC stamp>.txt`: boot log and API state of the serve the bench ran against.
- `needle-*.txt`, `count-*.txt`, `probe-*.json`: agent-readiness probe outputs.
- `trail.tsv`, `decision.tsv`: what was tried, kept, reverted, and why.

Commit the raw output. Do not type numbers into a file.
