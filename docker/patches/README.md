# docker/patches/

Patches the image build applies on top of TensorFold at `TF_SHA` (`docker build --build-arg TF_PATCH=patches/<file> --build-arg TF_PATCH_SHA=<sha256>`). Each shipped patch's sha256 is pinned in `run.sh` (`PATCH_PINS`) and `recipe.yaml` (`engine.patches`), and `tests/` checks the files against the pin. `run.sh` refuses a patch file that differs from its pin. The build checks the copied file against the sha it is given, and `run.sh` refuses an image whose `tensorfold.patch_sha` label differs. A regenerated patch therefore needs a new pin, and new evidence for the numbers it affects.

## pr141-on-0.6.2.patch

`PROFILE=concurrent` uses it. Two-rank `--parallel N` for Flash Next on CUDA. TensorFold 0.6.2 refuses that combination (`families/qwen4_exp/cuda/engine.py:64-66`).

- **Source.** Upstream [TensorFold PR #141](https://github.com/ashhart/TensorFold/pull/141) "feat(flash next cuda): --parallel N on two ranks". Its unique commits are `3854dc4` (Bill H., BHCC2025) and `cb5101d` (ashhart), on branch `pr-141-0.6.1` (base `e5f50dd`, a pre-release snapshot of 0.6.1). The PR was open and in no release on 2026-10-02.
- **Base.** TensorFold v0.6.2, `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`.
- **License.** Apache-2.0, as TensorFold (its `LICENSE` and `NOTICE` apply to the patched files).
- **The file.** `git diff v0.6.2 HEAD` of a local port branch with four commits:
  1. `4d534b2` port: the two PR commits squashed and applied with `git apply --3way`. Three conflicted files (`cuda/geometry.py`, `qwen4_exp/cuda/engine.py`, `qwen4_exp/cuda/multi.py`) were hand-merged. 0.6.1/0.6.2 prompt-pass fixes were kept in `multi_fill.PromptPasses`. Values a rank reads from its own timing or arrivals are fixed under two ranks. 0.6.2's prefix forks are planned on rank 0 and replayed by rank 1 (`Shadow.copy_prefix`).
  2. `4b7b3e7` review fixes:
     - rank 1 keeps following after any admission error rank 0 survives (before, rank 0 hung in NCCL);
     - a slot change that fails on one rank fails both before any model collective;
     - streams a client left are named in the "drop" message, so both ranks' stream lists stay equal.
  3. `5a49305` the lone-stream graph slot keeps its cache geometry across requests, so its CUDA graphs are not recaptured each time.
  4. `dd1bda6` a busy lone-graph slot gives its spare rows back under memory pressure.
- **One-GPU behaviour.** 0.6.2's, with one deliberate fix: a stream that waits for memory flushes its deferred DeltaNet rows before the shared scratch is reused. In 0.6.2 such a stream resumes with another stream's rows. This recipe runs two ranks, so this does not reach it.
- **Validation.** On GB10: [`evidence/s6-pr141/`](../../evidence/s6-pr141/).
  - TensorFold's CUDA suites: 160 passed.
  - Two-rank serve: every concurrent reply equals its solo and serial run at 1-8 users.
  - Failure probes: a client leaving mid-stream, a grammar request getting HTTP 400, serving continuing afterwards.
  - Host tests (`tests/test_flashnext_round_plan.py` in the patch) cover planning and replay.

To change the patch: edit the port branch, re-run TensorFold's host tests and `evidence/s6-pr141/gpu_tests.sh`, regenerate with `git diff v0.6.2 HEAD > pr141-on-0.6.2.patch`, rebuild the image, and re-run `tools/probe_parallel.py` and `tools/vendor/bench_concurrent.py --alone --serial` on two ranks. Never edit the patch file by hand.
