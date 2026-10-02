# AGENTS.md

Standing orders for any agent working in this recipe repo. Read before acting.

1. **PR only.** Commit on `agent/**` branches. Open a PR against `main`. Never push `main`. Never merge. Never force-push.
2. **Evidence.** Every number in `README.md` has a file under `evidence/` that proves it. No file, no number. Never gitignore `evidence/`. `recipe.yaml` names the file per measured row; `python3 kit/render.py --check` lists the rows without one.
3. **GPU lease.** The Sparks may be serving. Do not run `./run.sh` or start a vLLM container without a lease on the lane (`gpu/lease.sh acquire pair` in forge). One serve per lane. Never start a second instance. Never stop a container this run did not start.
4. **Validate first.** `VALIDATE_ONLY=1 ./run.sh` before any serve. It needs no lease and no Docker.
5. **Source of truth.** Values live in `recipe.yaml`. Edit it and run `python3 kit/render.py`. Never edit inside the generated markers by hand.
6. **Voice.** Short sentences. Numbers first. No marketing language. No emojis.
7. **Secrets.** Never commit tokens, Tailscale IPs, or auth headers.
