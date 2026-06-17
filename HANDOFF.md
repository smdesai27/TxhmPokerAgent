# HANDOFF — AlphaHoldEm audit, fixes, and rigor work

> Single source of truth for a fresh agent. Read this top-to-bottom, then you can do the
> Phase 2 writeup or continue the work without re-deriving anything. Companion doc:
> [AUDIT_AND_PRESENTATION_PLAN.md](AUDIT_AND_PRESENTATION_PLAN.md) holds the full original
> audit (≈50 findings with file:line) and the presentation strategy/research. This file is
> the *current state + what changed + how to continue*.
>
> Owner context (from the user's PKM): Brown PLME undergrad; North Star = AI × medicine ×
> mechanistic understanding; near-term target = Neuralink MLE intern (be ready ~2026-07-15).
> This poker project is OFF that core neuro track — position it as transferable
> real-time-ML / first-principles / evaluation-rigor evidence, NOT "a poker bot." Deliverable
> goal: a research-style technical writeup + polished live demo.

---

## 0. TL;DR of the whole engagement

1. **Audited** the repo (local + OSCAR training state + AlphaHoldem paper fidelity) with a
   15-agent workflow. Core finding: the project is strong ML-systems work, but its framing was
   a credibility liability — see §3.
2. **Phase 0 (committed + verified):** fixed a fatal Trainer crash, hardened/■fixed the serving
   demo, rewrote the README honestly, added a LICENSE, and de-bloated the git repo.
3. **Phase 1 (committed + verified):** made the evaluation statistically valid (Student-t CIs),
   added a duplicate-hand evaluator, and built a **Leduc exploitability validation** that is the
   opponent-independent rigor artifact, plus an **ablation** of the paper's Trinal-Clip + K-Best
   self-play contributions.
4. **Ops:** fixed an over-quota OSCAR home dir; pushed the branch to GitHub; synced OSCAR so
   training is runnable again.

Everything below is the detail.

---

## 1. Where everything lives (state snapshot)

- **Git branch:** `claude/sharp-swartz-266392` (pushed to GitHub `smdesai27/anitgravity-txhm`).
  Branched from `main` @ `aa45059`. **Not merged to main; no PR opened** (open one at the URL
  GitHub printed if desired).
- **Commits on this branch (newest → oldest):**
  | SHA | What |
  |---|---|
  | `38239f5` | Phase 1: Trinal-Clip + Elo-kBSP ablation flags in the Leduc harness |
  | `1710fc3` | Phase 1: averaged-policy exploitability tracking + curve artifact |
  | `63ef0ba` | Phase 1: Leduc exploitability validation script |
  | `e8670c0` | Phase 1: Student-t eval CIs + duplicate-hand evaluator |
  | `f3a0f98` | Phase 0: Trainer-crash fix, serving hardening, honest README, LICENSE |
  | `2477fd1` | Phase 0: stop tracking checkpoints/logs/artifacts in git (447→112 files) |
- **OSCAR** (`smdesai@sshcampus.ccv.brown.edu`, repo `/users/smdesai/antigravity_poker/anitgravity-txhm`):
  on branch `autoresearch-run1` with the **fixed source applied to the working tree** (via
  `git checkout origin/claude/... -- poker_rl_agent tests configs scripts pyproject.toml`) — so
  the Trainer is runnable. Those are **uncommitted working-tree changes** on OSCAR; commit or
  fully switch to the branch as you like. All checkpoints intact (a plain branch *switch* would
  delete the untracked-on-this-branch snapshot zoo — don't do that without `git stash`/backup).
- **Champion checkpoint (the defensible model):**
  `checkpoints/snapshots/interview_ready/interview_ready_1.pt` — Stage D FCHPA (5-action) @ 21k
  steps. `checkpoints/latest.pt` is the WORSE, gate-failing 49k model — do not present it.
- **Live demo:** Vercel frontend + Render backend (FastAPI). Render serves the champion after
  the Phase 0 changes (`render.yaml` → `CHECKPOINT_PATH=checkpoints/serving_champion.pt`, copied
  from the champion in the Dockerfile). The live demo URL still needs filling into the README.

### OSCAR working notes (gotchas)
- **No Python on login nodes.** Run anything Python via SLURM: `srun --account=default
  --partition=batch --time=... --cpus-per-task=2 --mem=8G python ...`. `batch` has idle nodes;
  `srun` allocates fast. The venv `…/venv` has torch + open_spiel + pytest + numpy; it does NOT
  have fastapi/starlette (serving deps) — install those to a scratch `--target` dir if needed.
- **Home quota:** WAS over (115/100 GB, GRACE_EXPIRED, blocking writes). FIXED 2026-06-17 by
  moving `~/.cache` (25 GB) + `~/.ollama` (1.9 GB) to `/oscar/scratch/smdesai/home_offload/`
  with symlinks. Home is writable again. `checkquota` may still show a cached 115 GB (GPFS
  accounting lag) — the write test is the ground truth. `/oscar/scratch/smdesai` has ~500 GB free
  and is the place to do all verification work.
- **zsh gotcha (local):** the harness shell is zsh, which does NOT word-split unquoted vars.
  `SSHQ="ssh -o ..."; $SSHQ host cmd` fails ("command not found"). Use literal `ssh -o ... host cmd`
  or `${=SSHQ}`. (Cost two failed runs before I caught it.)
- **Verification pattern that works:** `rsync` the source to a fresh `/oscar/scratch/smdesai/<dir>`,
  `scp` a small runner `.sh`, then `ssh host "bash <dir>/runner.sh"` where the runner `srun`s the
  Python. Avoids nested-heredoc quoting.

---

## 2. What actually trained (the true results)

| Stage | Action space | Steps | Result | Status |
|---|---|---|---|---|
| Stage C | FCPA (4) | 20k | +326 bb/100 vs MCCFR-ES, CI-lo 302; all gates pass | certified |
| **Stage D** | **FCHPA (5)** | **21k** | **+897.7 bb/100 (CI-lo 815); robust 934/879; holdout 873/795; ALL gates pass** | **CHAMPION** |
| Stage D | FCHPA | 34k | +804 (CI-lo 748, below champion floor) | NOT promoted |
| "final_opt" | FCHPA | 49k | beats solver tiers but fold 62.6%, raise 3.5%, entropy collapsed | GATE FAIL = `latest.pt` |

The champion behavior is balanced (entropy 1.44 bits, fold 46%, raise 11%). Stage E (full game)
never certified. The whole training campaign + eval lives on OSCAR; result JSONs are in
`logs/` and `artifacts/runs/` there (untracked locally now). **The champion's raw per-seed cert
JSON is gone from disk** (`logs/stage_d/eval/eval_selected_21k_auto_cert_*.json`); its numbers
survive only in `checkpoints/snapshots/interview_ready/interview_ready_1.json` and run manifests.

---

## 3. Fidelity to the AlphaHoldem paper (reproduced vs deviated) — IMPORTANT for honest framing

| Axis | Paper | This repo |
|---|---|---|
| Card/action input | 6×4×13 + 24×4×9 binary tensors → ConvNets | 53-token embedding → MLP + LSTM over betting history |
| Loss | **Trinal-Clip PPO** (ratio clip + δ₁ neg-adv clip + dynamic chips value-target clip) | vanilla clipped PPO + Huber value + homegrown "style" regularizer |
| Self-play | top-K **by Elo** | top-K by raw rollout score; "PFSP" = rating-softmax (mislabeled) |
| Game scope | abstraction-free full no-limit | FCPA(4)/FCHPA(5) discrete betting abstraction |
| Inference | search-free single forward pass | search-free single forward pass ✓ |
| Params | ~8.6M (conv+FC) | ~1.5M (LSTM-heavy) |

**Honest framing:** "an AlphaHoldem-**inspired** end-to-end self-play RL agent" — NOT "a
reproduction." Reproduced: the two-tower pseudo-siamese actor-critic silhouette + search-free
inference + a self-play league. Deviated: input rep, loss, abstraction, scale. Extended: the
multi-stage curriculum + the behavior-gate evaluation harness.

### The eval-credibility problem (the single most important discovery)
The headline "+897 bb/100 vs MCCFR-ES" is **NOT a strength-vs-solver claim.** OpenSpiel's MCCFR
average policy plays **uniform-random on unvisited infosets**, and full-deck HUNL has too many
infosets, so the "solver" is near-random on most lines. Proof: the harness's own **zero-iteration
(uniform) control** beats the agent LESS than the trained solver tiers do — `control_delta` is
NEGATIVE for every tier (champion: default −72.8, robust −36.2). So adding MCCFR iterations does
NOT make the opponent harder. +897 bb/100 = ~+9 BB/hand ≈ **80× AlphaHoldem's +11 mbb/h vs Slumbot**
— a number that only happens vs a near-random opponent. **Never juxtapose this with the paper's
numbers.** Win-rate vs this baseline can be inflated by over-folding (that's how the gate-failing
49k model still "won").

### Claims ledger (use in the writeup)
- ✅ CAN say: built an end-to-end self-play RL system (PPO+GAE, self-play league, multi-stage
  curriculum, served demo); an automated behavior-gate eval harness that **caught policy collapse**
  and **refused to promote a regressing run**; champion learned a balanced non-degenerate strategy;
  engineering judgment (LayerNorm-not-BatchNorm for PPO stability, post-forward masking, near-zero
  head init, STRICT_ABSTRACTION fail-fast); validated the learning method converges on Leduc (§5).
- ❌ CANNOT say: "reproduced AlphaHoldem" (→ "inspired by"); "beat a solver by ~900 bb/100"
  (→ "vs an under-trained MCCFR / scripted baselines"); "Trinal-Clip PPO" (not in the main trainer);
  present `latest.pt` numbers (gate-failing); cite league Elo as absolute strength.

---

## 4. What changed (commit-by-commit, with rationale + verification)

### Phase 0 — `2477fd1` + `f3a0f98`  (verified: 60/60 tests pass on OSCAR; live serving smoke PASS)
- **Trainer crash (fatal):** `Trainer.__init__`/`_ppo_update` called `_load_style_target_profile`
  / `_compute_style_regularizer` / `_style_reg_coef` which had been commented out → `AttributeError`
  on construction, so `scripts/debug_training.py` crashed and ~13 tests failed. **Fix:** restored
  the three methods (they no-op when `STYLE_REG_ENABLE` is false — every shipped config — so
  behavior is unchanged). `poker_rl_agent/training/trainer.py`.
- **Serving demo 404s:** `GET /` served `index.html` but assets mounted only at `/static/` while
  the HTML uses root paths → backend-served page was unstyled/broken (only Vercel worked).
  **Fix:** mount `public/` at `/` with `html=True` after the API routes. `serving/app.py`.
- **Served the wrong model:** deployed `latest.pt` (gate-failing). **Fix:** `Dockerfile` copies the
  champion to `checkpoints/serving_champion.pt`; `render.yaml` points `CHECKPOINT_PATH` there +
  adds `CHECKPOINT_LABEL`; `/api/v1/health` now returns `checkpoint_label`, `num_actions`,
  `game_mode` (`serving/app.py`, `serving/schemas.py`).
- **Public API hardening:** added a dependency-free per-IP `RateLimitMiddleware` (90 req/60s
  global + 15 session-creates/300s, honors `X-Forwarded-For`); tightened session TTL 3600→1200s,
  cap 100→64. `serving/app.py`, `serving/game_manager.py`.
- **Honest README rewrite** (reframe "reproduction"→"inspired by", drop the misleading headline,
  add a Differences-from-the-paper table + honest results/limitations), **MIT LICENSE**, pyproject
  metadata. Removed brittle SLURM-script-presence tests (asserted archived files). `tests/test_stability.py`.
- **Repo hygiene:** `git rm --cached` ~179 MB of checkpoint snapshots + `logs/` + `artifacts/` +
  egg-info (kept the champion + deploy weight). 447→112 tracked files. `.gitignore` updated.
  No history rewrite (blobs still in history; a `git filter-repo`/BFG pass is the optional next step).

### Phase 1 — `e8670c0`, `63ef0ba`, `1710fc3`, `38239f5`
- **Student-t CIs** (`e8670c0`): `evaluate.py` / `evaluate_complete.py` `_aggregate` used a
  Normal 1.96 over n=3–5 seed means → CIs ~40% too narrow. Added
  `poker_rl_agent/evaluation/stats.py` (Student-t critical values, df=n−1, Normal fallback for
  df>30) and wired it in. **Verified locally:** the champion's `default_cfr` CI half-width was
  ±75 (Normal) but is ±106 (correct Student-t). This is a writeup-ready finding.
- **Duplicate-hand eval** (`e8670c0`): `Evaluator.evaluate_duplicate()` + `_play_one_hand()` —
  common-RNG seat-swap so the agent plays both holdings of the same deal; paired Student-t CI.
  **Caveat (verified on OSCAR):** variance reduction is WEAK vs a *stochastic* opponent (0.93×
  vs RandomAgent) because pairing cancels hole-card luck but not the opponent's randomness / board
  runout. It helps vs *deterministic* opponents; full reduction needs a fixed deck. Documented in
  the docstring. Config flags `EVAL_ENABLE_DUPLICATE`, `EVAL_DUPLICATE_PAIRS` added.
- **Leduc exploitability validation** (`63ef0ba`, `1710fc3`, `38239f5`): see §5 — the main rigor
  artifact. `poker_rl_agent/scripts/validate_leduc_exploitability.py`. Also documented in-code that
  `evaluate_nash_conv` is intractable on full HUNL (exact tree traversal).

---

## 5. Leduc exploitability validation + ablation (THE rigor artifact)

**Why:** NashConv (exploitability) is the opponent-independent strength metric, intractable on
full HUNL but exact on solvable Leduc. We validate that the project's PPO + self-play recipe (same
hyperparameters as the HUNL configs, reusing `RolloutBuffer.compute_advantages` for GAE) moves the
policy toward Nash, and we ablate the paper's two signature contributions.

**Script:** `poker_rl_agent/scripts/validate_leduc_exploitability.py`
- Net: small actor-critic MLP over OpenSpiel's `information_state_tensor` (Leduc differs
  structurally from HUNL, so we can't reuse the universal_poker card/action towers).
- Flags: `--policy_loss {vanilla, trinal_clip}` (trinal = δ₁ dual-clip on negative-advantage
  policy term + a chips-scaled value-target clip), `--selfplay {rolling, kbest_elo}` (Elo top-K
  pool + PFSP-weighted opponent sampling). Reports NashConv of BOTH the current iterate and the
  time-averaged (fictitious-play-style) policy.

**Key result — 500-iter baseline run** (artifact: [docs/leduc_exploitability_500iter.json](docs/leduc_exploitability_500iter.json),
figure rendered in chat):
- **Current iterate:** NashConv 4.76 → **best 1.53** (iter 275) → **cycles back up to 4.88** —
  textbook best-response cycling in imperfect-information self-play.
- **Average policy:** falls and **stays stable** (best 1.54, ~3.1× below init; final 2.68). At
  iter 475 the iterate spikes to 3.65 while the average holds at 2.01.
- **Narrative:** the recipe demonstrably reduces exploitability ~3×; the current iterate cycles
  (expected) while the time-average is the convergent estimate — empirically reproducing exactly
  why AlphaHoldem uses a K-Best self-play league and why NFSP/CFR average the policy. This is the
  honest, instructive centerpiece for the writeup.

**Ablation (4 variants, 400 iters, seed 42 — same everything except the ablated feature).**
Artifacts pulled into the repo: [docs/leduc_ablation/](docs/leduc_ablation/) (4 JSONs).

| Variant | current iterate (best → final) | averaged policy (best → final; ×below init) |
|---|---|---|
| vanilla × rolling | 1.39 → 1.79 (cycles) | 1.50 → 1.52 (**3.17×**) |
| trinal_clip × rolling | **bit-identical to vanilla×rolling** | identical |
| vanilla × kbest_elo | stuck ~3.2–3.5, drops late → 1.53 | 2.49 → 2.49 (1.92×) |
| trinal_clip × kbest_elo | **bit-identical to vanilla×kbest_elo** | identical |

**Two honest findings (report exactly — do NOT dress up as wins):**
1. **Trinal-Clip had ZERO effect here — bit-identical curves to vanilla.** The δ₁ dual-clip only
   activates when the importance ratio exceeds δ₁=3, which never happens in near-on-policy PPO
   (fresh rollouts each iter → ratios ≈ 1) — it's an OFF-POLICY/replay control. The value-target
   clip is inert at Leduc's ~14-chip scale (a HUNL ~20000-chip variance control). So the paper's
   clips simply never fire in this regime. This is an honest demonstration of WHEN they matter
   (AlphaHoldem's off-policy replay + huge pots), not a knock on the idea.
2. **The simplified Elo-kBSP UNDERPERFORMED the rolling pool (honest negative result).** PFSP
   (play the hardest opponents 70% of the time) kept the current iterate stuck ~3.2–3.5 for most
   of training and reached a worse averaged-policy exploitability (2.49 vs rolling's 1.50). It did
   make the late iterate non-cyclic, but converged slower overall. Likely causes: on a small game,
   over-weighting the hardest opponents slows learning; and/or the PFSP rate / pool dynamics need
   tuning. The simple rolling baseline won here.

Net: the headline rigor result stands (the recipe drives exploitability ~3× down; iterate cycles,
average converges). The ablations are honest null/negative results that show understanding of the
regimes where the paper's contributions actually bite — good writeup material, not a strength claim.

**To run/reproduce the ablation (OSCAR):** `rsync` the repo source to
`/oscar/scratch/smdesai/leduc_verify/`, then `bash` a runner that `srun`s the 4 variants (see
`/tmp/leduc_ablation.sh` content, reproduced in §7). Each run is a few minutes on a CPU node.

---

## 6. Phase 2 — the writeup (recommended next step; everything it needs is ready)

Spine: *"Building the agent was the easy half; knowing whether it was good was the hard half —
here are the times my evaluation harness told me my 'better' model was actually worse."*

Outline (full version in AUDIT_AND_PRESENTATION_PLAN.md §6):
1. Hook + 3-bullet TL;DR (what built / one honest number / one honest gap).
2. The problem (HUNL as the canonical imperfect-info benchmark; why AlphaHoldem mattered).
3. What I built (pseudo-siamese towers, PPO+GAE, self-play league, FCPA→FCHPA curriculum) — decisions, not features.
4. **Differences from the paper** (§3 table).
5. **Did it actually work? (the eval harness)** — behavior gates; the weak-baseline discovery (control_delta); Student-t CI correction (±75→±106).
6. **What broke** — entropy collapse (the 49k model), the non-promoted 34k, the cycling self-play.
7. **Validating the method (Leduc)** — the exploitability figure (§5) + the ablation of Trinal-Clip / kBSP.
8. Reproducibility + limitations + request-for-research.

Figures ready: the Leduc current-vs-average exploitability curve (data in `docs/`), the ablation
comparison (from `abl_*.json`), and the differences table. Honest claims ledger in §3.

---

## 7. Open items / next steps
- Ablation done + pulled into `docs/leduc_ablation/`; §5 filled in (figure rendered in chat —
  re-render from those JSONs for the writeup). Honest takeaway: paper's clips inert in this
  regime; kBSP underperformed rolling.
- **Fill the live-demo URL** into the README; verify the Render deploy serves the champion
  (the Phase 0 serving smoke passed in a scratch env: `/health` fields, asset mount, rate-limiter 429s).
- **Optional bigger swings** (gate on the Neuralink timeline): a genuine HUNL strength number via
  the **Slumbot public API** (needs an API client + bet-size action translation, since the agent
  is FCHPA-discrete); a fixed-deck version of the duplicate-hand eval; implementing true
  Trinal-Clip + Elo-kBSP in the *main* HUNL trainer (currently only in the Leduc harness).
- **Optional:** `git filter-repo` to purge the ~179 MB of checkpoints from history (untracking
  was done, but history still carries them); open a PR / merge to main.
- **Do NOT** re-train the HUNL champion to chase a bigger bb/100 vs the weak baseline — meaningless.

### Reproduction commands (key)
```bash
# Tests (Linux/OSCAR compute node):  pytest tests/test_stability.py -q
# Leduc validation (one variant):
python poker_rl_agent/scripts/validate_leduc_exploitability.py \
  --policy_loss trinal_clip --selfplay kbest_elo --iterations 400 --eval_every 25 \
  --output_json logs/leduc_trinal_kbest.json
# Serving (defaults to the champion checkpoint):
python -m poker_rl_agent.serving.app --host 0.0.0.0 --port 8000
```
OSCAR SSH: `ssh -o BatchMode=yes smdesai@sshcampus.ccv.brown.edu`. Project:
`/users/smdesai/antigravity_poker/anitgravity-txhm`. Always run Python via `srun ... --account=default --partition=batch`.
