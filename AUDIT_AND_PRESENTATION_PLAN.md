# AlphaHoldEm — Audit & Presentation Plan

> Prepared 2026-06-16. Audience: Sanil. Goal: turn this project into a high-signal research-style
> technical writeup + polished live demo that holds up to ML engineers, researchers, and recruiters.
> Method: 15-agent adversarial audit of the local repo + OSCAR training state + the AlphaHoldem paper,
> plus external research on how to present technical projects. Every result number below is sourced.

---

## 0. Executive verdict

**This project is stronger than its current framing — but the current framing is a credibility liability.**

The repo is marketed as an *"AlphaHoldEm reproduction"* whose headline result is *"+897 bb/100 vs an
MCCFR-ES solver."* Both claims would be dismantled by any expert reader in minutes:

1. It is **not an architectural reproduction** of AlphaHoldem (LSTM action tower, not the paper's
   24-channel ConvNet; vanilla PPO + a homegrown "style" regularizer, not the paper's signature
   Trinal-Clip loss; FCPA/FCHPA betting abstraction, not the paper's abstraction-free full game;
   ~1.5M params, not the paper's 8.6M).
2. The **headline win-rate is not a strength signal.** The "MCCFR-ES solver" baseline is effectively
   **uniform-random on most game states**, proven by the harness's *own* zero-iteration control: the
   champion scores **higher** vs the 0-iteration uniform baseline (970.5 bb/100) than vs the 5k-iter
   "solver" (897.7) and 10k-iter "solver" (934.3). Adding solver iterations does **not** make the test
   harder — so "+897 bb/100" means "beats a near-random opponent by ~9 BB/hand," which is **~80× the
   margin AlphaHoldem reports vs Slumbot (+11 mbb/h)**. Juxtaposing those numbers is a landmine.

**The good news: the *true* story is exactly what your target audiences value most.** What you actually
built is a complete, from-scratch deep-RL **system** — PPO + GAE + a K-best-style self-play league +
a multi-stage curriculum + checkpointing + an **unusually rigorous automated evaluation harness with
behavior gates** + a **deployed interactive demo**. And — this is the differentiator — your process
**honestly recorded its own failures**: the 34k run was correctly *not promoted* (missed the CI floor),
the 49k "final optimization" *failed its behavior gates* (entropy collapse → fold-heavy), and the
auto-tuning experiments crashed and that's logged. That disciplined "freeze the champion, don't promote
regressions, catch policy collapse automatically" methodology is a genuine **research-maturity signal**
— and it maps one-to-one onto your physician-scientist North Star (rigorous evaluation under
uncertainty, honest reporting of negative results, decisions under imperfect information).

**The move: stop selling "I reproduced AlphaHoldem and beat a solver." Sell "I built a complete deep-RL
self-play system *and the evaluation discipline to know when it was actually working* — including when
it wasn't."** Make the eval harness and the honest negative results the *centerpiece*, not something to
hide. External research is unanimous that a "what didn't work" section + evaluation rigor are the
highest-leverage, most-skipped signals in a portfolio — and you already have both, for real.

---

## 1. What actually exists today (the true state)

### Training journey (sourced)

| Stage | Action space | Steps | Result | Status |
|---|---|---|---|---|
| Stage A | (early) | ~3k | +640 bb/100 vs MCCFR-ES, CI ±444 (2k eps/seed) | inconclusive (huge CI) |
| Stage B | FCPA-era | 5k/10k | +617 / +599 bb/100, CI-lower ~490/502 | certified pass |
| **Stage C** | **FCPA (4-action)** | **20k** | **+326 bb/100, CI-lower 302; pass_all=true** | **certified, clean** |
| **Stage D** | **FCHPA (5-action)** | **21k** | **+897.7 bb/100, CI-lower 815; robust 934/879; holdout 873/795; ALL gates pass** | **CHAMPION (`interview_ready_1.pt`)** |
| Stage D | FCHPA | 34k | +804 bb/100, CI-lower 748 (−67 below champion) | **NOT promoted** (CI floor) |
| Stage E-ish "final_optimization" | FCHPA | 49k | beats solver tiers, but **fold 62.6%, raise 3.5%, entropy collapsed** | **GATE FAIL** (this is `latest.pt`) |
| autoresearch (Mar 15) | FCHPA | 34k→34.1k | 100-iter nudges on an already-degraded base; one crashed | not real runs |

**The defensible champion is the Stage D FCHPA 21k checkpoint** (`checkpoints/snapshots/interview_ready/interview_ready_1.pt`),
which passes every gate with balanced behavior (entropy 1.443 bits, fold 45.9%, raise_total 10.7%).
Stage E (full-game progressive curriculum) was attempted but **never produced a certified checkpoint** —
the project effectively plateaued at the Stage D 21k champion.

### Code state (important and currently broken)

- **The Trainer is dead-on-arrival.** `Trainer.__init__` calls `self._load_style_target_profile()`
  ([trainer.py:113](poker_rl_agent/training/trainer.py:113)) and `_ppo_update` calls
  `self._compute_style_regularizer()` ([trainer.py:781](poker_rl_agent/training/trainer.py:781)) — both
  definitions are commented out. Constructing `Trainer(config)` raises `AttributeError` immediately. So
  **`python scripts/debug_training.py` crashes on the documented command**, and ~13 tests fail. The
  OSCAR branch HEAD is literally a *revert* of the fix for this crash. (The deployed demo/eval don't
  touch `Trainer`, so the live site still works.)
- **Tests are substantially red:** ~13 `Trainer`-constructing tests + 14 tests asserting the existence
  of SLURM scripts that have since been deleted (only 2 of 16 remain).
- **~179 MB of `.pt` checkpoints + 335 non-code files** (logs, artifacts) are committed to git.
- **No LICENSE file.**

---

## 2. Fidelity to the paper — reproduced vs deviated

This table *is* an asset: shipping it as a "Differences from the paper" section converts the biggest
credibility risk into a display of understanding.

| Axis | AlphaHoldem (paper) | This repo | Verdict |
|---|---|---|---|
| Card input | 6×4×13 multi-channel binary tensor → ConvNet | 53-token embedding → 3-layer MLP | **Deviation** |
| Action input | 24×4×9 tensor (history + legal mask in-tensor) → ConvNet | integer sequence → 2-layer **LSTM** | **Deviation** |
| Towers | Two **untied ConvNets**, late FC fusion | Untied MLP + LSTM, late fusion | Silhouette ✓, components ✗ |
| Heads | Policy + value | Policy + value | ✓ |
| Loss | **Trinal-Clip PPO** (ratio clip + δ₁=3 neg-adv clip + dynamic value-target clip to chips) | Vanilla ratio clip + Huber value + **homegrown style regularizer** + advantage clip | **Deviation (the signature contribution is absent)** |
| Self-play | **K-best by ELO** | Top-K by **raw rollout score**; "PFSP" = rating-softmax (not win-prob weighting) | Partial / mislabeled |
| Inference | **Abstraction-free, search-free**, full game, 1 forward pass | **FCPA/FCHPA abstraction**, 1 forward pass | Search-free ✓, abstraction-free ✗ |
| Params | 8.6M (1.8M conv / 6.8M FC) | ~1.5M (LSTM-heavy) | ~5.7× smaller |
| γ | 0.999 | 0.995 | minor deviation |
| Extra | (two input streams only) | + 11-dim scalar feature stream | addition |

**Honest one-liner:** *"an AlphaHoldem-**inspired** end-to-end self-play RL agent that keeps the
two-tower pseudo-siamese actor-critic silhouette and the search-free single-forward-pass inference, but
deliberately substitutes an embedding+LSTM front-end, vanilla+regularized PPO, and a discrete betting
abstraction — trading architectural fidelity for something trainable on a single GPU."*

---

## 3. The credibility ledger (read this before writing one sentence publicly)

### ✅ You CAN truthfully claim
- "Built a complete end-to-end deep-RL training system for Heads-Up No-Limit Hold'em from scratch in
  PyTorch: PPO + GAE, a K-best self-play league, a multi-stage FCPA→FCHPA curriculum, v2 checkpointing,
  and a deployed play-against-the-agent web demo."
- "Designed an automated evaluation harness with multi-seed CIs, holdout-seed independence checks, and
  behavior gates (fold frequency, aggression, entropy, pot-mix) that **automatically caught a policy
  collapse** (an entropy-collapsed, fold-heavy 49k checkpoint) and **correctly refused to promote a
  longer 34k run** that regressed below the champion's confidence floor."
- "The Stage D FCHPA 21k agent learned a non-degenerate, balanced HUNL strategy (entropy 1.44 bits,
  fold 46%, raise 11%) and beats scripted baselines and an under-trained MCCFR opponent."
- "Engineering judgment under PPO instability: LayerNorm-over-BatchNorm for small-batch stability,
  near-zero head init, dtype-safe post-forward action masking, correct GAE episode-boundary handling,
  STRICT_ABSTRACTION fail-fast integrity checks."
- "Deployed it" — only ~23% of ML practitioners have ever shipped a model to production.

### ❌ You CANNOT claim (these get you caught)
- ❌ "Reproduced AlphaHoldem." (Architecture, loss, and game scope all differ.) → say **"inspired by."**
- ❌ "Beat a solver by ~900 bb/100." The MCCFR baseline is near-uniform-random on most states (proven
  by the negative control_delta). → say **"vs an under-trained MCCFR / scripted baselines,"** and
  *never* put this number next to AlphaHoldem's +11 mbb/h.
- ❌ "Trinal-Clip PPO." Not implemented.
- ❌ Present `checkpoints/latest.pt` numbers as the result — that's the **gate-failing, collapsed**
  49k model. The champion is `interview_ready_1.pt` (21k).
- ❌ Cite ELO/league rating as an absolute strength metric — it's an internal PFSP rating pinned near
  1200, not measured against any external opponent.

### 🎯 The most impressive TRUE framing (the spine of the writeup)
> *"I set out to reproduce AlphaHoldem and instead learned a harder lesson: in imperfect-information
> games, **building the agent is the easy half — knowing whether it's actually good is the hard half.**
> This is the story of an end-to-end self-play RL system, the evaluation harness I had to build to trust
> it, and the three times that harness told me my 'better' model was actually worse."*

That framing is honest, technically deep, memorable, and is *exactly* the research-taste signal that
ML labs and a physician-scientist path reward.

---

## 4. Audit findings by severity

### 🔴 Blockers — must fix before anything goes public (they make you look broken or dishonest)
1. **Trainer crashes on instantiation** ([trainer.py:113](poker_rl_agent/training/trainer.py:113),
   [:781](poker_rl_agent/training/trainer.py:781)). Restore no-op stubs for the style-reg methods (or
   delete the call sites), then add a one-line `Trainer(Config())` smoke test. *(S)*
2. **Backend-served frontend 404s its assets** — `GET /` serves `index.html` but assets mount at
   `/static/` while the HTML uses root paths; the Render URL is unstyled/broken. Mount `public/` at `/`
   with `html=True` or fix hrefs. *(S)*
3. **Served checkpoint confusion** — `latest.pt` got overwritten and is a non-best snapshot; deploy and
   document the champion (`interview_ready_1.pt`) and surface its name + eval in `/health`. *(S)*
4. **Honest README rewrite** — reframe "reproduction" → "inspired by," delete the overclaimed win-rate
   numbers, add the §2 differences table. *(M)*
5. **Public API hardening** — unauthenticated, no rate limit, global 100-session cap = trivial DoS on
   the free tier; `/health` leaks `active_session_count`. Add per-IP rate limit + per-IP session cap.
   *(S–M)* ([app.py:189](poker_rl_agent/serving/app.py:189), [game_manager.py:72](poker_rl_agent/serving/game_manager.py:72))
6. **Add a LICENSE** (MIT/Apache-2.0) + license field in pyproject. *(S)*
7. **Repo cleanup** — purge ~179 MB of checkpoints + logs/artifacts from git tracking (keep only the
   one deployed weight + a representative eval report); remove tracked `egg-info` and stale `.env`
   files; add `logs/`, `artifacts/`, `checkpoints/snapshots/**` to `.gitignore`. *(M)*

### 🟠 High — define whether the writeup has a *trustworthy* result
8. **The eval baseline is not a solver.** Either (a) **enable & run NashConv/exploitability on the
   FCPA 4-action abstraction** — the code already exists ([evaluator.py:673](poker_rl_agent/evaluation/evaluator.py:673)),
   just disabled (`EVAL_ENABLE_NASH_CONV=False`) — to get an **opponent-independent** strength number;
   and/or (b) implement **duplicate/mirrored-hand evaluation** (play the same dealt cards from both
   seats) to cut variance an order of magnitude and get trustworthy CIs. **This is the single
   highest-value technical addition for a research writeup.** *(M–L)*
9. **CIs use Normal 1.96 over n=3–5 seeds (should be Student-t) and no variance reduction.** Reported
   intervals are ~40–120% too narrow. Switch to Student-t; add bootstrap CIs over hands. *(S)*
10. **Model selection runs on the weak baseline** — `best_eval` maximizes single-seed win-rate vs the
    near-random MCCFR, which is *how* the entropy-collapsed model scored "high." Select on a composite
    that includes behavior gates / multi-seed. *(M)*
11. **Trinal-Clip / K-best-by-ELO / PFSP are mislabeled or absent.** Either implement them (the value
    head and per-state masks are already exposed) or describe the method accurately. The value-target
    clip alone drove the paper's ELO 1308→1597, so it's the single most impactful fidelity upgrade if
    you want to push toward the paper. *(M–L each)*

### 🟡 Medium — quality/rigor polish (great writeup material)
12. **Self-play is fully serial with batch-1 inference** → GPU is starved. Batched inference across
    parallel envs is a **10–50× rollout speedup** and a clean, *demonstrable* engineering result for
    the writeup. *(M)* The O(L²) per-episode history replay
    ([state_representation.py:34](poker_rl_agent/environment/state_representation.py:34)) is a cheap
    incremental-tracking fix that pairs with it. *(S)*
13. **Dropout-on/off mismatch** between rollout (eval) and PPO update (train) biases the importance
    ratio. *(S)*
14. **γ=0.995 vs paper's 0.999**; **config STACK/BLIND fields are silently ignored** (decorative);
    **reward unit soup** (chips vs BB vs BB/100 mixed) — pick mbb/hand and report mean + CI in one unit.
    *(S each)*
15. **No CI, floor-pinned deps, OpenSpiel install hand-waved off-Linux** — add a GitHub Action that runs
    `pytest`, pin deploy-path deps, document a known-good install (or devcontainer). *(M)*

*(Full per-finding evidence with file:line is in the workflow output; ~50 findings total across 7 audits.)*

---

## 5. What the research says (tailored to you)

**How recruiters/leaders actually read projects** (InterviewNode, Eugene Yan, the "100 portfolios"
analysis, TheLadders eye-tracking):
- You get **~6 s on a resume, <60 s on a first portfolio pass, <2 min inside a repo.** The README's
  first screen (one-line what-it-is + one honest headline number + a demo GIF + one-command run) decides
  everything.
- **Signal density beats effort/LOC.** Toy clones (Titanic/MNIST) trigger instant rejection; an analysis
  of 100 portfolios found 72% of rejected ones were over-engineered toys and 43% undifferentiated clones.
- The thing junior portfolios skip most and that reads as *senior*: **documented decisions/tradeoffs and
  an honest "what didn't work."** Eugene Yan (Amazon principal, runs hiring) probes "how did you evaluate
  / what error analysis / what hard tradeoffs / what broke and how did you fix it" — rewarding judgment
  and intellectual honesty over polish.
- **A deployed demo is the rare unfair advantage** (~23% of practitioners ever ship). You have one.

**What makes a great reproduction writeup** (the 37 Implementation Details of PPO; amid.fish "Lessons
Reproducing a Deep RL Paper"; Karpathy's Recipe; Distill "Research Debt"; rliable):
- Open on a **concrete hook + <100-word TL;DR**; readers consume only ~20–28% of body text.
- **Name exactly what you reproduce and quantify the gap honestly** (the 37-details post pins the exact
  commit and openly flags what it did *not* match). Your §2 table is this.
- **Report rigor, not point estimates:** mbb/hand with bootstrap CIs, multiple seeds, performance vs a
  *named* baseline (cite rliable / "Deep RL that Matters").
- **A first-class failure-modes section** is the credibility core (amid.fish centers failure as the
  story: 8 months not 3, debugging 4× implementation). Your entropy-collapse + non-promoted-34k + the
  weak-baseline realization *are* that section.
- Add **interactive/illustrated elements** (architecture diagram, a training-iteration slider showing
  strategy shift, an interactive hand replay).

**Why a poker project is fine for a Neuralink-track candidate** (the Neuralink MLE posting itself):
the posting says *"no prior knowledge of neuroscience is required; we value simple solutions grounded in
first principles"* and asks for *"designing, building, and shipping real-time ML products."* An
off-domain RL **system** that you built, trained, evaluated, and shipped reads as **transferable
real-time-ML + first-principles** evidence — *if* you pitch it as a "real-time self-play RL training
system + evaluation harness," not as "a poker bot." Reproductions are a near-elite signal precisely
because ~46% of attempts fail to match numbers and "99% of candidates aren't doing this."

**Live demo craft:** reviewers decide in 15–30 s and a broken link is worse than no link. Keep the free
instance warm (5-min health ping / UptimeRobot, or the \$7 Starter), show **action probabilities + a
win-probability meter + a "why" chip** (explainable-AI legibility), guard with per-IP rate limits +
budget caps, and put **"Try the agent" / "Read how it was trained"** CTAs side by side.

---

## 6. The plan (prioritized, with effort + anti-goals)

> Sequencing assumes your Neuralink window is ~5 weeks out (ready ~07-15). This plan is designed so the
> project is **presentable after Phase 0+1 (≈1 week)** and **excellent after Phase 2+3 (≈2–3 weeks)**,
> without colliding with your core neuro portfolio.

### Phase 0 — Stop the bleeding (≈2–3 days). Make it not-broken and not-dishonest.
Fix blockers 1–7 above. After Phase 0 the repo is clonable-and-runnable, the demo works from any link,
the README is honest, and nothing embarrassing is exposed. **Do this even if you do nothing else.**

### Phase 1 — Earn one trustworthy claim (≈3–5 days). This is what makes it *research*.
- Enable + run **NashConv/exploitability on FCPA** for the champion (and ideally Stage C) → an
  opponent-independent strength number (finding #8a).
- Implement **duplicate/mirrored-hand eval** + Student-t/bootstrap CIs (#8b, #9) → trustworthy
  intervals you can defend.
- Regenerate the champion's headline figures from `logs/metrics/*.jsonl` (training curves: entropy,
  fold-freq, value loss, the 39k→49k *collapse* curve — that collapse plot is gold).
- *(Optional, high-ROI for an MLE pitch)* the **batched-inference speedup** (#12) with a measured
  before/after — a concrete "I made training 10–50× faster" result.

### Phase 2 — The writeup (≈4–6 days). The core deliverable.
Recommended home: a personal blog / GitHub Pages, cross-posted, linked from the README top. Long-form
(Lil'Log/Distill style), one idea per section, every claim reproducible to a command. Outline:

1. **Hook + TL;DR** — the "building it is the easy half" framing; 3 bullets: what you built, the one
   honest number (exploitability on FCPA + "beats scripted/under-trained baselines"), the one honest gap.
2. **The problem** — HUNL as the canonical imperfect-information benchmark; why AlphaHoldem mattered
   (end-to-end RL replacing prohibitively expensive CFR, one GPU). *Why it matters* = anti-novelty-chasing.
3. **What I built** — the pseudo-siamese two-tower actor-critic, PPO+GAE, the self-play league, the
   FCPA→FCHPA curriculum; an architecture diagram. Decisions-not-features (LayerNorm vs BatchNorm and
   *why*, post-forward masking, near-zero init).
4. **Differences from the paper** — the §2 table. Honest, precise, deep.
5. **Did it actually work? (the eval harness)** — the centerpiece. The behavior gates; the
   exploitability number; *and the weak-baseline discovery* ("my +897 bb/100 was vs a near-random
   opponent — here's the zero-iteration control that proved it, and what I changed").
6. **What broke** — the entropy collapse (with the 39k→49k plot), the non-promoted 34k run, the
   crashed auto-tuning; what signal exposed each (entropy, fold-freq drift). Cite amid.fish / Irpan.
7. **Results** — champion behavior + exploitability + CIs, contextualized; what's reproduced vs not.
8. **Reproducibility + request-for-research** — one-command repro, seeds, hardware, limitations, and
   the open threads (Stage E full game, true Trinal-Clip, continuous bet sizing).

### Phase 3 — The demo (≈2–3 days). The unfair advantage.
- Deploy the **champion** (not `latest.pt`); show its name + eval in `/health`.
- **Legibility:** action-probability bar chart + win-prob meter + a "why" chip per bot action (you
  already compute the policy distribution server-side — surface it).
- Cold-start: keep-warm ping or \$7 Starter + a "waking up…" client state.
- Guardrails: per-IP rate limit + session cap; explicit error states.
- Link: "Try the agent" ↔ "Read how it was trained" CTAs; a 20-s GIF in the README; landing on a
  finished hand.

### ⛔ Anti-goals (protect your time given Neuralink)
- **Do NOT re-train from scratch / chase the paper's numbers.** The champion is good enough to write
  about; new full runs are weeks of GPU for marginal narrative gain.
- **Do NOT implement full Trinal-Clip / ConvNet towers** unless you decide this is a multi-week flagship
  — list them as "what I'd do next" instead. (Document the *intent* to show you understand them.)
- **Do NOT start a new project.** Three deep, deployed, documented projects beat fifteen notebooks; make
  this the RL-depth anchor next to one on-track neuro project.
- **Do NOT polish the 33 Stage-D ablation presets / SLURM zoo** — archive them; they read as scratch space.

### Resume bullet + one-liner
- **Bullet:** *"Built an end-to-end self-play deep-RL system for Heads-Up No-Limit Hold'em (PyTorch:
  PPO+GAE, K-best self-play league, multi-stage curriculum, served inference) and an automated
  evaluation harness whose behavior-gate + exploitability checks caught policy-collapse regressions that
  raw win-rate hid; deployed a live play-against-the-agent demo."*
- **One-liner pitch:** *"A real-time self-play RL agent for imperfect-information poker — and the
  evaluation discipline to actually trust it."*

---

## 7. Strategic recommendation

Given Neuralink is ~5 weeks out and this project is **off your documented neuro spine**: do **Phase 0+1
now** (≈1 week) so the project is honest, runnable, and has one trustworthy claim — that alone removes
all downside and makes it a usable portfolio entry and interview talking point. Then do **Phase 2 (the
writeup)** as the real value, because the writeup is what survives the 2-minute scan and converts 10k
LOC into a research-taste narrative. Treat **Phase 3 (demo polish)** as parallel/optional polish. Slot
the writeup *after* your Neural Decoder Zoo / Assad deliverables if they conflict — but the
honesty-and-evaluation framing here is genuinely the same muscle the physician-scientist path rewards,
so it's not wasted motion. **Recommendation: aim for Phase 0+1+2; let Phase 3 be opportunistic.**

---

## Appendix — key sources
- Eugene Yan, *How to Interview and Hire ML/AI Engineers* — eugeneyan.com/writing/how-to-interview/
- *The 37 Implementation Details of PPO* — iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- amid.fish, *Lessons Learned Reproducing a Deep RL Paper* — amid.fish/reproducing-deep-rl
- Agarwal et al., *rliable / Deep RL at the Edge of the Statistical Precipice* — arxiv.org/abs/2108.13264
- Henderson et al., *Deep RL that Matters* — arxiv.org/abs/1709.06560
- Olah & Carter, *Research Debt (Distill)* — distill.pub/2017/research-debt/
- Karpathy, *A Recipe for Training Neural Networks* — karpathy.github.io/2019/04/25/recipe/
- *ML Engineer Portfolio Projects That Will Get You Hired* — interviewnode.com (+ "100 portfolios" red-flags analysis)
- Neuralink MLE posting — job-boards.greenhouse.io/neuralink/jobs/5663271003
- AlphaHoldem (AAAI 2022) — cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf
