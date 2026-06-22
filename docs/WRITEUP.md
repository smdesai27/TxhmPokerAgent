# Knowing whether it was good was the hard half

*An AlphaHoldem-inspired self-play RL agent for Heads-Up No-Limit Hold'em — and the evaluation discipline that kept catching my "better" models being worse.*

---

Building the agent was the easy half; knowing whether it was good was the hard half.

This is an AlphaHoldem-inspired, from-scratch self-play reinforcement-learning agent for Heads-Up No-Limit Texas Hold'em (HUNL), written in PyTorch. The architecture borrows the silhouette of the AlphaHoldem paper (Zhao et al., AAAI 2022): a pseudo-siamese two-tower network — a card tower over the 7 cards and an LSTM action tower over the betting history — feeding a shared trunk and policy/value heads, around 1.5M parameters, trained with vanilla PPO + GAE against a K-best self-play league over OpenSpiel's universal_poker. It runs on a single GPU.

It is worth being precise about what this is *not*. It is not a reproduction of AlphaHoldem — the input representation, the loss, the betting abstraction, and the scale all deviate from the paper, deliberately, so the whole thing trains on one machine. It is not a Slumbot-beater; I never measured the agent against Slumbot, and at the scale a solo developer can reach, beating it is out of reach. And it is not near-Nash: on full HUNL, exploitability is intractable to even measure (on the order of 10^161 information sets), so I make no exploitability claim about the HUNL agent at all.

So if not those, then what is the point? The interesting content here is the evaluation, not the agent. The differentiator is measurement discipline and honest negative results: choosing metrics that can't be gamed, building a harness that repeatedly caught regressions the training curves had masked, and reporting the failures plainly rather than dressing them up.

Three acts carry that thesis. First, a rigorous small-game study: on Leduc poker — small enough that exact exploitability (NashConv) is computable — the from-scratch self-play stack walks through the textbook failure-and-fix, showing where naive self-play's guarantees end. Second, a reconciliation: AlphaHoldem reports "convergence" via training loss, Elo, and head-to-head win-rate, but never measures exploitability — so its claim and my small-game finding are about different quantities, and I work through why that distinction matters. Third, a scaled attempt: I built a vectorized collector, fixed a self-play bug, and ran 90,000 iterations on a GPU (~9e7 hands) warm-started from the best checkpoint — and a fixed-reference evaluation caught that the new model had quietly regressed, losing 511 bb/100 (95% CI [−582, −440], 12k duplicate pairs) to the very checkpoint it started from, even as the training metric stayed positive the entire run.

The honest ceiling: strong-but-exploitable, measured carefully. The value is the method, not a magnitude.

## The system, and the harness that judged it

### The agent, briefly

The model is the pseudo-siamese actor-critic described above, roughly 1.5M parameters. A **card tower** (a 53-token embedding followed by a 3-layer MLP over the 7 cards — 2 hole, 5 community) and an **LSTM action tower** (over the betting history) feed a shared trunk, which splits into a policy head and a value head. A few design choices were load-bearing for stability rather than for headlines: **LayerNorm throughout** (PPO's small, correlated batches make BatchNorm statistics unreliable), and **post-forward action masking** — the network emits raw logits and legality is applied afterward, so the masking never has to be differentiable or live inside the trunk.

Training is vanilla **PPO + GAE** against a **K-best self-play league** (Elo-rated, with prioritized-fictitious-self-play opponent sampling) on OpenSpiel's `universal_poker`, under discrete betting abstractions — FCPA (4 actions) and FCHPA (5). That part worked. It is not the interesting part.

### Knowing whether it was any good

The harder half was measurement, and most of the engineering effort went here. The agent emits a single win-rate number against a baseline, and that number is easy to fool — an over-folding policy can post a large margin against a weak opponent while being strategically degenerate. So the harness does not promote on win-rate alone. It runs **behavior gates**: envelopes on fold frequency, aggression, policy entropy, and the pot-size mix of bets. A run only graduates if it stays inside all of them.

This caught a real regression. A later, longer run scored well on the win-rate metric but had **collapsed to a fold-heavy, low-entropy policy** — it was winning by quitting. The entropy and fold gates flagged it automatically and the harness **refused to promote it**. The training-side metric never noticed; the gates did. That episode is the whole argument for having them.

Two further corrections made the numbers trustworthy:

- **Student-t confidence intervals.** With only 3–5 seed means, the prior Normal (1.96) intervals were roughly **40% too narrow** — they overstated certainty. Switching to a Student-t critical value (df = n−1) widened them honestly; on the champion's headline tier the half-width went from ±75 to **±106**.
- **Duplicate / mirrored-seat hands.** The agent plays both holdings of the same deal under a common RNG, cancelling hole-card luck. (Against a *stochastic* opponent the reduction is weak — opponent randomness and board runout survive the pairing — which the docstring states plainly; it bites hardest against deterministic opponents.)

### The champion, and one number I retired

The defensible model is the **Stage D FCHPA 21k checkpoint** (`interview_ready_1`): it passes every gate and plays a balanced, non-degenerate strategy — **entropy 1.44 bits, fold 46%, raise 11%**.

One number does *not* survive scrutiny, and the harness is what exposed it. The often-quoted **"+897 bb/100 vs MCCFR-ES"** is not a strength-versus-solver result. OpenSpiel's MCCFR average policy plays uniform-random on unvisited information states, and full-deck HUNL has far too many for the solver to be anything but near-random on most lines. The harness's own zero-iteration (uniform) control proves it: adding MCCFR iterations makes the opponent *easier*, not harder — the `control_delta` is negative across every tier. So I treat that figure as a sanity check against a near-random control, not as evidence of strength.

That retirement is what motivated the next act. If win-rate against a baseline can be this misleading, I needed a metric that does not depend on the choice of opponent at all.

## The rigorous core: exact exploitability on Leduc

Head-to-head win-rate, the metric most poker bots report, is opponent-dependent and gameable — Lisý and Bowling's local-best-response work showed two bots that tied head-to-head (within ~20 mbb/g) yet differed by roughly 1300 mbb/g in exploitability. The ungameable metric is exploitability (NashConv): the gain an optimal counter-strategy extracts, independent of any fixed opponent. On full HUNL that quantity is intractable — the game has on the order of 10^161 information sets — so I could not measure it on the agent I trained. Instead I moved the question to a game where the same metric is *exact*: Leduc poker, 936 information states, where OpenSpiel computes NashConv in closed form. Leduc is not the deliverable; it is an instrument for probing the **learning dynamics** of the exact PPO + self-play recipe the HUNL stack uses. The harness reuses the project's own GAE and post-forward masked logits and the same PPO hyperparameters as the HUNL configs, so it exercises the real recipe rather than a toy.

One units caveat shapes every comparison below: OpenSpiel's `nash_conv` returns the **sum of both players' best-response gains**, roughly 2x the single-player "exploitability" most papers quote. So tuned NFSP's published ~0.06 single-player sits near ~0.12 on this axis, and tabular CFR+ lands around ~0.05 NashConv. On the same axis a uniform-random policy measures ~4.76 and Nash is 0.

### The investigation (R1–R10)

I ran the recipe through ten rounds of sweeps (~35 configs). The arc is a textbook failure-and-fix:

- **The tuned PPO self-play iterate cycles.** Entropy, larger batches, and learning-rate annealing drove the *best checkpoint* to NashConv 0.67 (~7x below random), but the current iterate does not settle — its final value diverges back up into the 1.3–2.6 range, and on the 500-iter baseline the iterate spikes as high as 4.88 while its time-average holds near 2.0. This is the expected signature of best-response cycling in imperfect-information self-play: the current iterate orbits the equilibrium rather than approaching it.
- **Averaging the snapshot networks fails.** The obvious fix — averaging policy-network parameters/outputs over the trajectory — gave 0.84–1.28, *worse* than the best single checkpoint. The iterates make wide excursions (0.67 to ~2.6), so averaging their parameters yields a muddy mixture, not the equilibrium.
- **NFSP-style strategy-averaging is the only non-divergent estimator.** Averaging the *strategy* — training a supervised average-policy net on a reservoir of best-response actions, the NFSP-lite construction (Heinrich and Silver 2016) — stops the cycling. With a strong best-responder over 20k iterations across two seeds, the averaged policy stabilizes at NashConv finals of 0.538 and 0.588 (mean 0.56). The curve is broadly decreasing then plateauing with local fluctuations; it is not monotone (one seed up-steps near 19k, the other rises slightly at the very end) and it does not approach Nash.

I then spent three more rounds (R8–R10) trying to make the *iterate itself* converge last-iterate, via MMD / NashPG — PPO with a reverse-KL term toward a periodically-refreshed magnet policy (the MMD → R-NaD/DeepNash lineage). This is an honest negative: across hard-refresh, explicit proximal-KL, and smooth-EMA / no-refresh magnets, and sweeps over the KL coefficient, refresh rate, and epoch count, regularization reliably **damped the catastrophic divergence** (plain PPO blows up to ~4.88; MMD stays bounded ~1–2.5) but never pinned the iterate low — it found ~0.8–1.0 dips and then wandered ~0.9–2.5. The diagnosed cause matches the literature: clean MMD/R-NaD/NeuRD rely on all-actions counterfactual values, whereas a *sampled* GAE advantage is too noisy. Supplying all-actions values is Deep-CFR-scale work, out of scope here. A faithful-NFSP variant (anticipatory eta = 0.1) gave a cleaner monotone descent but was data-starved on an on-policy PPO best-responder (0.73 at 8k), so the eta = 1 result above stands.

### What this is, and is not

Calibrated against the references — random ~4.76, tabular CFR+ ~0.05, tuned NFSP ~0.06, Nash 0 — the 0.56 plateau is ~8x below random but still ~9x above tuned NFSP. It is **not near-Nash and not unexploitable**, and the magnitude is not a result on its own: Leduc is a solved game where tabular CFR reaches ~0 in seconds. A four-lens adversarial review (game-theory, statistics, skeptic, recruiter) returned a unanimous verdict to report this as **methodology, not a headline number**. I agree. The contribution is the discipline: choosing an exact, deterministic, opponent-independent metric precisely *because* HUNL's is intractable; a controlled three-way ablation isolating *why* strategy-averaging works; and honest negatives reported rather than buried — snapshot-network averaging rejected, NFSP-lite initially worse than the PPO checkpoint, Trinal-Clip bit-identical to vanilla at Leduc's chip scale, and MMD failing to converge last-iterate. This is a learning-dynamics probe on Leduc; the 0.56 number belongs to that probe and is never attached to the HUNL agent.

## Reconciling with AlphaHoldem's "convergence"

An obvious objection: AlphaHoldem reports that its agent converged, so why does the self-play iterate in my Leduc probe cycle rather than settle? The short answer is that we measured different quantities. There is no contradiction.

AlphaHoldem never measures exploitability. Its notion of "convergence" rests on three signals: the Trinal-Clip PPO training loss flattening, Elo plateauing inside its self-play league (the model-selection metric), and head-to-head win-rate in mbb/h — the paper reports +111.6 against Slumbot, with smaller margins against a DeepStack reimplementation and against humans. On full HUNL (on the order of 10^161 information sets), it explicitly concedes that best-response computation is prohibitive and never computes a NashConv. So "converged" there means "the loss stopped moving, the league rating stabilized, and it wins games" — not "approaches a Nash equilibrium."

That distinction matters because win-rate is not exploitability, and as the Lisý and Bowling result already showed, the gap can be enormous — two agents tied head-to-head yet ~1300 mbb/g apart in measured exploitability. A strong head-to-head record is therefore consistent with a strategy that remains highly exploitable — it simply cannot be detected without a best-response test, which HUNL does not admit.

So my Leduc study (936 information states, where exact exploitability is computable) is not a failed reproduction; it is the test AlphaHoldem could not run. It shows that on a game where the answer is checkable, a plain self-play iterate cycles, while strategy-averaging is the only non-divergent estimator I found.

It is worth being precise about what AlphaHoldem genuinely did differently: Trinal-Clip PPO to bound a high-variance gradient, K-Best self-play to damp Elo cycling against the league, pseudo-siamese CNN towers, and roughly 2.7B hands over 8 GPUs for three days. Each of those targets win-rate and training stability. None of them targets exploitability — so none of them resolves the dynamics the small-game probe exposes.

## The scaled GPU attempt — and the regression the evaluation caught

Having a champion that passed every gate, I wanted to know whether I could push it further: a stronger head-to-head FCHPA agent, trained at larger scale on a GPU. This is the part of the project where the spine — *knowing whether it was good was the hard half* — earned its keep.

### Making scale possible

The first obstacle was throughput, not the algorithm. The original self-play collector stepped one game at a time with batch-size-1 inference, which left the GPU effectively idle — the wall-clock was dominated by Python-side game stepping, not matrix multiplies. I rewrote the collector to run B games in lockstep and batch all policy/value inference into a single forward pass, which lifted collection throughput by roughly 30–100x depending on configuration. Along the way I fixed an inverted opponent-sampling bug in the league (the PFSP weighting had been favoring the *strongest* opponents, the opposite of the intended prioritization) and switched the loss to Trinal-Clip.

With that in place I warm-started from the champion (the Stage D FCHPA 21k checkpoint) and ran 90,000 iterations on a single GPU — about 38 hours of wall-clock and on the order of 9×10⁷ hands.

### The training metric looked fine the whole way

Throughout the run, the metric you would naturally watch — rollout return against the co-evolving self-play league — stayed positive, hovering around +3 to +12 bb the entire time. By that signal the model was steadily winning. Nothing in the training curve suggested a problem.

### What the fixed-reference ladder showed

The training metric is measured against a *moving* opponent, so it can only tell you the model is improving relative to a league that is itself drifting. To get an absolute read I ran the own-checkpoint ladder: the new model against a fixed strong reference — the champion — over 12,000 duplicate, mirrored-seat hands.

First the sanity check. New-versus-new (the same checkpoint against itself) came back at +14.6 bb/100 with a 95% CI of [−11, +40], straddling zero — the harness was behaving correctly. Then the real comparison:

> **New (90k) vs Champion (21k): −511 bb/100, 95% CI [−582, −440].**

The scaled model *lost* to the checkpoint it started from, decisively and outside the interval. I confirmed it was not a loading artifact — the strict `load_state_dict` accepted the weights without error, so the trained network was genuinely the one being measured. After 38 hours and ~90M hands, the model had regressed.

### Diagnosing it: over-aggression drift, not collapse

A behavior probe located the failure precisely. The new model was not collapsed — against a random opponent it scored +1029 bb/100, even harder than the champion's +948. But against a calling station (an always-call opponent), it lost −612 bb/100, where the balanced champion *won* +539. It had drifted into a bluff-happy style: punishing opponents that fold, trivially exploited by anyone who simply calls down.

The root cause was in the training mix. There were no calling opponents present to punish over-aggression, and the co-evolving league learned to fold to the rising aggression — a feedback loop that rewarded ever-more-aggressive play right up until it met something that wouldn't fold.

### Why this is the point, not a footnote

I'll be plain: as an attempt to build a *stronger* bot, this was a negative result. The champion remains the strongest model I have.

But it is the cleanest demonstration of the entire project's thesis. The training metric — return versus the co-evolving league — stayed positive for the full run while the strategy quietly wandered away from a certified-good policy. Only the fixed strong reference exposed it. That is the same lesson the small-game learning-dynamics probe surfaced, and the same caution the AlphaHoldem authors raise about league win-rate: progress measured against a moving target is not progress. The evaluation harness caught exactly the kind of regression that training curves are structurally unable to see — which is the reason I built it the way I did.

## Throughlines, what it demonstrates, and honest limitations

Three throughlines run through this project, and none of them is "I built a poker bot."

**1. Choose ungameable metrics over flattering ones.** The headline win-rate against the MCCFR-ES baseline turned out to be near-random — the harness's own zero-iteration control beat the agent by more than the trained tiers did, so the number measured the opponent's weakness, not the agent's strength. The response was to move toward metrics that resist gaming: exact exploitability (NashConv) on Leduc, where it is computable over 936 information states; fixed-reference self-play ladders; and a Local Best Response lower bound. On Leduc, with the same recipe as the HUNL configs, the raw PPO self-play iterate cycles (best checkpoint 0.67, final wandering to 1.3–2.6), averaging the snapshot networks fails (0.84–1.28), and only NFSP-style strategy-averaging is non-divergent, plateauing at NashConv 0.538 / 0.588 across two seeds (random ≈ 4.76, tuned NFSP ≈ 0.06, Nash = 0). That last number is roughly 9x above tuned NFSP — reported as methodology, not as a headline.

**2. Honest negative results are the backbone.** The self-play iterate cycles rather than settling. Network-parameter averaging fails. Trinal-Clip is bit-identical to vanilla PPO at Leduc's ~14-chip scale (the dual-clip never fires on-policy). The simplified Elo-kBSP league underperformed a plain rolling pool (averaged-policy exploitability 2.49 vs 1.50 on the ablation). And the scaled run — a vectorized collector at roughly 30–100x throughput, 90,000 iterations, ~9e7 hands, warm-started from the champion — lost 511 bb/100 (CI95 [−582, −440], 12k duplicate pairs) to that champion. The behavior probe diagnosed over-aggression drift: the new model beat a random opponent harder than the champion (+1029 vs +948) but was trivially punished by a calling station (−612, where the champion sat at +539). Root cause was a training opponent mix with no calling opponents and a co-evolving league that folded to aggression.

**3. Evaluation discipline repeatedly caught what training metrics masked.** Throughout that scaled run, the rollout reward against the co-evolving league stayed positive; only the fixed-reference ladder surfaced the regression. The behavior gates earlier refused to promote an entropy-collapsed checkpoint. The Student-t CI correction widened a half-width from ±75 to ±106, exposing intervals the Normal approximation had understated by roughly 40%.

The honest ceiling: this is strong-but-exploitable, measured rigorously — not near-Nash, not a record win-rate. The throughput gap against AlphaHoldem's ~2.7B hands over 8 GPUs and 3 days is real and not closeable solo. What transfers is the part that matters for real-time ML work: first-principles systems building, measurement rigor, and the discipline to publish the negative result rather than the flattering one.

---

## Code and artifacts

Every quantitative claim above is backed by a committed config, script, or run log:

- **Leduc exploitability investigation (R1–R10):** [`docs/leduc_sweep_log.md`](leduc_sweep_log.md) (round-by-round results + the "report as methodology" review), harnesses `poker_rl_agent/scripts/validate_leduc_exploitability.py` (PPO self-play, MMD/NashPG) and `validate_leduc_nfsp.py` (strategy-averaging), tabular anchor `leduc_cfr_anchor.py`. The ~0.05 CFR+ figure is the local anchor run on the NashConv axis.
- **Reconciling with AlphaHoldem + the path-to-strong analysis:** [`HANDOFF.md`](../HANDOFF.md) §5.3 and [`docs/strong_hunl_roadmap.md`](strong_hunl_roadmap.md).
- **The scaled GPU attempt + the regression:** [`docs/strong_fchpa_run_state.md`](strong_fchpa_run_state.md) (run config, throughput fix, the −511 ladder result, and the over-aggression behavior probe). The vectorized collector is `poker_rl_agent/algorithms/vectorized_self_play.py`; the PFSP fix is in `poker_rl_agent/training/league_manager.py`; the fixed-reference ladder is `poker_rl_agent/scripts/evaluate_own_ladder.py`.
- **Evaluation harness:** behavior gates and duplicate-hand evaluation in `poker_rl_agent/evaluation/evaluator.py`; Student-t intervals in `poker_rl_agent/evaluation/stats.py`.

*Reproducibility note: head-to-head figures are reported with sample size and 95% CI; the behavior-probe point estimates (vs random / always-call) are diagnostic — their intervals are in `docs/strong_fchpa_ladder/behavior_probe.json` and do not overlap, so the qualitative diagnosis (loses to a calling station, beats a folder) holds.*