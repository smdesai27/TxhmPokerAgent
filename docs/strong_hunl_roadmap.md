# Path to a "strong" FCHPA full-HUNL bot — realistic roadmap

> From a 4-lens design workflow (define-strong / end-to-end-RL / CFR+search / pragmatic-solo) + synthesis,
> grounded in the verified literature (see HANDOFF.md §5.3) and our actual codebase. 2026-06-18.

## The two "strong"s — and which is reachable solo
- **Strong = wins head-to-head (mbb/h).** Reachable in principle but **throughput-capped**: AlphaHoldem used
  ~2.7B hands / 8 GPUs × 3 days; our self-play is single-process CPU (`self_play.py`). Solo ≈ 1–2% of that.
  Our R1–R10 ablations already showed the *algorithm* tweaks are nulls (Trinal-Clip bit-identical on-policy;
  simplified kBSP underperformed). Scaling is a race we can't win, not the thesis.
- **Strong = low-exploitability / near-Nash.** Only the **CFR + depth-limited search** path (DeepStack/ReBeL/
  Student of Games) reaches it — a *different architecture* (counterfactual-value vector net + re-solving),
  ~8–12 eng-weeks minimum, full HUNL is 6–12 mo + GPU cluster. Out of scope.
- **Exploitability is INTRACTABLE to measure on HUNL** (~10^161 states). So "strong" must be a tiered,
  measured claim, never "near-Nash."

## What "strong" should MEAN (tiered, measured, honest)
Report all tiers with Student-t CIs over duplicate/mirrored-seat hands (machinery exists: `stats.py`,
`evaluator.evaluate_duplicate`).
- **Tier 1 — own-ladder + behavior:** champion beats prior checkpoints (Stage C, 34k, collapsed latest.pt)
  head-to-head, CI excludes 0; passes behavior gates. Cheating-proof; retires the "+897 vs near-random" claim.
- **Tier 2 — LBR lower bound:** Local Best Response over the FCHPA tree → a guaranteed *lower bound* on
  exploitability. "Strong" = relative: champion's LBR ≪ always-call and ≪ collapsed checkpoint. Low LBR does
  NOT prove low exploitability (Lisý&Bowling: head-to-head-tied bots differ ~1300 mbb/g); high LBR is a
  disqualifier. **Validate LBR on Leduc vs exact `nash_conv` first** (LBR ≤ NashConv).
- **Tier 3 — Slumbot (headline if reachable):** mbb/h vs the free public bot (our chip economy already matches:
  STACK 20000 / SB 50 / BB 100 == Slumbot 200bb 50-100). Anchors: fold-always = −750 mbb/g; AlphaHoldem +111.6.
  Defensible target = "competitive / loses slowly" (CI ~ −50 to −150 mbb/h) given our 5-action abstraction
  handicap; a CI-significant WIN is aspirational.

## Staged roadmap (recommended order)
| Stage | Goal | Key work | Gate | Cost |
|---|---|---|---|---|
| **0 — own-ladder** | kill "+897 vs near-random" landmine | checkpoint-vs-checkpoint via `PolicyAgent` + `evaluate_duplicate`; demote MCCFR-ES to "near-random floor"; make `RuleBasedAgent` a real deterministic TAG so duplicate variance-reduction works | champion beats Stage C/34k/latest.pt, CI excl. 0; gates pass | ~2–3 days, CPU, no training |
| **1 — LBR** | only HUNL exploitability-flavored number (lower bound) | `lbr_agent.py` (1–2 ply best-responder + range belief + equity rollout via `_ModelPolicyAdapter`); `validate_lbr.py` on Leduc vs `nash_conv` + the CFR+ ~0.05 anchor | LBR ≤ NashConv on Leduc; champion LBR < always-call & < collapsed | ~0.5–1 wk, CPU |
| **2 — Slumbot** *(optional)* | the cited external headline | `slumbot_client.py` (no-auth HTTP) + `abstraction_bridge.py` (5-action ↔ continuous bets — load-bearing/risky) + `evaluate_vs_slumbot.py`, ≥100k duplicate hands | bridge round-trip tested; mbb/h + CI + abstraction caveat; a clean LOSS is publishable | ~1–1.5 wk, network-bound; API has no SLA |
| **3 — writeup** | the actual deliverable | rigorous-eval + honest-negatives story; apply all landmine guards (AlphaHoldem-INSPIRED; never juxtapose with +111.6; never attach Leduc 0.56 to HUNL agent) | skeptic accepts headline; every number has CI + sample size + caveat | ~3–5 days, no compute |

## Honest ceiling (what we could defensibly claim)
"An AlphaHoldem-INSPIRED from-scratch deep-RL HUNL agent measured with unusual rigor — beats all prior
checkpoints head-to-head (Student-t CIs exclude 0), carries a Leduc-validated LBR exploitability lower bound
below its rejected checkpoints and below always-call, behavior gates auto-caught policy collapse, and a Leduc
exact-NashConv probe (NFSP-lite ~0.56 vs CFR+ ~0.05 vs random 4.76) shows self-play cycles while strategy-
averaging stabilizes." + (if Stage 2) "plays Slumbot to a measured, bounded margin (realistically a loss of
tens-to-low-hundreds of mbb/h), honestly framed as a 5-action-abstracted agent vs full-action Slumbot."
**Never claimable:** near-Nash, unexploitable, any HUNL exploitability *number* (only an LBR bound), or "beats
Slumbot" without a CI showing it. The differentiator is **measurement rigor + honesty, not a record win-rate**
(the ~5–6 orders-of-magnitude throughput gap puts that permanently out of solo reach).

## Why not the other paths (as PRIMARY)
- **End-to-end RL scaling (faithful AlphaHoldem):** bottleneck is throughput, not the algorithm; our ablations
  proved the headline tweaks are nulls. ~3–4 eng-weeks for polish, not a thesis. (One real bug worth a flagged
  A/B: `league_manager.py:72` PFSP is inverted — "stronger = more likely to sample".)
- **CFR + search (DeepStack/ReBeL-lite):** different architecture (counterfactual-value vector net + re-solve);
  8–12 eng-weeks minimum. Its one cheap high-value piece — the LBR bound — is already pulled into Stage 1.

## Off-track recommendation (given ~1-month other deadline)
Do **Stage 0 + Stage 1 + Stage 3** (~1.5–2.5 eng-weeks, no retraining, no external dependency). Stage 2
(Slumbot) and any PPO scale-up are strictly optional/time-permitting. **Do NOT** spend the month on a
50–100k-iter PPO run — R1–R10 proved it improves head-to-head but NOT exploitability; that's the trap, not the
thesis. Position the artifact as transferable real-time-ML + first-principles + measurement-discipline evidence.
