# Lowering Leduc exploitability below the ~0.56 NFSP-lite plateau — ranked ideas

> Source: a 4-agent research workflow (3 online-literature lenses + 1 code audit of the harness) +
> adversarial synthesis, 2026-06-17. Companion: [leduc_sweep_log.md](leduc_sweep_log.md) (the 7-round
> result this builds on).

## Units caveat (state this in any writeup — it changes the comparison)
OpenSpiel `exploitability.nash_conv()` returns **NashConv = sum of BOTH players' best-response gains**,
≈ **2× the single-player "exploitability"** most papers report. So:
- Our axis: random ≈ **4.76**, our NFSP-lite ≈ **0.56**, Nash = 0.
- Published NFSP avg-policy ≈ 0.06 single-player ≈ **~0.12 NashConv** (our axis).
- Tabular CFR+/DCFR ≈ **~0.004–0.02 NashConv**, reached in seconds.
- Deep CFR / DREAM are usually quoted in **mbb/g** (different unit again).
Net: our 0.56 is **~4–5× above faithful NFSP**, not ~9×. Real gap, but smaller than the naive read.

## The tension (resolve it explicitly)
- **Lowest number** is trivially won by **tabular CFR** (~0) — but that abandons the RL-self-play recipe
  the project is about. Use it only as a labeled, different-paradigm **anchor line**.
- **Best story for THIS project** = make the *neural self-play iterate itself converge* (last-iterate),
  which directly answers our documented "the iterate cycles, only averaging works" finding.

## Ranked interventions
| # | Intervention | Category | Realistic NashConv | Effort | Lowers # / Improves story |
|---|---|---|---|---|---|
| 1 | **Moving-magnet MMD / NashPG** — add KL-to-a-periodically-refreshed-reference-net to `ppo_update` | regularized last-iterate | **~0.05–0.15 (last iterate, no averaging)** | ~40–80 LOC, ~0.5–1 day | **both** |
| 2 | **Faithful NFSP** — anticipatory η=0.1, gate reservoir to BR-mode adds, 2M reservoir, stronger/longer BR | faithful-nfsp | ~0.1–0.2 | ~50–100 LOC, ~1 day + more compute | lowers number |
| 3 | **Constant-α MMD (uniform magnet, no refresh)** — entropy + KL-to-previous-iterate; proof cycling stops | regularized last-iterate | ~0.1–0.2 (the α-QRE) | ~15–30 LOC, ~2–4 h | improves story (stepping stone to #1) |
| 4 | **NeuRD** — gradient on logits = advantage (drop through-softmax scaling); CFR-in-disguise | regularized last-iterate | ~0.2–0.4 alone (lower w/ magnet) | ~30–60 LOC, ~0.5 day | improves story |
| 5 | **Tabular CFR/CFR+ anchor** (OpenSpiel) — NOT an RL method; honest reference line | solver-baseline | ~0.004–0.02 in seconds | ~15 LOC, <1 h | improves story (honesty) |
| 6 | **QPG/RPG/RMPG** — all-actions regret-style actor-critic loss | regularized last-iterate | ~0.2–0.4 | ~40–70 LOC (needs all-actions Q plumbing) | both |

## Flagship (#1) — concrete recipe
MMD realized as a PPO add-on (the MMD paper itself implemented MMD as "PPO with reverse-KL regularization";
lineage MMD → R-NaD/DeepNash → NashPG):
1. Instantiate `ref_net = PolicyValueNet(...)`, load `net`'s weights (frozen "magnet" ρ).
2. In `ppo_update` minibatch loop (`validate_leduc_exploitability.py:~245–264`), after building `dist`,
   compute ref logits under `torch.no_grad()`, `ref_dist = Categorical(logits=masked_logits(ref_logits, legal[b]))`,
   `kl = kl_divergence(dist, ref_dist).mean()`, add `+ kl_coef * kl` to the loss.
3. In `main()`, every `--kl_refresh_every` iters: `ref_net.load_state_dict(net.state_dict())`.
4. New flags: `--kl_coef` (default 0.0 = off), `--kl_refresh_every`. Sweep α∈{0.5,0.2,0.1,0.05}, K∈{500,1000,2000}.
- **Validate:** the *current-iterate* NashConv curve should **flatten** instead of diverging to 1.3–2.6.
- **Risk:** α too small → still cycles; too large → stalls at a biased QRE (report honestly). Sampled
  (not all-actions) advantage is noisier than tabular MMD → expect ~0.05–0.15, not ~0. Ablate KL direction
  (fwd vs rev) and whether to keep the PPO clip alongside the KL.

## Code-audit cheap wins (from reading the actual harness)
- **NFSP-lite runs at effective η≈1** (BR always best-responds to π̄, reservoir adds every BR turn) —
  Heinrich & Silver show η≈1 **plateaus**. Restoring **anticipatory η=0.1** + **gating `reservoir.add` to
  BR-mode-only** (`validate_leduc_nfsp.py:99`) is the highest-leverage faithfulness fix.
- BR may be under-best-responding: one on-policy PPO batch/iter; raise `--ppo_epochs` (4→20–30).
- `hidden=128` MLP + reservoir 400k default are likely under-sized for the low-exploitability regime.

## Recommended plan (off-track portfolio → method matters more than magnitude)
1. **Now (<1 h):** add the **tabular CFR/CFR+ anchor** (#5) + the random-4.76 line so every figure is
   honestly calibrated. Write `scripts/leduc_cfr_anchor.py` (~15 LOC, OpenSpiel `cfr.CFRPlusSolver`).
2. **Flagship (~1 day):** implement **moving-magnet MMD / NashPG** (#1), staged as constant-α QRE first
   (#3) to *prove cycling stops*, then add the magnet refresh to drive toward Nash. This is the cool,
   on-thesis result: "we made our neural self-play iterate converge last-iterate, fixing the documented
   cycling — MMD/R-NaD lineage, ~40 LOC."
3. **Honest baseline (parallel, ~1 day + compute):** the **faithful-NFSP fixes** (#2) → confirm the 0.56
   plateau was a known faithfulness bug (η≈1), not a method ceiling; should reach ~0.1–0.2.
4. **Optional:** NeuRD (#4) as a cheap secondary ablation.
Frame the writeup as **"making neural self-play converge on Leduc,"** with CFR as a labeled different-paradigm
reference — **never** a claim to have beaten CFR.

## Key citations
- **MMD:** Sokota et al., "A Unified Approach to RL, QRE, and Two-Player Zero-Sum Games," NeurIPS'22/ICLR'23,
  arXiv:2206.05825, code github.com/ssokota/mmd.
- **R-NaD / DeepNash:** Perolat et al., "Mastering Stratego…," Science 2022.
- **NeuRD:** Hennes et al., "Neural Replicator Dynamics," AAMAS'20, arXiv:1906.00190.
- **Regret-based actor-critic (QPG/RPG/RMPG):** Srinivasan et al., NeurIPS'18, arXiv:1810.09026.
- **NFSP:** Heinrich & Silver 2016, arXiv:1603.01121 (anticipatory η, reservoir, ε-greedy BR).
