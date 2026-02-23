# Stage D Goal Alignment

- Decision: freeze champion, do not promote 34k branch.
- Champion checkpoint: `checkpoints/snapshots/interview_ready/interview_ready_1.pt`
- Champion cert eval: `logs/stage_d/eval/eval_selected_21k_auto_cert_20260216_161321.json`
- Branch checkpoint: `checkpoints/snapshots/stage_d/recovery/longrun_rebalance_34k_20260220_012455.pt`
- Branch screen eval: `logs/stage_d/eval/eval_longrun_rebalance_34k_screen_20260220_012455.json`
- Branch gate log: `logs/stage_d/slurm/ah_stage_d_fchpa_gate_check_432657.out`

## Why Not Promote 34k

- Branch default CI95 lower: `747.652`
- Champion default CI95 lower: `814.955`
- Delta (branch - champion): `-67.303`
- Gate promotion pass: `False`
- CI-floor pass: `False`

## Stability Snapshot (34k Branch)

- nonfinite_grad_skips_max: `0`
- nonfinite_loss_batches_max: `0`
- preflop_fold_last100: `0.4224`
- preflop_call_last100: `0.4517`
- preflop_half_last100: `0.1110`
- preflop_pot_last100: `0.0021`
- preflop_allin_last100: `0.0129`
- preflop_entropy_last100: `1.4901`

## Interview Framing

- We promote only when statistical gates and CI-floor pass.
- This keeps claims conservative, reproducible, and technically defensible.
