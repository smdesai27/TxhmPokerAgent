# Active Stage D Scripts

These scripts are the active Stage D FCHPA EV-first workflow.

## Core Training/Eval Jobs
- `scripts/slurm_stage_d_fchpa_corrective_train.slurm`
- `scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm`
- `scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm`

## Gate + Selection Jobs
- `scripts/slurm_stage_d_fchpa_eval_gate_check.slurm`
- `scripts/slurm_stage_d_fchpa_recovery_select.slurm`
- `scripts/slurm_stage_d_fchpa_recovery_prepare_winner.slurm`

## Submit Wrappers
- `scripts/submit_stage_d_fchpa_recovery_cycle.sh`
- `scripts/submit_stage_d_fchpa_interview_continue.sh`

Legacy and experimental stage scripts are archived under:
- `scripts/archived/legacy_stage_runs/`
