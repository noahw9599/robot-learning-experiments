# Reversible Environment Patches

These scripts modify pinned MuJoCo Playground environment files while preserving a backup for restoration. They are research artifacts, not a replacement for maintaining a long-lived fork.

## Panda Curriculum Order

1. `apply_panda_lift_reward_v2_patch.py`
2. `apply_panda_transport_reward_v3_patch.py`
3. `apply_panda_transport_reward_v4_patch.py`

The transport v4 patch requires v3 to be active because it extends the transport objective with near-target gripper-opening behavior. The final-state audit showed that this did not produce stable placement; see `docs/experiments/panda-manipulation-curriculum.md`.

## G1 Recovery Reference

`apply_g1_recovery_reference_v1_patch.py` records a controlled recovery-reference experiment. Its scalar reward did not outperform the official baseline in the five-seed push evaluation, which is why the repository reports it as a diagnostic branch rather than a replacement baseline.

## Safety

- Run patches only against the pinned MuJoCo Playground revision documented in the repository.
- Review the generated diff before training.
- Keep the automatically created backup until the experiment has been reproduced.
- Do not evaluate an existing checkpoint against an observation or network interface with a different shape.
