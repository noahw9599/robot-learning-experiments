# Evaluation Tools

## G1 Rollout Diagnostic

`g1_policy_rollout_diagnostic_v1.py` restores a PPO checkpoint and records per-step balance and control signals, including:

- center of mass relative to the foot support region
- torso orientation
- pelvis linear and angular velocity
- foot contacts and contact forces
- action magnitude and actuator force
- selected leg-joint positions, velocities, and targets

It supports deterministic or stochastic actions, multiple seeds, configurable environment overrides, randomized pushes, and optional rendering.

## Panda Transport Evaluator

`evaluate_panda_transport_metrics.py` records behavior-level manipulation outcomes rather than relying on total reward alone. Metrics include lift, hold, target approach, post-lift gripper opening, transient near-target open-gripper windows, final target distance, and final target success. The transient-window metric is intentionally not labeled stable placement because it does not prove that the object remained at the target.

Large checkpoints and raw run directories are intentionally excluded. Four small, selected rollout videos are included for qualitative before-and-after comparison.
