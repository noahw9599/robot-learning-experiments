# Verified Results Summary

Only results backed by saved checkpoints or instrumented evaluations are included here. Placement terminology reflects the final-state audit performed after rendering the v4 policy.

## Unitree G1 Locomotion

| Metric | Result |
|---|---:|
| Training transitions | 202,342,400 |
| Initial evaluation reward | -5.801 |
| Final evaluation reward | 15.578 |
| Peak evaluation reward | 16.157 |
| Push-evaluation seeds | 5 |
| Total push-evaluation steps | 1,250 |
| Full-horizon survival | 5/5 seeds |
| Early terminations | 0 |

## Franka Panda Manipulation

| Metric | Mean across 3 seeds |
|---|---:|
| Evaluation episodes | 60 total |
| Lift success | 100% |
| Hold success | 100% |
| Target-region approach | 96.7% +/- 5.8 percentage points |
| Post-lift gripper opening | 86.7% +/- 2.9 percentage points |
| Transient near-target open-gripper window | 85.0% +/- 5.0 percentage points |
| Final target placement | 0% |
| Closest target distance | 3.72 +/- 0.45 cm |
| Final target distance | 77.64 +/- 0.95 cm |

The transient-window metric records at least 10 consecutive steps with an open gripper and the cube within 8 cm of the target. It does not represent final placement. The target was sampled in unsupported free space, and all 60 episodes failed the evaluator's final-distance criterion.

## Resume-Safe Description

Reproduced a Unitree G1 locomotion baseline in MuJoCo Playground using JAX/MJX and Brax PPO, evaluated the policy under randomized planar pushes, and developed a staged Panda reward curriculum that achieved 100% lift and hold success with 96.7% target-region approach across 60 episodes. Audited a misleading transient release metric against final-state data and rendered behavior, identifying 0% true final placement and a task-design mismatch caused by unsupported free-space targets.

## Scope

These figures describe the recorded simulation protocols in this repository. They do not imply real-world transfer or performance outside the tested command, disturbance, object, and target distributions.
