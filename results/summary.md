# Verified Results Summary

Only results backed by saved checkpoints or instrumented evaluations are included here.

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
| Release success | 86.7% +/- 2.9 percentage points |
| Stable placement | 85.0% +/- 5.0 percentage points |
| Closest target distance | 3.72 +/- 0.45 cm |

## Resume-Safe Description

Reproduced a Unitree G1 locomotion baseline in MuJoCo Playground using JAX/MJX and Brax PPO, evaluated the policy under randomized planar pushes, and developed a staged Panda manipulation curriculum that achieved 85% stable placement over 60 episodes across three seeds.

## Scope

These figures describe the recorded simulation protocols in this repository. They do not imply real-world transfer or performance outside the tested command, disturbance, object, and target distributions.
