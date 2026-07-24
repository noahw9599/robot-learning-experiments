# Project Retrospective

## Project Goal

This project began as a hands-on introduction to modern robot learning. The goal
was to understand MuJoCo, reinforcement learning, reward design, policy
evaluation, and cloud training by reproducing strong baselines before proposing
new algorithms.

The final repository contains two focused studies:

1. Unitree G1 humanoid locomotion reproduction and disturbance evaluation.
2. Franka Panda staged manipulation reward design and metric auditing.

## Technical System

- Simulation: MuJoCo Playground with the MJX backend
- Learning algorithm: Brax Proximal Policy Optimization (PPO)
- Compute stack: JAX, XLA, CUDA, and NVIDIA A10G cloud GPUs
- Experiment tracking: TensorBoard, checkpoint manifests, run reports, and raw
  evaluation JSON
- Reproducibility: pinned upstream source, reversible patch scripts, deterministic
  seeds, and bounded evaluation protocols
- Operations: headless EGL rendering and automatic shutdown safeguards for
  cost-controlled cloud runs

## Study 1: Humanoid Locomotion

Early custom balance and walking experiments frequently fell forward, stepped
with only one leg, or exploited rewards through scooting and other non-walking
motion. These failures showed that reward improvement alone was not evidence of
useful locomotion.

The project returned to the official `G1JoystickFlatTerrain` baseline to establish
a known-good reference:

| Metric | Result |
|---|---:|
| Training transitions | 202,342,400 |
| Initial evaluation reward | -5.801 |
| Final evaluation reward | 15.578 |
| Peak evaluation reward | 16.157 |
| Training runtime | Approximately 2 hours 32 minutes |

The resulting policy produced sustained walking and exited the fixed camera
frame. A deterministic diagnostic then recorded torso orientation, center of
mass relative to support, velocity, contacts, action magnitude, and actuator
force.

A randomized push evaluation tested five independent seeds for 250 steps each:

| Metric | Result |
|---|---:|
| Evaluation seeds | 5 |
| Total evaluation steps | 1,250 |
| Full-horizon survival | 5/5 |
| Early terminations | 0 |

This is evidence for repeatability under the defined protocol, not a universal
robustness claim.

## Study 2: Manipulation Curriculum

The initial Panda policy learned to drag the cube rather than lift it. Reward
shaping was introduced in stages so that each behavioral change could be
isolated:

1. Lift the cube.
2. Hold it above the surface.
3. Transport it toward a target.
4. Open the gripper near the target.

The v3 policy visibly grasped, lifted, held, and transported the object. The final
v4 checkpoint was evaluated over 60 episodes across three deterministic seeds:

| Metric | Mean across seeds |
|---|---:|
| Lift success | 100% |
| Hold success | 100% |
| Target-region approach | 96.7% +/- 5.8 percentage points |
| Post-lift gripper opening | 86.7% +/- 2.9 percentage points |
| Transient near-target open-gripper window | 85.0% +/- 5.0 percentage points |
| Final target placement | 0% |
| Mean closest target distance | 3.72 +/- 0.45 cm |
| Mean final target distance | 77.64 +/- 0.95 cm |

## Most Important Failure

The first version of the evaluator called a ten-step near-target open-gripper
window "stable placement." A rendered rollout showed the cube passing near the
target and then falling away. Rechecking the saved final-state metric confirmed
0 successful final placements in 60 episodes.

The environment itself sampled targets in unsupported free space. A released
object cannot remain at such a target under gravity without a platform,
receptacle, or grasp-maintenance objective. The public claim was corrected, and
the evaluator now separates transient behavior from end-of-episode success.

This was not just a failed reward. It demonstrated a central research lesson:
the metric, environment physics, and stated task must describe the same outcome.

## Other Negative Results

- A custom G1 recovery objective did not outperform the official checkpoint
  under the paired push protocol.
- Expanding the G1 actor observation from 103 to 109 features invalidated the
  existing network input and normalization state. Two implementations reached
  expensive compilation but no valid training update within the bounded budget.
- Walker experiments exposed reward hacking through dragging, scooting, and
  asymmetric stepping.

These results are preserved because they changed the experimental method:
baseline first, one variable at a time, explicit stop conditions, and behavioral
evaluation before more training.

## What I Learned

- PPO improves a distribution over actions; it does not retain and replay the
  single best rollout.
- Reward curves are optimization diagnostics, not proof of task success.
- Termination conditions affect both behavior and sample efficiency.
- Observation changes can break checkpoint compatibility even when the reward
  and network hidden layers are unchanged.
- Multi-seed metrics and deterministic diagnostics are more credible than one
  selected video.
- Simulation task design must respect physical feasibility.
- Compilation, rendering, storage, and idle instances are part of the real
  research budget.

## Final Scope

The project successfully reproduced a humanoid locomotion baseline, evaluated it
under a defined disturbance protocol, changed manipulation behavior through
staged reward design, and built instrumentation that caught an overstated
success metric.

It did not demonstrate real-world transfer, true object placement, imitation
learning, or a novel reinforcement-learning architecture. Those are appropriate
directions for a future project built on the experimental discipline developed
here.
