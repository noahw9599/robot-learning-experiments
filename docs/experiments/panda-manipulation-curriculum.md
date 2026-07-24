# Panda Manipulation Curriculum

## Objective

Understand how reward design changes learned manipulation behavior, and verify that evaluation metrics match the behavior they claim to measure.

## Curriculum

1. Lift reward: encourage the gripper to raise the cube.
2. Hold reward: encourage sustained grasp and height.
3. Transport reward: encourage movement toward the target.
4. Release reward: encourage gripper opening near the target.

The fourth stage was initially described as release-and-settle. A later rendered rollout and final-state audit showed that this description was incorrect: the target was sampled in free space, so the released cube could pass near it before falling away.

## Audited v4 Results

Evaluation covered 60 episodes across 3 seeds after 1,638,400 final-stage transitions.

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

## Evaluation Audit

The original evaluator labeled an episode as stable placement when the gripper was open and the cube remained within 8 cm of the target for at least 10 consecutive simulation steps at any point in the episode. It did not require the cube to remain near the target at the end.

Every evaluated episode failed the existing final-distance check. A portfolio render reproduced the failure mode: the policy lifted and transported the cube, opened near the floating target, and then allowed the cube to fall away.

## Environment Mismatch

The environment sampled targets approximately 0.25-0.42 m above the ground without a platform or receptacle. A free object cannot settle at an unsupported spatial target under gravity. A true placement task must supply a support surface or container and evaluate release, support contact, low object velocity, and final-state retention.

## Main Lesson

Reward shaping successfully changed behavior from dragging to lifting, holding, and target-directed transport. It did not solve stable placement. The most important result was the evaluation audit: reward and intermediate success signals must be checked against final physical behavior and rendered rollouts.
