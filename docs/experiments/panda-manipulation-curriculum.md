# Panda Manipulation Curriculum

## Objective

Understand how reward design changes learned manipulation behavior.

## Curriculum

1. Lift reward: encourage the gripper to raise the cube.
2. Hold reward: encourage sustained grasp and height.
3. Transport reward: encourage movement toward the target.
4. Release-and-settle reward: encourage release over the target and stable placement.

## Final v4 Results

- Actual transitions: 1,638,400
- Robustness evaluation: 60 episodes across 3 seeds
- Lift: 100%
- Hold: 100%
- Target-region approach: 96.7% +/- 5.8 percentage points
- Release: 86.7% +/- 2.9 percentage points
- Stable placement: 85.0% +/- 5.0 percentage points
- Mean closest target distance: 3.72 +/- 0.45 cm

## Main Lesson

The baseline learned to drag the cube. Staged reward design changed the behavior toward lifting, carrying, releasing, and settling. The experiment showed why success metrics must measure the actual task objective rather than only total return.
