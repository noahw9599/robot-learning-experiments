# G1 Official Baseline

## Objective

Reproduce the official MuJoCo Playground Unitree G1 locomotion baseline before changing the environment, reward, or network.

## Configuration

- Environment: `G1JoystickFlatTerrain`
- Backend: JAX/MJX
- Pinned MuJoCo Playground commit: `87a4bf98f1806adefd240d72dd53a2c3ceeb2f0d`
- Actor observation: 103 features
- Actor and critic hidden layers: 512, 256, 128
- Actual transitions: 202,342,400

## Results

- Initial reward: -5.801
- Final reward: 15.578
- Peak reward: 16.157
- Runtime: approximately 2 hours 32 minutes
- The final rollout showed the G1 walking out of the camera frame.

## Interpretation

This reproduction established a known-good reference policy. The scalar reward improved substantially, but the result was evaluated behaviorally rather than judged from reward alone.

## Lessons

- Checkpoints must be paired with the matching source and observation interface.
- Headless rendering on cloud machines requires EGL configuration.
- Reward curves are useful diagnostics, not complete measures of locomotion quality.
