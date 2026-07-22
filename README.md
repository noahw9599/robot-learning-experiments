# Robot Learning Experiments

A reproducible robot-learning project focused on understanding reinforcement learning, imitation learning, simulation, and evaluation for humanoid robotics.

This repository documents **Project Jiraiya**, a MuJoCo Playground study built around two controlled tracks:

- Unitree G1 humanoid locomotion with JAX/MJX and Brax PPO
- Franka Panda manipulation with staged reward design

The goal is not to claim a novel algorithm. The goal is to reproduce strong baselines, understand why they work, measure failure modes, and make controlled changes.

## Verified Results

### Unitree G1 locomotion

- Environment: `G1JoystickFlatTerrain`
- MuJoCo Playground commit: `87a4bf98f1806adefd240d72dd53a2c3ceeb2f0d`
- Training: `202,342,400` transitions
- Reward: `-5.801 -> 15.578`
- Peak reward: `16.157`
- Five-seed push evaluation: `5/5` full-horizon survival
- Push evaluation: `1,250` total simulation steps, `0` early terminations

### Franka Panda manipulation

A staged curriculum changed the task objective from lift, to transport, to release and settle.

- Final v4 training: `1,638,400` actual transitions
- Robustness evaluation: `60` episodes across `3` seeds
- Lift: `100%`
- Hold: `100%`
- Target-region approach: `96.7% +/- 5.8 percentage points`
- Stable placement: `85.0% +/- 5.0 percentage points`

## What I Learned

- PPO optimizes expected long-term return; it does not undo a bad action like a search algorithm.
- A higher scalar reward does not necessarily mean better locomotion or disturbance recovery.
- Reward shaping can create exploitable behaviors such as dragging, scooting, or over-conservative motion.
- Asymmetric actor-critic training gives the critic privileged simulator state while keeping the actor closer to deployable observations.
- Checkpoints are architecture-dependent: changing observation size requires training from scratch.
- Robust evaluation requires multiple seeds and behavior-level metrics, not only TensorBoard reward curves.
- Cloud compute must be treated as an experimental budget with bounded tests and automatic shutdown safeguards.

## Reproducibility

The project uses MuJoCo Playground, JAX/MJX, Brax PPO, TensorBoard, and AWS EC2 GPU compute. Setup and learning notes are in `docs/`; reusable setup and training helpers are in `scripts/`.

The original cloud run used headless EGL rendering when visual artifacts were required. Large checkpoints, raw run directories, credentials, and cloud-specific private identifiers are intentionally excluded from this public repository.

## Research Limitations

The G1 push result is valid for the stated five-second, five-seed disturbance protocol; it is not a universal stability guarantee.

A six-feature G1 observation augmentation was implemented as a controlled hypothesis, but first-pass XLA/CUDA compilation exceeded the cost-controlled smoke-test budget before a valid training update. No performance claim is made for that experiment.

## Repository Map

- `docs/`: environment notes, setup walkthroughs, experiment reports, and cloud reproducibility notes
- `scripts/`: setup, diagnostics, and bounded training helpers
- `results/`: concise tables and links to reproducible metrics
- `LICENSE`: project licensing

## Author

Noah Wilson  
Conquest Robotics
