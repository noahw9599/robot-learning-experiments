# Robot Learning Experiments

Reinforcement-learning experiments for humanoid locomotion and robotic manipulation in MuJoCo Playground.

This repository presents two controlled studies built with MuJoCo/MJX, JAX, and Brax PPO:

- Unitree G1 locomotion baseline reproduction and disturbance evaluation
- Franka Panda pick-and-place reward curriculum and multi-seed evaluation

The emphasis is experimental discipline: reproduce a strong baseline, isolate one change at a time, measure behavior across seeds, document failure modes, and keep claims proportional to the evidence.

## Results at a Glance

| Study | Training | Evaluation result |
|---|---:|---|
| Unitree G1 locomotion | 202,342,400 transitions | Reward improved from -5.801 to 15.578; peak 16.157 |
| G1 disturbance robustness | Existing baseline checkpoint | 5/5 seeds completed the 250-step push horizon; 0 early terminations |
| Panda release-and-settle curriculum | 1,638,400 final-stage transitions | 85.0% stable placement across 60 episodes and 3 seeds |

The G1 result reproduces the official `G1JoystickFlatTerrain` task. The Panda result comes from a staged curriculum that progressed from lifting to holding, transport, release, and stable placement.

## Learning System

```mermaid
flowchart LR
    A["MuJoCo / MJX environment"] --> B["Actor observations"]
    B --> C["Brax PPO policy"]
    C --> D["Joint position targets"]
    D --> A
    A --> E["Reward and termination"]
    E --> F["Parallel PPO updates"]
    F --> C
    A --> G["Privileged simulator state"]
    G --> H["Value function"]
    H --> F
```

The G1 policy uses an asymmetric actor-critic design: the actor receives a 103-element deployable state vector, while the critic receives additional privileged simulation information during training.

## Key Engineering Work

- Reproduced the official G1 baseline from pinned MuJoCo Playground source.
- Built deterministic, per-step diagnostics for support geometry, torso orientation, velocity, contacts, action authority, and actuator force.
- Evaluated G1 robustness with randomized planar pushes across independent seeds.
- Designed reversible reward patches for Panda lifting, transport, release, and settling.
- Built an instrumented evaluator for lift, hold, target approach, release, final placement, and closest target distance.
- Used TensorBoard, checkpoint manifests, headless EGL rendering, and bounded cloud runs with shutdown safeguards.

## What the Experiments Showed

- PPO optimizes expected return; it does not preserve or revert to one previously successful action sequence.
- Higher scalar reward did not consistently imply better balance or disturbance recovery.
- Poorly specified rewards produced recognizable shortcuts, including dragging and scooting.
- Behavior-level metrics and multi-seed evaluation were more informative than reward curves alone.
- Changing the actor observation size invalidated checkpoint normalization statistics and required training from scratch.
- Compilation time and cloud cost are part of the practical experiment budget.

## Repository Structure

```text
docs/experiments/    Experiment protocols, results, and limitations
results/             Concise, resume-safe metrics
scripts/diagnostics/ Per-step G1 and Panda evaluators
scripts/patches/     Reversible reward and recovery modifications
scripts/             Environment setup and smoke tests
```

## Reproducing the Environment

The setup script checks out the exact MuJoCo Playground commit used for the official baseline:

```bash
bash scripts/setup_mujoco_playground.sh
python scripts/smoke_test_mujoco_playground.py
```

Large checkpoints, raw run directories, credentials, and cloud-specific identifiers are intentionally excluded. The scripts assume Linux, Python 3.12, `uv`, and accelerator-compatible JAX for full training.

## Experiment Reports

- [G1 official baseline](docs/experiments/g1-official-baseline.md)
- [G1 push robustness](docs/experiments/g1-push-robustness.md)
- [Panda manipulation curriculum](docs/experiments/panda-manipulation-curriculum.md)
- [G1 observation augmentation attempt](docs/experiments/g1-observation-augmentation.md)
- [Verified results summary](results/summary.md)

## Limitations

The G1 push result applies only to the stated five-seed, 250-step protocol. It is not a universal stability guarantee.

The six-feature G1 observation experiment did not reach a valid training update before its compilation budget expired. It is documented as a bounded negative result, not as evidence that the proposed features help or hurt policy learning.

## Author

Noah Wilson  
Mechanical Engineering, University of Florida
