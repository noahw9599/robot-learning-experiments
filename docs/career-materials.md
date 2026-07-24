# Career Materials

This page translates the verified project evidence into resume, interview, and
LinkedIn language. All quantitative claims match the saved evaluations in the
[verified results summary](../results/summary.md).

## Recommended Resume Entry

**Robot Learning Experiments | Independent Project**

*MuJoCo Playground, MJX, JAX, Brax PPO, AWS EC2, NVIDIA A10G*

- Reproduced a Unitree G1 humanoid locomotion baseline with JAX/MJX and Brax PPO,
  training for 202.3M transitions and improving evaluation reward from -5.801 to
  15.578.
- Built deterministic balance and control diagnostics, then evaluated randomized
  planar pushes across five seeds and 1,250 simulation steps; all five seeded
  rollouts completed the test horizon with zero early terminations.
- Designed a staged Panda manipulation reward curriculum that achieved 100% lift
  and hold success with 96.7% target-region approach over 60 episodes; audited a
  misleading transient metric and identified 0% true final placement caused by
  unsupported free-space targets.

## Two-Bullet Resume Version

- Reproduced and evaluated a Unitree G1 locomotion policy in MuJoCo Playground,
  training for 202.3M transitions and achieving 5/5 full-horizon randomized push
  evaluations across 1,250 simulation steps.
- Developed staged Panda reward shaping and multi-seed behavior metrics; achieved
  100% lift/hold and 96.7% target approach over 60 episodes, then corrected an
  overstated placement metric through final-state and visual validation.

## Hardware-Focused Resume Version

- Investigated simulated humanoid balance using center-of-mass/support geometry,
  torso orientation, contact forces, joint targets, actuator force, and
  randomized disturbances in MuJoCo.
- Built a reproducible JAX/MJX evaluation pipeline for Unitree G1 locomotion and
  Franka Panda manipulation, connecting mechanical behavior to policy actions,
  reward design, contacts, and termination logic.
- Diagnosed nonphysical manipulation success caused by an unsupported target and
  redesigned evaluation criteria around final object position and physically
  feasible task outcomes.

## One-Line Project Summary

Built and audited a reproducible robot-learning pipeline for Unitree G1
locomotion and Franka Panda manipulation using MuJoCo Playground, JAX/MJX, Brax
PPO, and AWS GPUs.

## 30-Second Interview Answer

I built an independent robot-learning study in MuJoCo Playground using JAX and
Brax PPO. I first reproduced the official Unitree G1 locomotion baseline for
202 million transitions, then built diagnostics for balance, contacts, actuator
authority, and randomized push evaluation. I also developed a staged Panda
manipulation curriculum that learned lifting, holding, and transport. The most
important result was auditing my own placement metric: a reported transient
success did not match the final physical state, so I corrected the claim and the
evaluator. That taught me to treat reward, metrics, and rendered behavior as
separate sources of evidence.

## 90-Second Interview Answer

The project started with a question: how do modern robot-learning systems turn a
simulated mechanical system into a useful policy, and how do we know the learned
behavior is real?

I used MuJoCo Playground with the MJX backend and Brax PPO. On the humanoid side,
I reproduced the official Unitree G1 flat-terrain baseline for 202.3 million
transitions. I instrumented center of mass relative to support, torso
orientation, contacts, velocities, actions, and actuator forces. Under a defined
five-seed randomized push protocol, the policy completed all 1,250 evaluation
steps without early termination.

On the manipulation side, the first Panda policy dragged the cube, which was a
reward shortcut. I introduced reward stages for lifting, holding, transport, and
near-target gripper opening. Across 60 episodes, the final policy achieved 100%
lift and hold and 96.7% target approach. However, visual inspection showed that
the cube fell away after release. I audited the evaluator and found 0% final
placement because the target was floating in unsupported space. I corrected the
public result and separated transient from final-state metrics.

The project taught me that successful robot learning is as much about mechanics,
task design, measurement, and reproducibility as it is about optimizing a neural
network.

## Likely Interview Follow-Ups

**Why PPO?**

It is the baseline used by MuJoCo Playground for these tasks, supports massively
parallel simulation, and let the project focus on reproduction and measurement
before algorithm changes.

**Why did reward sometimes decrease late in training?**

PPO updates a stochastic policy to improve expected return from sampled batches.
It does not save and replay the single highest-reward action sequence, so updates
can trade off behaviors or temporarily reduce evaluation return.

**Why not continue training until placement worked?**

The failure was not only insufficient optimization. The target represented an
unsupported point in space, so stable released placement was physically
ill-defined. More compute would optimize the wrong task.

**What would you change next?**

Use a supported target such as a tray or platform, define success before
training, and require release, support contact, low object velocity, and
final-state retention across randomized seeds.

**Was this imitation learning?**

No. The published results use PPO and reward shaping. I intentionally avoid
claiming imitation learning or architectural novelty that was not implemented.

## LinkedIn Post 1: Main Project

**Attach:** `media/g1-official-baseline.mp4`

I trained a Unitree G1 humanoid policy to walk in MuJoCo, but the most valuable
result of the project was learning when not to trust a reward curve.

Over this independent robot-learning study, I:

- reproduced the official G1 locomotion baseline with JAX/MJX and Brax PPO for
  202.3M transitions
- built diagnostics for center of mass, support geometry, torso orientation,
  contacts, actions, and actuator force
- evaluated randomized pushes across five seeds and 1,250 simulation steps, with
  all five rollouts completing the test horizon
- developed a staged Franka Panda manipulation curriculum and evaluated it over
  60 episodes

The Panda policy reached 100% lift and hold success and 96.7% target approach.
Then a rendered rollout exposed a problem: my "stable placement" metric was only
detecting a brief near-target release. The cube finished at the target in 0 of 60
episodes because the target was unsupported in free space.

I corrected the result and rebuilt the evaluator around final-state behavior.
That failure became the clearest lesson of the project: in robot learning, reward,
metrics, simulation physics, and visible behavior all need to agree.

Repository: https://github.com/noahw9599/robot-learning-experiments

#Robotics #ReinforcementLearning #MuJoCo #HumanoidRobotics

## LinkedIn Post 2: Manipulation Follow-Up

**Attach:** `media/panda-transport-v3.mp4`

A robot can optimize exactly what you ask for and still miss what you meant.

My first Franka Panda policy learned to drag a cube instead of lifting it. I used
a staged PPO reward curriculum to change the behavior from contact, to lift and
hold, to target-directed transport.

Across 60 final-checkpoint evaluations:

- lift success: 100%
- hold success: 100%
- target-region approach: 96.7%
- final target placement: 0%

The last number mattered most. A transient metric initially looked successful,
but visual and final-state validation showed that the unsupported target made
stable released placement physically impossible.

The lesson I am carrying forward is to define success in physical terms before
spending compute: where should the object finish, what supports it, how slowly
should it move, and how long must the state persist?

Full experiment and evaluator:
https://github.com/noahw9599/robot-learning-experiments

#RobotLearning #Manipulation #ReinforcementLearning #MuJoCo

## Video Selection

| Video | Use | Reason |
|---|---|---|
| `media/g1-official-baseline.mp4` | Main LinkedIn post and featured GitHub result | The 20-second clip is the clearest positive result and most relevant to humanoid robotics roles. |
| `media/panda-transport-v3.mp4` | Separate manipulation follow-up | The 3-second clip clearly shows grasp, lift, hold, and transport without implying final placement. |
| `media/g1-early-forward-fall.mp4` | GitHub before-and-after evidence | Useful as a diagnostic comparison, but too failure-focused for the primary post. |
| `media/panda-baseline-dragging.mp4` | GitHub reward-hacking evidence | Demonstrates the curriculum's starting failure mode and supports the technical story. |

[LinkedIn currently allows one video per post](https://www.linkedin.com/help/linkedin/answer/a554002).
Use the G1 clip for the launch post and the Panda clip for a later technical
follow-up. Enable automatic captions, choose a clear thumbnail, and keep
important visual content away from the frame edges.

## Claims to Avoid

- Do not say the Panda policy achieved stable placement or solved pick-and-place.
- Do not call the project imitation learning.
- Do not claim a novel PPO architecture or a new algorithm.
- Do not generalize the five-seed push result beyond its 250-step protocol.
- Do not describe a selected rollout video as a success rate.

The strongest story is not that every experiment worked. It is that the project
used baselines, controlled changes, multi-seed evaluation, physical reasoning,
and public correction when a metric did not match reality.
