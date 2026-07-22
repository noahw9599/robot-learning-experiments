# Project Jiraiya Results Summary

| Track | Key result |
|---|---|
| G1 official locomotion | 202M transitions; reward -5.801 to 15.578 |
| G1 push robustness | 5/5 seeds survived 250 steps |
| Panda manipulation | 85.0% stable placement across 60 episodes and 3 seeds |
| Observation augmentation | Bounded before training due to compilation cost |

## Resume-Safe Summary

Reproduced a Unitree G1 locomotion baseline in MuJoCo Playground using JAX/MJX and Brax PPO, evaluated it under randomized pushes, and built a staged Panda manipulation curriculum that achieved 85% stable placement across three seeds. Documented reward exploitation, checkpoint compatibility, evaluation design, and cloud-compute controls.
