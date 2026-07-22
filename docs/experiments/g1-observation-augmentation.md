# G1 Observation Augmentation Attempt

## Hypothesis

Giving the actor explicit balance signals might help a newly trained policy recover from disturbances.

## Proposed Features

- Planar COM velocity X/Y
- Capture-point error X/Y
- Signed COM/support error X/Y

The reward, termination logic, and network architecture were held constant. The observation size changed from 103 to 109, so the official checkpoint could not be reused.

## Outcome

Two reversible implementations reached expensive first-pass XLA/CUDA compilation but did not produce a valid training update within the cost-controlled smoke-test budget. The planned 2M-transition run was stopped before training began.

## What This Taught Me

- Observation changes require training from scratch when normalized statistics and network inputs change.
- Compilation cost is part of the practical research budget.
- The experiment does not prove the features would help or hurt.
- A bounded negative result is more informative than spending credits without a valid comparison.
