# G1 Push Robustness Evaluation

## Question

Does the official G1 policy remain upright under randomized planar pushes?

## Protocol

- Official checkpoint at 202,342,400 transitions
- Five independent seeds
- 250 steps per seed
- 1,250 total simulation steps
- Built-in planar pushes
- Push interval: 5-10 seconds
- Push magnitude: 0.1-2.0
- Rendering disabled

## Results

| Metric | Result |
|---|---:|
| Full-horizon survival | 5/5 seeds |
| Early terminations | 0/5 |
| Maximum absolute torso-gravity deviation | 0.134 |
| Worst COM-ahead-of-support value | +0.115 m |
| Maximum absolute forward velocity | 0.701 m/s |

## Interpretation

The policy survived this defined five-second disturbance protocol across every seed. This is evidence of repeatable behavior, not a universal stability guarantee. Longer horizons, stronger pushes, and directional sweeps remain future work.
