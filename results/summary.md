# Results

## Published designs, same constraints

| Case | Emitters | Steering | Published (dB) | This work (dB) | Improvement (dB) | Mean-level floor (dB) | Independent scan (dB) | Beam FWHM | Search |
|---|---|---|---|---|---|---|---|---|---|
| zang2025-32 | 32 | ±60° | -9.12 | **-11.48** | 2.36 | -14.73 | -11.48 | 0.754° | 15 min, 1109 local runs |
| zang2025-64 | 64 | ±60° | -11.75 | **-14.35** | 2.60 | -17.81 | -14.35 | 0.387° | 30 min, 1086 local runs |
| ma2024-128 | 128 | ±47° | -11.90 | **-16.22** | 4.32 | -20.81 | -16.22 | 0.089° | 45 min, 91 local runs |
| zhuang2018-128 | 128 | ±82° | -14.34 | **-16.53** | 2.19 | -21.24 | -16.53 | 0.066° | 90 min, 83 local runs |
| komljenovic2017-192 | 192 | ±90° | -13.50 | **-19.00** | 5.50 | -23.16 | -19.00 | 0.082° | 120 min, 71 local runs |

## Same metric, different optimisers

| Case | Method | Runs | Best (dB) | Median (dB) |
|---|---|---|---|---|
| zang2025-32 | Particle swarm | 3 | -9.89 | -9.65 |
| zang2025-32 | Genetic algorithm | 3 | -9.98 | -9.84 |
| zang2025-32 | Random search | 80,000 layouts | -8.76 | -6.11 (all) |
| zang2025-32 | **This work** | 1 | **-11.48** | |
| zang2025-64 | Particle swarm | 3 | -11.88 | -11.58 |
| zang2025-64 | Genetic algorithm | 3 | -12.16 | -11.60 |
| zang2025-64 | Random search | 80,000 layouts | -11.02 | -8.49 (all) |
| zang2025-64 | **This work** | 1 | **-14.35** | |

## Fewer channels for the same sidelobe level

* Komljenovic 2017: 80 emitters reach the published -13.5 dB of the 192-emitter design
* Zhuang 2018: 96 emitters reach the published -14.34 dB of the 128-emitter design

## Tolerance

Worst-case PSL over the steering range (dB). Displacement: every emitter moved by a uniform random amount up to the stated value, mean of 200 trials. Phase: Gaussian residual error per emitter.

| Design | Layout | Nominal | ±50 nm | ±100 nm | 10° phase error | Worst over 1500-1600 nm |
|---|---|---|---|---|---|---|
| komljenovic2017-192 | this work | -19.00 | -17.35 | -16.15 (95 %: -15.35) | -17.10 | -14.50 |
| ma2024-128 | this work | -16.22 | -15.13 | -14.23 (95 %: -13.61) | -14.49 | -15.45 |
| zang2025-32 | this work | -11.48 | -10.26 | -9.28 (95 %: -8.28) | -9.82 | -11.37 |
| zang2025-32 | particle swarm | -9.89 | -9.18 | -8.51 (95 %: -7.73) | | |
| zang2025-32 | genetic algorithm | -9.98 | -9.28 | -8.60 (95 %: -7.67) | | |
| zang2025-64 | this work | -14.35 | -13.00 | -11.96 (95 %: -11.18) | -12.45 | -12.01 |
| zang2025-64 | particle swarm | -11.88 | -11.30 | -10.75 (95 %: -9.90) | | |
| zang2025-64 | genetic algorithm | -12.16 | -11.67 | -11.04 (95 %: -10.32) | | |
| zhuang2018-128 | this work | -16.53 | -15.10 | -14.02 (95 %: -13.33) | -14.82 | -15.73 |

## Designing over a wavelength band

| Case | 1550 nm design, at 1550 nm | 1550 nm design, worst over band | Band design, at 1550 nm | Band design, worst over band |
|---|---|---|---|---|
| zang2025-32 | -11.48 | -11.37 | -11.23 | -11.23 |
| zang2025-64 | -14.35 | -12.01 | -14.24 | -14.23 |

## Planar array

| Layout | Worst-case PSL over the 40° cone (dB) |
|---|---|
| Published, uniform amplitudes | -12.7 |
| Published, amplitudes and phases optimised per beam | -17.0 |
| Random Poisson-disc layouts (best of 50) | -13.92 |
| **This work, uniform amplitudes** | **-19.07** |

Direct steering check over 37 beam directions: worst -19.07 dB. Mean-level floor -26.97 dB.
