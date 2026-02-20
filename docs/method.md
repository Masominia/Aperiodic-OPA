# Method

## 1. The design problem

An optical phased array of `N` identical emitters at positions `x_n` steers its beam by phase
alone. Silicon waveguides cannot sit closer than about 1.3 to 3 wavelengths without coupling,
and a uniform array at that pitch puts grating lobes into the visible range as soon as the
beam moves. Aperiodic layouts break the grating lobes into a floor of sidelobes. The question
is how to place the emitters so that the **highest sidelobe seen at any steering angle in the
operating range** is as low as possible, subject to the fabrication rules:

* minimum (and sometimes maximum) gap between neighbouring emitters,
* a fixed or bounded aperture,
* uniform emitter amplitudes and phase-only steering, as in every published design compared here,
* the emitter's own radiation pattern `E(u)`.

The figure of merit is the **worst-case peak sidelobe level (PSL) over the steering range**,
in dB relative to the main beam. It is the number a LiDAR designer cares about: the false-echo
margin at the worst pointing direction.

## 2. The whole steering range is one problem

With direction cosines `u = sin(theta)`, phases that point the beam at `u0`, and equal
amplitudes, the far-field intensity is

```
I(u) = |E(u)|^2 |A(u - u0)|^2,        A(v) = sum_n exp(i k x_n v),   k = 2 pi / lambda.
```

The array factor depends only on the offset `v = u - u0` from the beam. A sidelobe of `A` at
offset `v` is visible at steering `u0` whenever `|u0 + v| <= 1`, with level

```
|E(u0 + v)|^2 |A(v)|^2 / ( |E(u0)|^2 N^2 ).
```

So the worst case over every steering angle in `[-s, s]` (with `s = sin(theta_max)`) is

```
PSL_worst = max_{v_null <= |v| <= 1 + s}  w(v) |A(v)|^2 / N^2,
w(v)      = max_{|u0| <= s, |u0 + v| <= 1}  |E(u0 + v)|^2 / |E(u0)|^2.
```

This is an equality, not a bound: every `(u0, u)` pair is covered exactly once. The steering
dimension disappears, and a design judged over a continuous steering range costs no more to
optimise than a broadside one. Published studies instead sample a handful of angles (0, 10, ...,
60 degrees) and miss the sidelobes that peak in between.

**Closed form for the weight.** `|A(-v)| = |A(v)|` for real positions, so only `v >= 0` is
needed. If `E` is even and log-concave on `[-1, 1]` (a slit `sin(a)/a` inside its main lobe, a
Gaussian, `cos^q`), then `g = log E` has a non-increasing derivative and

```
d/du0 [ g(u0 + v) - g(u0) ] = g'(u0 + v) - g'(u0) <= 0   for v >= 0,
```

so the ratio is largest at the smallest admissible `u0`, the far edge `u0 = -s`:

```
w(v) = |E(v - s)|^2 / |E(s)|^2,        0 <= v <= 1 + s.
```

The worst view of any sidelobe is with the beam pushed to the opposite edge of the range, where
the main beam is dimmed by the element pattern and the sidelobe sits nearer broadside. For
isotropic emitters `w = 1` and the problem is the classic one over `|v| <= 1 + s`. The tests
check the closed form against a brute-force maximum over steering angles for every pattern, and
check the reduced metric against an independent evaluator that steers the beam angle by angle
(`aopa.farfield.steering_scan`); they agree to 0.01 dB.

**Planar arrays.** With positions in the plane and a steering cone `|u0| <= s`, the same
argument with `u0` anti-parallel to `v` gives `w(|v|)` with the one-dimensional weight, over the
disc `|v| <= 1 + s` (half of it, by symmetry).

## 3. Optimiser

### 3.1 Local step: sequential linear programming

`PSL_worst` is a maximum over a continuum of smooth functions of the positions. At a sidelobe
maximum the derivative in `v` vanishes, so to first order a peak moves with the positions only
through its explicit dependence on them:

```
d/dx_n [ w |A(v)|^2 ] = -2 k v w Im( conj(A(v)) exp(i k x_n v) ).
```

At each iterate every sidelobe maximum within 10 dB of the worst (all wavelengths) is located
on a grid of eight samples per lobe width, refined by parabolic interpolation, and linearised.
The edge of the visible range, `v = 1 + s` (the rim `|v| = 1 + s` in 2-D), is tracked exactly as
well: when the map is still rising there, the edge itself is the maximum, and an optimiser that
only watches grid maxima will happily push a lobe just past the edge while its flank stays high.
The linear program

```
minimise t   s.t.  f_j + g_j . dx <= t                 (every tracked peak)
                   gap_min <= x_{n+1} + dx_{n+1} - x_n - dx_n <= gap_max
                   aperture bounds,   x_0 fixed,   |dx_n| <= rho
```

is solved with HiGHS. The step is accepted only if the true worst case, re-evaluated over the
whole range, goes down; then the trust radius `rho` grows by 1.6, otherwise it shrinks by 0.4.
This is the exchange (Remez-type) approach to minimax design: it drives the active sidelobes
to equal height, which is what an optimal minimax layout looks like.

For planar arrays the minimum-distance constraints are linearised as
`|p_i - p_j| + u_ij . (dp_i - dp_j) >= d_min`. Distance is a convex function of the separation,
so its linearisation never overestimates it, and every LP step is exactly feasible.

### 3.2 Global search

Minimax layout problems have many local minima. The search runs the local step from ten
random feasible layouts and then an iterated local search: perturb a few gaps of the incumbent
by a fraction of the minimum gap, restore feasibility, re-optimise, keep the result if it is
better. The best layout gets a final polish that tracks every sidelobe within 20 dB of the worst
and scores it with peaks refined to machine precision.

Planar layouts with hundreds of emitters first go through a smooth stage: L-BFGS-B on the soft
maximum `(1/q) log sum_j f_j^q` of the whole sidelobe map, with `q` raised from 8 to 64 and a
quadratic penalty on pairs closer than `d_min`. The gradient of the soft maximum with respect to
every position is one type-2 non-uniform FFT per coordinate, so a full gradient costs about as
much as one far-field map. The linear-programming stage then finishes the layout.

## 4. Evaluation

* **Grid and refinement.** Maps are sampled at 8 points per lobe width (`lambda / L`) in one
  dimension, 4 in two, with non-uniform FFTs (`finufft`). Reported values come from peaks refined
  by nested zoom searches to well below 0.001 dB.
* **Main lobe.** The main lobe is the region around `v = 0` down to the first minima of `|A|`
  (in 2D, the pixels reached from the origin by never-increasing steps).
* **Independent check.** `steering_scan` recomputes the steered far field angle by angle,
  bounds the main lobe by the nearest minima around the actual beam, and normalises by the
  actual beam peak. Every stored design is checked against it at 1 degree spacing.
* **Floor.** The mean of the weighted sidelobe map over `[v_null, 1 + s]` is a hard lower bound
  on that layout's PSL: no peak can sit below the mean. The gap between the two shows how close
  a design is to having equal sidelobes everywhere. By Parseval's theorem the integrated
  sidelobe power of any layout of `N` unit emitters is close to `N` per unit of `v`, so the mean
  floor sits near `-10 log10(N)` dB plus a correction for the region and the element pattern;
  sidelobe levels below that are out of reach for any layout with uniform amplitudes.

## 5. Benchmark protocol

Each published design is restated as a specification with the constraints its authors used:
element count, gap bounds, aperture, steering range, element pattern and wavelength
(`aopa/cases.py`). Where a paper leaves something open, the choice is recorded in the case as
an assumption. The comparison is against the authors' **reported** worst-case level over their
steering range, from their array-factor model. Published layouts are not available, so they
cannot be re-scored here. Two things make the comparison conservative for this work:

* the metric here is the worst case over the **continuous** steering range, refined to the true
  peaks, while published figures come from a few sampled angles;
* the particle swarm and genetic algorithms used in those papers are re-implemented in
  `aopa/baselines.py` and run on exactly this metric, so the difference between the optimisers
  is also measured on equal terms.

## 6. Robustness

Emitter displacement and residual phase errors after calibration do not depend on the pointing
direction, so the reduction of section 2 still holds with complex emitter weights; the array
factor is no longer even, so both signs of `v` are scanned and the main lobe is normalised by its
actual, lowered peak (`aopa/robustness.py`). Wavelength changes rescale `k`; the optimiser can
judge a layout at several wavelengths at once and minimise the worst of them.

## 7. What did not help

* **Fixed displaced copies as extra scenarios.** Judging a layout together with eight copies whose
  emitters were displaced by fixed random amounts up to 100 nm (`search(..., tolerance=0.1)`) gave
  layouts that held their level for those copies, but under fresh random displacements they did no
  better than the nominal designs (mean −9.0 vs −9.3 dB for 32 emitters at ±100 nm). The option is
  kept in the code; it is not used for any result.
* **Tracking only grid maxima.** An early version linearised only the maxima found on the sampling
  grid. At ±90° steering the optimiser then pushed a lobe just past the edge of the visible range while
  its flank stayed high, and a layout that scored −19.9 dB on the grid was −17.5 dB when evaluated
  exactly. Tracking the edge of the range exactly (section 3.1) removed the problem; every stored result
  was produced or re-scored with it.
