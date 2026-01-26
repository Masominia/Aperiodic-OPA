"""Minimax layout synthesis: sequential linear programming inside an iterated local search.

**Local step.** At a layout ``x`` the sidelobe maxima within a factor ``frac``
of the worst are located (all wavelengths, all offsets ``v``), and each is
linearised in the positions. At a maximum the derivative in ``v`` vanishes,
so to first order the peak value moves only through its explicit dependence
on ``x``. The linear program

    minimise t   subject to   f_k + g_k . dx <= t        (every tracked peak)
                              gap_min <= gaps(x + dx) <= gap_max
                              aperture bounds,  |dx_n| <= rho

gives the step. It is accepted if the true worst case, re-evaluated over the
whole steering range, goes down; the trust radius ``rho`` grows after a
success and shrinks after a failure. This is the exchange (Remez-type)
approach to minimax design, here applied to element positions.

**Global search.** Local minima are plentiful. Multistart from random
feasible layouts is followed by an iterated local search that perturbs a few
gaps of the incumbent and re-optimises, keeping the result if it is better.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .farfield import Evaluator
from .spec import Spec, positions, project_gaps, random_layout


@dataclass
class Result:
    x: np.ndarray
    worst_db: float
    history: list[tuple[float, float]] = field(default_factory=list)
    local_runs: int = 0


def _difference_matrix(n: int) -> np.ndarray:
    d = np.zeros((n - 1, n))
    d[np.arange(n - 1), np.arange(n - 1)] = -1.0
    d[np.arange(n - 1), np.arange(1, n)] = 1.0
    return d


def local_minimax(
    ev: Evaluator,
    x: np.ndarray,
    rho: float = 0.3,
    frac: float = 0.1,
    max_iter: int = 600,
    rho_min: float = 1e-6,
    rho_max: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Sequential linear programming from ``x``; returns the layout and its worst level (linear)."""
    spec = ev.spec
    n = spec.n
    x = np.asarray(x, dtype=float) - x[0]
    best = ev.worst(x)
    dmat = _difference_matrix(n)
    gap_rows = np.hstack([dmat, np.zeros((n - 1, 1))])
    span_row = np.zeros((1, n + 1))
    span_row[0, n - 1] = 1.0
    c = np.zeros(n + 1)
    c[-1] = 1.0
    gmax = spec.gap_max if np.isfinite(spec.gap_max) else 1e9
    for _ in range(max_iter):
        pk = ev.peaks(x, frac)
        f, g = ev.value_and_grad(x, pk)
        gaps = dmat @ x
        a_ub = np.vstack([np.hstack([g, -np.ones((len(f), 1))]), gap_rows, -gap_rows, span_row, -span_row])
        b_ub = np.concatenate(
            [-f, gmax - gaps, gaps - spec.gap_min, [spec.aperture_max - x[-1]], [x[-1] - spec.aperture_min]]
        )
        bounds = [(0.0, 0.0)] + [(-rho, rho)] * (n - 1) + [(None, None)]
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            rho *= 0.5
            if rho < rho_min:
                break
            continue
        trial = x + res.x[:n]
        value = ev.worst(trial)
        if value < best:
            x, best = trial, value
            rho = min(rho * 1.6, rho_max)
        else:
            rho *= 0.4
        if rho < rho_min:
            break
    return x, best


def perturb(spec: Spec, x: np.ndarray, rng: np.random.Generator, n_gaps: int, scale: float) -> np.ndarray:
    """Move ``n_gaps`` random gaps by Gaussian amounts of ``scale`` (µm), then restore feasibility."""
    g = np.diff(x).copy()
    idx = rng.choice(len(g), min(n_gaps, len(g)), replace=False)
    g[idx] += rng.normal(0.0, scale, len(idx))
    total = float(np.clip(g.sum(), spec.aperture_min, spec.aperture_max))
    return positions(project_gaps(g, total, spec.gap_min, spec.gap_max))


def search(
    spec: Spec,
    seconds: float = 600.0,
    starts: int = 10,
    seed: int = 0,
    initial: list[np.ndarray] | None = None,
    log=None,
    tolerance: float = 0.0,
    scenarios: int = 8,
) -> Result:
    """Multistart, then iterated local search on the best layout until ``seconds`` have passed.

    With ``tolerance > 0`` (µm) the layout is judged together with ``scenarios``
    copies whose emitters are displaced by fixed random amounts of up to
    ``tolerance``, so the result keeps its sidelobe level under such errors.
    """
    rng = np.random.default_rng(seed)
    offsets = None
    if tolerance > 0:
        draws = rng.uniform(-tolerance, tolerance, (scenarios, spec.n))
        offsets = np.vstack([np.zeros(spec.n), draws])
    ev = Evaluator(spec, offsets=offsets)
    t0 = time.perf_counter()
    candidates = list(initial or [])
    candidates += [random_layout(spec, rng) for _ in range(max(0, starts - len(candidates)))]
    best_x, best = None, np.inf
    history = []
    runs = 0
    for x0 in candidates:
        x, value = local_minimax(ev, x0)
        runs += 1
        if value < best:
            best_x, best = x, value
            history.append((time.perf_counter() - t0, 10 * np.log10(best)))
            if log:
                log(f"start {runs}: {10 * np.log10(best):.3f} dB")
        if time.perf_counter() - t0 > seconds:
            break
    cur_x, cur = best_x, best
    while time.perf_counter() - t0 < seconds:
        k = int(rng.integers(2, max(3, spec.n // 6)))
        scale = float(rng.uniform(0.05, 0.5)) * spec.gap_min
        x, value = local_minimax(ev, perturb(spec, cur_x, rng, k, scale))
        runs += 1
        if value < cur:
            cur_x, cur = x, value
            if cur < best:
                best_x, best = cur_x, cur
                history.append((time.perf_counter() - t0, 10 * np.log10(best)))
                if log:
                    log(f"{time.perf_counter() - t0:7.0f} s  run {runs}: {10 * np.log10(best):.3f} dB")
    best_x, best = polish(ev, best_x)
    return Result(best_x, 10 * np.log10(best), history, runs)


def polish(ev: Evaluator, x: np.ndarray) -> tuple[np.ndarray, float]:
    """A last local run tracking every sidelobe within 20 dB of the worst, scored with refined peaks."""
    x, _ = local_minimax(ev, x, rho=0.05, frac=0.01, max_iter=200, rho_min=1e-8)
    return x, ev.worst(x, precise=True)
