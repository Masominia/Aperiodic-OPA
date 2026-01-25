"""Reference layouts and the population-based optimisers used in the literature, on the same metric.

Running particle swarm and genetic algorithms on exactly the objective and
constraints used here separates two questions: whether a better number comes
from a different way of measuring sidelobes, or from a better optimiser.
Both optimisers act on the gap vector, are repaired to feasibility after
every move, and score candidates by the worst case over the whole steering
range (a stronger objective than the single-angle fitness most papers use).
"""

from __future__ import annotations

import time

import numpy as np

from .farfield import Evaluator
from .spec import Spec, positions, project_gaps, random_layout


def _repair(spec: Spec, g: np.ndarray) -> np.ndarray:
    hi = spec.gap_max if np.isfinite(spec.gap_max) else spec.aperture_max
    g = np.clip(g, spec.gap_min, hi)
    total = float(np.clip(g.sum(), spec.aperture_min, spec.aperture_max))
    return project_gaps(g, total, spec.gap_min, spec.gap_max)


def _score(ev: Evaluator, g: np.ndarray) -> float:
    return ev.worst(positions(g))


def random_search(spec: Spec, evaluations: int, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Best of ``evaluations`` random feasible layouts; also returns every score (dB)."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(spec)
    best_x, best, scores = None, np.inf, []
    for _ in range(evaluations):
        x = random_layout(spec, rng)
        value = ev.worst(x)
        scores.append(value)
        if value < best:
            best_x, best = x, value
    return best_x, 10 * np.log10(best), 10 * np.log10(np.array(scores))


def particle_swarm(
    spec: Spec,
    particles: int = 80,
    iterations: int = 1000,
    seed: int = 0,
    w_start: float = 0.9,
    w_end: float = 0.3,
    c1: float = 2.0,
    c2: float = 2.0,
    seconds: float | None = None,
) -> tuple[np.ndarray, float, list[tuple[float, float]]]:
    """Particle swarm with linearly decreasing inertia, as in the published OPA studies."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(spec)
    hi = spec.gap_max if np.isfinite(spec.gap_max) else 2 * spec.aperture_max / (spec.n - 1)
    pos = np.array([np.diff(random_layout(spec, rng)) for _ in range(particles)])
    vel = rng.uniform(-1, 1, pos.shape) * 0.1 * (hi - spec.gap_min)
    vmax = 0.2 * (hi - spec.gap_min)
    fit = np.array([_score(ev, p) for p in pos])
    pbest, pfit = pos.copy(), fit.copy()
    gi = int(np.argmin(pfit))
    gbest, gfit = pbest[gi].copy(), pfit[gi]
    t0 = time.perf_counter()
    trace = [(0.0, 10 * np.log10(gfit))]
    for it in range(iterations):
        w = w_start - (w_start - w_end) * it / max(1, iterations - 1)
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = np.clip(w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos), -vmax, vmax)
        pos = np.array([_repair(spec, p) for p in pos + vel])
        fit = np.array([_score(ev, p) for p in pos])
        better = fit < pfit
        pbest[better], pfit[better] = pos[better], fit[better]
        gi = int(np.argmin(pfit))
        if pfit[gi] < gfit:
            gbest, gfit = pbest[gi].copy(), pfit[gi]
        trace.append((time.perf_counter() - t0, 10 * np.log10(gfit)))
        if seconds and time.perf_counter() - t0 > seconds:
            break
    x = positions(gbest)
    return x, ev.worst_db(x), trace


def genetic(
    spec: Spec,
    population: int = 50,
    generations: int = 1000,
    seed: int = 0,
    tournament: int = 3,
    crossover: float = 0.9,
    mutation: float = 0.05,
    elite: int = 2,
    seconds: float | None = None,
) -> tuple[np.ndarray, float, list[tuple[float, float]]]:
    """Real-coded genetic algorithm: tournament selection, blend crossover, Gaussian mutation, elitism."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(spec)
    hi = spec.gap_max if np.isfinite(spec.gap_max) else 2 * spec.aperture_max / (spec.n - 1)
    pop = np.array([np.diff(random_layout(spec, rng)) for _ in range(population)])
    fit = np.array([_score(ev, p) for p in pop])
    t0 = time.perf_counter()
    trace = [(0.0, 10 * np.log10(fit.min()))]
    for _ in range(generations):
        order = np.argsort(fit)
        children = [pop[i].copy() for i in order[:elite]]
        while len(children) < population:
            a, b = (pop[min(rng.choice(population, tournament, replace=False), key=lambda i: fit[i])] for _ in "ab")
            if rng.random() < crossover:
                alpha = rng.uniform(-0.25, 1.25, len(a))
                child = alpha * a + (1 - alpha) * b
            else:
                child = a.copy()
            mask = rng.random(len(child)) < mutation
            child[mask] += rng.normal(0, 0.1 * (hi - spec.gap_min), mask.sum())
            children.append(_repair(spec, child))
        pop = np.array(children)
        fit = np.array([_score(ev, p) for p in pop])
        trace.append((time.perf_counter() - t0, 10 * np.log10(fit.min())))
        if seconds and time.perf_counter() - t0 > seconds:
            break
    x = positions(pop[int(np.argmin(fit))])
    return x, ev.worst_db(x), trace
