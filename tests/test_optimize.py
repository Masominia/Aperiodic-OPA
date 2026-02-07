import numpy as np

from aopa import baselines
from aopa.farfield import Evaluator
from aopa.optimize import local_minimax, perturb, search
from aopa.patterns import Slit
from aopa.robustness import position_errors, worst_with_errors
from aopa.spec import Spec, project_gaps, random_layout

SMALL = Spec("small", n=16, gap_min=2.0, gap_max=5.0, aperture=50.0, steer_deg=45.0, pattern=Slit(0.5))
FREE = Spec("free", n=16, gap_min=2.0, gap_max=5.0, steer_deg=45.0)


def test_projection_keeps_bounds_and_total():
    g = project_gaps(np.random.default_rng(0).uniform(0, 9, 30), 100.0, 2.0, 5.0)
    assert np.all(g >= 2.0 - 1e-12) and np.all(g <= 5.0 + 1e-12) and np.isclose(g.sum(), 100.0)


def test_random_layouts_are_feasible():
    rng = np.random.default_rng(1)
    for spec in (SMALL, FREE):
        for _ in range(20):
            assert spec.feasible(random_layout(spec, rng))


def test_local_step_improves_and_stays_feasible():
    ev = Evaluator(SMALL)
    x0 = random_layout(SMALL, np.random.default_rng(2))
    x, value = local_minimax(ev, x0)
    assert SMALL.feasible(x)
    assert value < ev.worst(x0)
    assert np.isclose(10 * np.log10(value), ev.worst_db(x, precise=False))


def test_free_aperture_respects_bounds():
    ev = Evaluator(FREE)
    x, _ = local_minimax(ev, random_layout(FREE, np.random.default_rng(3)))
    assert FREE.feasible(x)


def test_perturbation_is_feasible():
    rng = np.random.default_rng(4)
    x = random_layout(SMALL, rng)
    assert SMALL.feasible(perturb(SMALL, x, rng, 5, 1.0))


def test_search_beats_random_search():
    res = search(SMALL, seconds=8, starts=3, seed=0)
    _, rand_best, _ = baselines.random_search(SMALL, evaluations=300, seed=0)
    assert SMALL.feasible(res.x)
    assert res.worst_db < rand_best - 1.0


def test_population_baselines_run():
    x, value, trace = baselines.particle_swarm(SMALL, particles=10, iterations=5)
    assert SMALL.feasible(x) and len(trace) == 6 and np.isfinite(value)
    x, value, trace = baselines.genetic(SMALL, population=10, generations=5)
    assert SMALL.feasible(x) and len(trace) == 6 and np.isfinite(value)


def test_error_model_reduces_to_ideal_case():
    x = random_layout(SMALL, np.random.default_rng(5))
    assert abs(worst_with_errors(SMALL, x) - Evaluator(SMALL).worst_db(x)) < 1e-3
    assert np.allclose(position_errors(SMALL, x, 0.0, trials=3), Evaluator(SMALL).worst_db(x), atol=1e-3)
