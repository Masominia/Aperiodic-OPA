import numpy as np

from aopa.patterns import Gaussian
from aopa.planar import PlanarEvaluator, PlanarSpec, local_minimax, poisson_disc, smooth_descent, steering_check

SPEC = PlanarSpec("t", n=36, side=20.0, d_min=1.55, steer_deg=25.0, pattern=Gaussian(37.5, 3.0))


def test_poisson_disc_is_feasible():
    assert SPEC.feasible(poisson_disc(SPEC, np.random.default_rng(0)))


def test_reduction_matches_direct_steering():
    p = poisson_disc(SPEC, np.random.default_rng(1))
    ev = PlanarEvaluator(SPEC)
    ang = np.radians(np.arange(0, 360, 3))
    rim = SPEC.s * np.stack([np.cos(ang), np.sin(ang)], 1)
    direct = steering_check(SPEC, p, rim)
    reduced = ev.worst_db(p)
    assert direct.max() <= reduced + 0.02
    assert direct.max() >= reduced - 0.3


def test_gradient_of_peak_levels():
    ev = PlanarEvaluator(SPEC)
    p = poisson_disc(SPEC, np.random.default_rng(2))
    v = ev.peaks(p, 0.5)
    f, g = ev.value_and_grad(p, v)
    d = np.random.default_rng(3).normal(size=(SPEC.n, 2)) * 1e-6
    f2, _ = ev.value_and_grad(p + d, v)
    assert np.allclose(f2 - f, g @ np.concatenate([d[:, 0], d[:, 1]]), rtol=1e-3, atol=1e-12)


def test_optimisation_improves_and_stays_feasible():
    ev = PlanarEvaluator(SPEC)
    p0 = poisson_disc(SPEC, np.random.default_rng(4))
    p1 = smooth_descent(ev, p0, exponents=(8.0, 32.0), iters=60)
    p2, _ = local_minimax(ev, p1, max_iter=40)
    assert SPEC.feasible(p1) and SPEC.feasible(p2)
    assert ev.worst_db(p2) < ev.worst_db(p0) - 1.0
