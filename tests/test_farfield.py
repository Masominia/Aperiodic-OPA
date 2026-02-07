import numpy as np
import pytest

from aopa.farfield import (
    Evaluator,
    array_factor,
    array_factor_grid,
    beam_metrics,
    steering_scan,
    steering_weight,
    steering_weight_numeric,
)
from aopa.patterns import Gaussian, Isotropic, Slit
from aopa.spec import Spec, random_layout

PATTERNS = [Isotropic(), Slit(0.5), Slit(1.2), Gaussian(48.6, 1.0), Gaussian(30.0, 3.0)]


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda p: p.describe())
def test_patterns_even_and_log_concave(pattern):
    u = np.linspace(-1, 1, 2001)
    e = pattern(u)
    assert np.allclose(e, e[::-1]) and np.isclose(pattern(np.array([0.0]))[0], 1.0)
    second = np.diff(np.log(e), 2)
    assert np.all(second <= 1e-12)


@pytest.mark.parametrize("pattern", PATTERNS, ids=lambda p: p.describe())
@pytest.mark.parametrize("steer", [0.0, 30.0, 60.0, 82.0, 90.0])
def test_closed_form_weight_matches_brute_force(pattern, steer):
    s = np.sin(np.radians(steer))
    v = np.linspace(0, 1 + s, 400)
    closed = steering_weight(pattern, s, v)
    brute = steering_weight_numeric(pattern, s, v, n_steer=20001)
    assert np.allclose(closed, brute, rtol=1e-3, atol=1e-6)


def test_nufft_matches_direct_sum():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 400, 50))
    dv = 1.55 / (400 * 8)
    direct = array_factor(x, np.arange(3000) * dv, 1.55)
    fast = array_factor_grid(x, dv, 3000, 1.55)
    assert np.max(np.abs(direct - fast)) < 1e-8


def test_half_wavelength_uniform_array_has_the_textbook_sidelobe():
    spec = Spec("uniform", n=64, gap_min=0.775, gap_max=0.775, aperture=63 * 0.775, steer_deg=0.0)
    x = np.arange(64) * 0.775
    assert abs(Evaluator(spec).worst_db(x) - (-13.26)) < 0.05


def test_wide_pitch_uniform_array_shows_grating_lobes():
    spec = Spec("uniform", n=32, gap_min=3.0, gap_max=3.0, aperture=93.0, steer_deg=30.0)
    x = np.arange(32) * 3.0
    assert Evaluator(spec).worst_db(x) > -0.5


@pytest.mark.parametrize("pattern", [Isotropic(), Slit(0.5), Gaussian(40.0, 2.0)], ids=lambda p: p.describe())
def test_reduction_equals_direct_steering_scan(pattern):
    spec = Spec("t", n=24, gap_min=2.0, gap_max=5.0, aperture=80.0, steer_deg=55.0, pattern=pattern)
    x = random_layout(spec, np.random.default_rng(4))
    reduced = Evaluator(spec).worst_db(x)
    angles = np.linspace(-55, 55, 221)
    scan = steering_scan(x, spec, angles)
    assert scan.max() <= reduced + 0.01
    assert abs(max(scan[0], scan[-1]) - reduced) < 0.02


def test_peaks_and_gradient_are_consistent():
    spec = Spec("t", n=20, gap_min=2.0, gap_max=5.0, aperture=60.0, steer_deg=40.0, pattern=Slit(0.5))
    ev = Evaluator(spec)
    x = random_layout(spec, np.random.default_rng(1))
    pk = ev.refine(x, ev.peaks(x, 0.5))
    f, g = ev.value_and_grad(x, pk)
    d = np.random.default_rng(2).normal(size=len(x)) * 1e-6
    f2, _ = ev.value_and_grad(x + d, pk)
    assert np.allclose(f2 - f, g @ d, rtol=1e-3, atol=1e-12)


def test_mean_level_is_below_the_peak():
    spec = Spec("t", n=32, gap_min=2.0, gap_max=5.0, aperture=103.0, steer_deg=60.0, pattern=Slit(0.5))
    ev = Evaluator(spec)
    x = random_layout(spec, np.random.default_rng(3))
    assert ev.mean_level_db(x) < ev.worst_db(x)


def test_beam_metrics():
    x = np.arange(64) * 0.775
    spec = Spec("u", n=64, gap_min=0.775, gap_max=0.775, aperture=63 * 0.775, steer_deg=0.0)
    m = beam_metrics(x, spec)
    expected = np.degrees(0.886 * 1.55 / (64 * 0.775))
    assert abs(m["fwhm_deg"] - expected) / expected < 0.03
    assert 0.85 < m["main_lobe_fraction"] < 0.95


def test_edge_of_range_is_tracked_and_scan_is_symmetric():
    spec = Spec("edge", n=24, gap_min=1.6, aperture=60.0, steer_deg=90.0)
    ev = Evaluator(spec)
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = random_layout(spec, rng)
        precise = ev.worst_db(x)
        assert abs(10 * np.log10(ev.worst(x)) - precise) < 0.3
        scan = steering_scan(x, spec, np.array([-90.0, 90.0]))
        assert abs(scan[0] - scan[1]) < 1e-6 and abs(scan.max() - precise) < 0.02
