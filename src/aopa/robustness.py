"""How designs hold up under fabrication and calibration errors and away from the design wavelength.

Static errors (emitter displacement, residual phase and amplitude errors
after calibration) do not depend on where the beam points, so the reduction
of the whole steering range to one weighted problem in ``v`` still holds. The
array factor is no longer even, so both signs of ``v`` are scanned, and the
main lobe is normalised by its actual (lowered, slightly shifted) peak.
"""

from __future__ import annotations

import dataclasses

import finufft
import numpy as np

from .farfield import steering_weight
from .spec import Spec


def worst_with_errors(
    spec: Spec, x: np.ndarray, weights: np.ndarray | None = None, wavelength: float | None = None, oversample: int = 8
) -> float:
    """Worst-case sidelobe level (dB) over the steering range for complex emitter weights ``weights``."""
    lam = wavelength or spec.wavelength
    x = np.asarray(x, dtype=float)
    c = np.ones(len(x), dtype=complex) if weights is None else np.asarray(weights, dtype=complex)
    span = x[-1] - x[0]
    dv = lam / (span * oversample)
    m = int(np.ceil(spec.v_max / dv)) + 2
    k = 2.0 * np.pi / lam
    theta = np.mod(k * dv * (x - x[0]) + np.pi, 2.0 * np.pi) - np.pi
    a = finufft.nufft1d1(theta, c, 2 * m + 1, eps=1e-12, isign=1, nthreads=1)
    v = (np.arange(2 * m + 1) - m) * dv
    p = np.abs(a) ** 2
    centre = m + int(np.argmax(p[m - oversample : m + oversample + 1])) - oversample
    left = centre
    while left > 0 and p[left - 1] <= p[left]:
        left -= 1
    right = centre
    while right < len(p) - 1 and p[right + 1] <= p[right]:
        right += 1
    peak = _refine_peak(x, c, k, v[centre], dv)
    w = steering_weight(spec.pattern, spec.s, np.abs(v)) * (np.abs(v) <= spec.v_max)
    f = w * p
    f[left : right + 1] = 0.0
    idx = np.flatnonzero((f >= np.roll(f, 1)) & (f >= np.roll(f, -1)) & (f >= 0.7 * f.max()))
    best = _refine_many(x, c, k, v[idx], dv, spec)
    return float(10.0 * np.log10(best / peak))


def _refine_many(x, c, k, v0, dv, spec: Spec, rounds: int = 5) -> float:
    """Largest weighted level near each start, by joint zoom searches over all starts."""
    lo, hi = v0 - dv, v0 + dv
    for _ in range(rounds):
        t = np.linspace(lo, hi, 11, axis=1)
        tv = np.clip(t, -spec.v_max, spec.v_max)
        f = np.abs(np.exp(1j * k * tv[..., None] * x) @ c) ** 2
        f = f * steering_weight(spec.pattern, spec.s, np.abs(tv))
        j = np.argmax(f, axis=1)
        rows = np.arange(len(t))
        lo, hi = t[rows, np.maximum(j - 1, 0)], t[rows, np.minimum(j + 1, 10)]
    return float(f.max())


def _refine_peak(x, c, k, v0, dv, spec: Spec | None = None, rounds: int = 5) -> float:
    lo, hi = v0 - dv, v0 + dv
    for _ in range(rounds):
        t = np.linspace(lo, hi, 11)
        f = np.abs(np.exp(1j * k * np.outer(t, x)) @ c) ** 2
        if spec is not None:
            f = f * steering_weight(spec.pattern, spec.s, np.abs(t)) * (np.abs(t) <= spec.v_max)
        j = int(np.argmax(f))
        lo, hi = t[max(j - 1, 0)], t[min(j + 1, 10)]
    return float(f.max())


def position_errors(
    spec: Spec, x: np.ndarray, amplitude: float, trials: int = 300, seed: int = 0, uniform: bool = True
) -> np.ndarray:
    """Worst-case level for ``trials`` layouts with each emitter displaced by up to ``amplitude`` µm."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        d = rng.uniform(-amplitude, amplitude, len(x)) if uniform else rng.normal(0.0, amplitude, len(x))
        out.append(worst_with_errors(spec, np.sort(x + d)))
    return np.array(out)


def phase_errors(spec: Spec, x: np.ndarray, sigma_deg: float, trials: int = 300, seed: int = 0) -> np.ndarray:
    """Worst-case level with Gaussian residual phase errors of ``sigma_deg`` on every emitter."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        phi = np.radians(rng.normal(0.0, sigma_deg, len(x)))
        out.append(worst_with_errors(spec, x, np.exp(1j * phi)))
    return np.array(out)


def wavelength_sweep(spec: Spec, x: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Worst-case level at each wavelength, with the element pattern held fixed."""
    return np.array([worst_with_errors(spec, x, wavelength=lam) for lam in wavelengths])


def with_band(spec: Spec, wavelengths: tuple[float, ...], name: str | None = None) -> Spec:
    """The same specification judged at several wavelengths at once."""
    return dataclasses.replace(spec, name=name or f"{spec.name}-band", wavelengths=tuple(wavelengths))
