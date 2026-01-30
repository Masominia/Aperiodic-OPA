"""Design specifications: what a layout must satisfy and over which steering range it is judged."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import patterns


@dataclass
class Spec:
    """A one-dimensional aperiodic OPA design problem.

    Lengths are in micrometres. ``aperture`` fixes the distance between the
    outermost emitters; with ``aperture=None`` it may take any value in
    ``[aperture_min, aperture_max]``, which default to whatever the gap bounds
    allow. ``steer_deg`` is the half-angle of the steering range, symmetric
    about broadside. ``wavelengths`` lists the wavelengths over which the
    layout is judged; the worst one counts.
    """

    name: str
    n: int
    gap_min: float
    gap_max: float = np.inf
    aperture: float | None = None
    aperture_min: float | None = None
    aperture_max: float | None = None
    steer_deg: float = 60.0
    wavelength: float = 1.55
    wavelengths: tuple[float, ...] = ()
    pattern: object = field(default_factory=patterns.Isotropic)
    reference: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.wavelengths:
            self.wavelengths = (self.wavelength,)
        lo = (self.n - 1) * self.gap_min
        hi = (self.n - 1) * self.gap_max
        if self.aperture is not None:
            if not lo - 1e-9 <= self.aperture <= hi + 1e-9:
                raise ValueError(f"{self.name}: aperture {self.aperture} incompatible with gap bounds")
            self.aperture_min = self.aperture_max = float(self.aperture)
        else:
            self.aperture_min = max(lo, self.aperture_min or lo)
            self.aperture_max = min(hi, self.aperture_max if self.aperture_max is not None else hi)
            if not np.isfinite(self.aperture_max):
                raise ValueError(f"{self.name}: unbounded aperture; set gap_max or aperture_max")

    @property
    def s(self) -> float:
        """Sine of the steering half-angle."""
        return float(np.sin(np.radians(self.steer_deg)))

    @property
    def v_max(self) -> float:
        """Largest direction-cosine offset from the beam that is ever visible: ``1 + s``."""
        return 1.0 + self.s

    def feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        g = np.diff(x)
        span = x[-1] - x[0]
        return bool(
            np.all(g >= self.gap_min - tol)
            and np.all(g <= self.gap_max + tol)
            and self.aperture_min - tol <= span <= self.aperture_max + tol
        )

    @classmethod
    def from_dict(cls, d: dict) -> Spec:
        d = dict(d)
        d["pattern"] = patterns.from_dict(d.get("pattern"))
        if "wavelengths" in d:
            d["wavelengths"] = tuple(d["wavelengths"])
        if d.get("gap_max") is None:
            d["gap_max"] = np.inf
        return cls(**d)


def project_gaps(g: np.ndarray, total: float, lo: float, hi: float, iters: int = 200) -> np.ndarray:
    """Gaps within ``[lo, hi]`` summing to ``total``, moving the free gaps by equal amounts."""
    g = np.clip(np.asarray(g, dtype=float), lo, hi)
    for _ in range(iters):
        excess = total - g.sum()
        if abs(excess) < 1e-10 * max(1.0, total):
            break
        free = g < hi - 1e-12 if excess > 0 else g > lo + 1e-12
        if not free.any():
            break
        g[free] += excess / free.sum()
        g = np.clip(g, lo, hi)
    return g


def positions(gaps: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(gaps)])


def random_layout(spec: Spec, rng: np.random.Generator) -> np.ndarray:
    """A random feasible layout: uniform gaps in the allowed band, rescaled to a random allowed aperture."""
    hi = spec.gap_max if np.isfinite(spec.gap_max) else 2.0 * spec.aperture_max / (spec.n - 1)
    g = rng.uniform(spec.gap_min, hi, spec.n - 1)
    total = rng.uniform(spec.aperture_min, spec.aperture_max)
    return positions(project_gaps(g, total, spec.gap_min, spec.gap_max))


def uniform_layout(spec: Spec, aperture: float | None = None) -> np.ndarray:
    total = aperture if aperture is not None else spec.aperture_max
    return np.linspace(0.0, total, spec.n)
