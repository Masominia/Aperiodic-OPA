"""Element (emitter) patterns as field amplitudes in direction cosine ``u = sin(theta)``.

The far field of an array of identical emitters is the element pattern times
the array factor. Patterns here are even in ``u`` and log-concave on
``[-1, 1]``; for such patterns the worst case over a steering range has a
closed form (see :func:`aopa.farfield.steering_weight`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Isotropic:
    """Unit amplitude in every direction (array factor only)."""

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(u, dtype=float))

    def describe(self) -> str:
        return "isotropic"


@dataclass(frozen=True)
class Slit:
    """Uniformly illuminated slit of width ``width`` (µm): ``sin(a)/a`` with ``a = pi * width * u / wavelength``.

    The usual model for the lateral pattern of a waveguide grating emitter.
    """

    width: float
    wavelength: float = 1.55

    def __call__(self, u: np.ndarray) -> np.ndarray:
        a = np.pi * self.width * np.asarray(u, dtype=float) / self.wavelength
        return np.sinc(a / np.pi)

    def describe(self) -> str:
        return f"slit {self.width:g} um"


@dataclass(frozen=True)
class Gaussian:
    """Gaussian intensity in ``u``: ``|E(u)|^2 = 10^(-drop_db/10 * (u/u_ref)^2)``.

    Defined by one point of the pattern: the intensity has dropped by
    ``drop_db`` at angle ``angle_deg``.
    """

    angle_deg: float
    drop_db: float

    def __call__(self, u: np.ndarray) -> np.ndarray:
        u_ref = np.sin(np.radians(self.angle_deg))
        return 10.0 ** (-self.drop_db / 20.0 * (np.asarray(u, dtype=float) / u_ref) ** 2)

    def describe(self) -> str:
        return f"gaussian, -{self.drop_db:g} dB at {self.angle_deg:g} deg"


def from_dict(d: dict | None):
    if not d or d.get("type", "isotropic") == "isotropic":
        return Isotropic()
    kind = d["type"]
    if kind == "slit":
        return Slit(d["width"], d.get("wavelength", 1.55))
    if kind == "gaussian":
        return Gaussian(d["angle_deg"], d["drop_db"])
    raise ValueError(f"unknown element pattern {kind!r}")
