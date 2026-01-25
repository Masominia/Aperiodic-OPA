"""Far fields of one-dimensional arrays and the steering-range sidelobe metric.

Coordinates are direction cosines ``u = sin(theta)``. With emitters at ``x_n``
(µm), equal amplitudes and the phases that point the beam at ``u0``, the
far-field intensity is

    I(u) = |E(u)|^2 |A(u - u0)|^2,      A(v) = sum_n exp(i k x_n v),

with ``E`` the element pattern and ``k = 2 pi / wavelength``. Writing
``v = u - u0``, a sidelobe of ``A`` at offset ``v`` is seen at steering ``u0``
with relative level ``|E(u0 + v)|^2 |A(v)|^2 / (|E(u0)|^2 N^2)``, as long as
``|u0 + v| <= 1``. The worst case over every steering angle in ``[-s, s]`` is
therefore one weighted problem in ``v``:

    PSL_worst = max_{v_null <= |v| <= 1 + s}  w(v) |A(v)|^2 / N^2,
    w(v) = max_{u0} |E(u0 + v)|^2 / |E(u0)|^2.

``|A|`` is even, so only ``v >= 0`` is needed. For an even, log-concave
pattern the maximum over ``u0`` sits at the far edge, ``u0 = -s``, giving
``w(v) = |E(v - s)|^2 / |E(s)|^2`` (:func:`steering_weight`).
"""

from __future__ import annotations

from dataclasses import dataclass

import finufft
import numpy as np

from .spec import Spec

OVERSAMPLE = 8
CHUNK = 4_000_000


def array_factor(x: np.ndarray, v: np.ndarray, wavelength: float) -> np.ndarray:
    """``A(v)`` by direct summation, in chunks to bound memory."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    k = 2.0 * np.pi / wavelength
    out = np.empty(v.shape, dtype=complex)
    flat_v, flat_out = v.reshape(-1), out.reshape(-1)
    step = max(1, CHUNK // max(1, len(x)))
    for i in range(0, len(flat_v), step):
        flat_out[i : i + step] = np.exp(1j * k * np.outer(flat_v[i : i + step], x)).sum(1)
    return out


def array_factor_grid(x: np.ndarray, dv: float, m: int, wavelength: float) -> np.ndarray:
    """``A(j dv)`` for ``j = 0 .. m-1`` with a type-1 non-uniform FFT."""
    k = 2.0 * np.pi / wavelength
    theta = np.mod(k * dv * np.asarray(x, dtype=float) + np.pi, 2.0 * np.pi) - np.pi
    f = finufft.nufft1d1(theta, np.ones(len(x), dtype=complex), 2 * m + 2, eps=1e-12, isign=1, nthreads=1)
    return f[m + 1 : 2 * m + 1]


def steering_weight(pattern, s: float, v: np.ndarray) -> np.ndarray:
    """Worst-case visibility of an array-factor sidelobe at offset ``v >= 0`` over steering in ``[-s, s]``."""
    v = np.asarray(v, dtype=float)
    e_edge = pattern(np.array([s]))[0]
    w = (pattern(v - s) / e_edge) ** 2
    return np.where(v - s <= 1.0 + 1e-12, w, 0.0)


def steering_weight_numeric(pattern, s: float, v: np.ndarray, n_steer: int = 4001) -> np.ndarray:
    """The same weight by brute force over steering angles, for any pattern."""
    u0 = np.linspace(-s, s, n_steer)
    e0 = pattern(u0) ** 2
    out = np.zeros(len(v))
    for sign in (1.0, -1.0):
        u = u0[None, :] + sign * np.asarray(v)[:, None]
        ratio = np.where(np.abs(u) <= 1.0, pattern(np.clip(u, -1.0, 1.0)) ** 2 / e0[None, :], 0.0)
        out = np.maximum(out, ratio.max(1))
    return out


@dataclass
class Peaks:
    """Sidelobe maxima: wavelength index, offset ``v`` and weighted level (linear)."""

    band: np.ndarray
    v: np.ndarray
    level: np.ndarray

    def __len__(self) -> int:
        return len(self.v)


class Evaluator:
    """Worst-case sidelobe level of layouts for one :class:`Spec`, with gradients at the peaks."""

    def __init__(self, spec: Spec, oversample: int = OVERSAMPLE, offsets: np.ndarray | None = None):
        """``offsets`` (``(K, n)``, µm) adds displaced copies of every layout as further scenarios.

        Every wavelength is paired with every offset; the worst scenario counts.
        With fixed random offsets this gives a layout that keeps its sidelobe
        level under fabrication errors of that size.
        """
        self.spec = spec
        self.n = spec.n
        offsets = np.zeros((1, spec.n)) if offsets is None else np.atleast_2d(offsets)
        self.grids = []
        for lam in spec.wavelengths:
            dv = lam / (spec.aperture_max * oversample)
            m = int(np.ceil(spec.v_max / dv)) + 2
            v = np.arange(m) * dv
            w = steering_weight(spec.pattern, spec.s, v)
            for off in offsets:
                self.grids.append((lam, dv, v, w, off))

    def weight(self, band: int, v: np.ndarray) -> np.ndarray:
        return steering_weight(self.spec.pattern, self.spec.s, v) * (v <= self.spec.v_max)

    def profile(self, x: np.ndarray, band: int = 0) -> tuple[np.ndarray, int]:
        """Weighted level on the grid of one wavelength, and the index of the first null of ``A``."""
        lam, dv, v, w, off = self.grids[band]
        p = np.abs(array_factor_grid(x + off, dv, len(v), lam)) ** 2 / self.n**2
        rising = np.flatnonzero(np.diff(p) > 0)
        null = int(rising[0]) if len(rising) else len(p) - 1
        f = w * p
        f[:null] = 0.0
        f[v > self.spec.v_max] = 0.0
        return f, null

    def _grid_peaks(self, x: np.ndarray, frac: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
        found, top = [], 0.0
        for b in range(len(self.grids)):
            f, _ = self.profile(x, b)
            interior = (f[1:-1] >= f[:-2]) & (f[1:-1] > f[2:])
            idx = np.flatnonzero(interior) + 1
            if len(idx):
                found.append((b, idx, f))
                top = max(top, f[idx].max())
        return [(b, idx[f[idx] >= frac * top], f) for b, idx, f in found], top

    def peaks(self, x: np.ndarray, frac: float = 0.1) -> Peaks:
        """Grid maxima within ``frac`` of the worst, located to sub-sample accuracy by parabolic interpolation."""
        bands, vs = [], []
        grid, _ = self._grid_peaks(x, frac)
        for b, idx, f in grid:
            _, dv, v, _, _ = self.grids[b]
            inner = (idx > 0) & (idx < len(f) - 1)
            shift = np.zeros(len(idx))
            i = idx[inner]
            lo, mid, hi = (np.log(np.maximum(f[j], 1e-300)) for j in (i - 1, i, i + 1))
            denom = lo - 2.0 * mid + hi
            shift[inner] = np.where(denom < 0, 0.5 * (lo - hi) / np.where(denom < 0, denom, -1.0), 0.0)
            vv = np.clip(v[idx] + np.clip(shift, -1, 1) * dv, 0.0, self.spec.v_max)
            bands.append(np.full(len(idx), b))
            vs.append(vv)
        for b in range(len(self.grids)):
            # The edge of the visible range is a maximum whenever the map still rises there; track it exactly.
            bands.append(np.array([b]))
            vs.append(np.array([self.spec.v_max]))
        band = np.concatenate(bands)
        v = np.concatenate(vs)
        level, _ = self.value_and_grad(x, Peaks(band, v, np.zeros(len(v))), grad=False)
        return Peaks(band, v, level)

    def value_and_grad(self, x: np.ndarray, pk: Peaks, grad: bool = True):
        """Weighted levels at the given peaks and their gradients with respect to the positions."""
        level = np.empty(len(pk))
        g = np.empty((len(pk), len(x))) if grad else None
        for b, (lam, _, _, _, off) in enumerate(self.grids):
            sel = np.flatnonzero(pk.band == b)
            if not len(sel):
                continue
            k = 2.0 * np.pi / lam
            v = pk.v[sel]
            ph = np.exp(1j * k * np.outer(v, x + off))
            a = ph.sum(1)
            w = self.weight(b, v)
            level[sel] = w * np.abs(a) ** 2 / self.n**2
            if grad:
                g[sel] = (w * -2.0 * k * v)[:, None] * np.imag(np.conj(a)[:, None] * ph) / self.n**2
        return level, g

    def refine(self, x: np.ndarray, pk: Peaks, rounds: int = 6) -> Peaks:
        """Golden-free zoom search around each peak to near machine precision in ``v``."""
        v = pk.v.copy()
        for b, (lam, dv, _, _, off) in enumerate(self.grids):
            sel = np.flatnonzero(pk.band == b)
            if not len(sel):
                continue
            half = dv
            vb = v[sel]
            for _ in range(rounds):
                trial = vb[:, None] + np.linspace(-half, half, 9)[None, :]
                trial = np.clip(trial, 0.0, self.spec.v_max)
                f = self.weight(b, trial) * np.abs(array_factor(x + off, trial, lam)) ** 2
                vb = trial[np.arange(len(vb)), np.argmax(f, axis=1)]
                half /= 4.0
            v[sel] = vb
        out = Peaks(pk.band, v, np.zeros(len(v)))
        out.level, _ = self.value_and_grad(x, out, grad=False)
        return out

    def worst(self, x: np.ndarray, precise: bool = False) -> float:
        """Worst-case sidelobe level over the steering range and wavelengths, linear."""
        pk = self.peaks(x, frac=0.5 if precise else 0.9)
        if precise:
            pk = self.refine(x, pk)
        return float(pk.level.max()) if len(pk) else 0.0

    def worst_db(self, x: np.ndarray, precise: bool = True) -> float:
        return float(10.0 * np.log10(self.worst(x, precise)))

    def mean_level_db(self, x: np.ndarray, band: int = 0) -> float:
        """Mean weighted sidelobe level over ``[v_null, 1 + s]``: no peak can lie below it."""
        f, null = self.profile(x, band)
        _, _, v, _, _ = self.grids[band]
        keep = (np.arange(len(f)) >= null) & (v <= self.spec.v_max)
        return float(10.0 * np.log10(f[keep].mean()))


def steering_scan(
    x: np.ndarray, spec: Spec, angles_deg: np.ndarray, wavelength: float | None = None, oversample: int = 16
) -> np.ndarray:
    """Peak sidelobe level (dB) at each steering angle, computed directly from the steered far field.

    Independent of :class:`Evaluator`: for every angle the intensity
    ``|E(u)|^2 |A(u - u0)|^2`` is sampled over the visible range, the main lobe
    is bounded by the nearest minima on either side of the beam, and the
    largest remaining maximum is refined and compared with the beam peak.
    """
    lam = wavelength or spec.wavelength
    x = np.asarray(x, dtype=float)
    span = x[-1] - x[0]
    du = lam / (span * oversample)
    u = np.linspace(-1.0, 1.0, int(np.ceil(2.0 / du)) + 1)
    out = []
    for ang in np.atleast_1d(angles_deg):
        u0 = np.sin(np.radians(ang))
        inten = spec.pattern(u) ** 2 * np.abs(array_factor(x, u - u0, lam)) ** 2
        c = int(np.argmin(np.abs(u - u0)))
        lo = max(c - 2, 0)
        c = lo + int(np.argmax(inten[lo : c + 3]))
        left = c
        while left > 0 and inten[left - 1] <= inten[left]:
            left -= 1
        right = c
        while right < len(u) - 1 and inten[right + 1] <= inten[right]:
            right += 1
        side = inten.copy()
        side[left : right + 1] = 0.0
        cand = np.flatnonzero((side >= np.roll(side, 1)) & (side >= np.roll(side, -1)) & (side >= 0.5 * side.max()))

        def level(uu, u0=u0):
            uu = np.clip(uu, -1.0, 1.0)
            return spec.pattern(uu) ** 2 * np.abs(array_factor(x, uu - u0, lam)) ** 2

        best = 0.0
        for i in cand:
            lo, hi = u[max(i - 1, 0)], u[min(i + 1, len(u) - 1)]
            for _ in range(5):
                t = np.linspace(lo, hi, 11)
                f = level(t)
                j = int(np.argmax(f))
                lo, hi = t[max(j - 1, 0)], t[min(j + 1, 10)]
            best = max(best, f.max())
        uu = np.linspace(u[max(c - 1, 0)], u[min(c + 1, len(u) - 1)], 41)
        for _ in range(4):
            f = level(uu)
            j = int(np.argmax(f))
            uu = np.linspace(uu[max(j - 1, 0)], uu[min(j + 1, len(uu) - 1)], 41)
        peak = level(uu).max()
        out.append(10.0 * np.log10(best / peak) if best > 0 else -np.inf)
    return np.array(out)


def beam_metrics(x: np.ndarray, spec: Spec, wavelength: float | None = None) -> dict:
    """Broadside FWHM (degrees) and the share of the intensity, integrated over ``u``, inside the main lobe."""
    lam = wavelength or spec.wavelength
    span = x[-1] - x[0]
    du = lam / (span * 64)
    u = np.arange(-1.0, 1.0 + du, du)
    inten = spec.pattern(u) ** 2 * np.abs(array_factor(x, u, lam)) ** 2
    c = int(np.argmin(np.abs(u)))
    half = inten[c] / 2.0
    left = c
    while inten[left] > half:
        left -= 1
    right = c
    while inten[right] > half:
        right += 1
    fwhm = np.degrees(np.arcsin(u[right]) - np.arcsin(u[left]))
    lo = c
    while lo > 0 and inten[lo - 1] < inten[lo]:
        lo -= 1
    hi = c
    while hi < len(u) - 1 and inten[hi + 1] < inten[hi]:
        hi += 1
    power = inten
    return {"fwhm_deg": float(fwhm), "main_lobe_fraction": float(power[lo : hi + 1].sum() / power.sum())}
