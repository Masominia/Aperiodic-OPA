"""Two-dimensional (planar) aperiodic arrays steered over a cone.

The one-dimensional reduction carries over. With emitters at ``p_n`` in the
plane and the beam pointed at ``u0`` (a direction-cosine vector with
``|u0| <= s``), a sidelobe of the array factor at offset ``v`` is seen with
relative level ``|E(u0 + v)|^2 |A(v)|^2 / (|E(u0)|^2 N^2)``. For a radially
symmetric pattern that is log-concave in ``|u|``, the worst steering for a
given ``v`` points the beam to the far rim, ``u0 = -s v/|v|``, so

    PSL_worst = max_{v outside main lobe, |v| <= 1 + s}  w(|v|) |A(v)|^2 / N^2

with the same weight as in one dimension. ``|A(-v)| = |A(v)|`` halves the
search to ``v_y >= 0``.

Positions live in a square ``[0, D]^2`` with a minimum centre-to-centre
distance ``d_min``. In the linear program the distance constraints are
linearised; distance is a convex function of the separation vector, so the
linearisation never overestimates it and every accepted step stays feasible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import finufft
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from . import patterns
from .farfield import steering_weight


@dataclass
class PlanarSpec:
    name: str
    n: int
    side: float
    d_min: float
    steer_deg: float
    wavelength: float = 1.55
    pattern: object = field(default_factory=patterns.Isotropic)
    reference: dict = field(default_factory=dict)

    @property
    def s(self) -> float:
        return float(np.sin(np.radians(self.steer_deg)))

    @property
    def v_max(self) -> float:
        return 1.0 + self.s

    def feasible(self, p: np.ndarray, tol: float = 1e-6) -> bool:
        inside = np.all(p >= -tol) and np.all(p <= self.side + tol)
        d, _ = cKDTree(p).query(p, k=2)
        return bool(inside and d[:, 1].min() >= self.d_min - tol)


def poisson_disc(spec: PlanarSpec, rng: np.random.Generator, max_tries: int = 200_000) -> np.ndarray:
    """``n`` points in the square with pairwise distance at least ``d_min``, by random sequential addition."""
    pts = np.empty((0, 2))
    tries = 0
    while len(pts) < spec.n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place all emitters; the density is too high for random addition")
        q = rng.uniform(0, spec.side, 2)
        if len(pts) == 0 or np.min(np.hypot(*(pts - q).T)) >= spec.d_min:
            pts = np.vstack([pts, q])
    return pts


class PlanarEvaluator:
    """Worst-case sidelobe level over a steering cone, with gradients at the sidelobe maxima."""

    def __init__(self, spec: PlanarSpec, oversample: int = 4):
        self.spec = spec
        self.k = 2.0 * np.pi / spec.wavelength
        self.dv = spec.wavelength / (spec.side * oversample)
        self.m = int(np.ceil(spec.v_max / self.dv)) + 1
        self.vx = (np.arange(2 * self.m + 1) - self.m) * self.dv
        self.vy = np.arange(self.m + 1) * self.dv
        r = np.hypot(self.vx[:, None], self.vy[None, :])
        self.w = steering_weight(spec.pattern, spec.s, r) * (r <= spec.v_max)

    def _af_grid(self, p: np.ndarray) -> np.ndarray:
        t = np.mod(self.k * self.dv * p + np.pi, 2 * np.pi) - np.pi
        f = finufft.nufft2d1(
            np.ascontiguousarray(t[:, 0]),
            np.ascontiguousarray(t[:, 1]),
            np.ones(len(p), dtype=complex),
            (2 * self.m + 1, 2 * self.m + 3),
            eps=1e-10,
            isign=1,
            nthreads=1,
        )
        return f[:, self.m + 1 : 2 * self.m + 2]

    def profile(self, p: np.ndarray) -> np.ndarray:
        """Weighted level on the half-plane grid, with the main lobe removed."""
        a = np.abs(self._af_grid(p)) ** 2 / self.spec.n**2
        f = self.w * a
        f[self._main_lobe(a)] = 0.0
        return f

    def _main_lobe(self, a: np.ndarray) -> np.ndarray:
        """Pixels reached from the origin by never-increasing steps: the main lobe down to its first nulls."""
        mask = np.zeros(a.shape, dtype=bool)
        stack = [(self.m, 0)]
        mask[self.m, 0] = True
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < a.shape[0] and 0 <= jj < a.shape[1] and not mask[ii, jj] and a[ii, jj] <= a[i, j]:
                    mask[ii, jj] = True
                    stack.append((ii, jj))
        return mask

    def peaks(self, p: np.ndarray, frac: float) -> np.ndarray:
        """Sidelobe maxima within ``frac`` of the worst as ``(K, 2)`` offsets, refined by 2-D parabolic fits."""
        f = self.profile(p)
        pad = np.pad(f, 1, constant_values=-1.0)
        centre = pad[1:-1, 1:-1]
        is_max = np.ones_like(centre, dtype=bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    is_max &= centre >= pad[1 + di : pad.shape[0] - 1 + di, 1 + dj : pad.shape[1] - 1 + dj]
        is_max &= centre > 0
        rim = self._rim(p)
        top = max(centre[is_max].max(), rim[1].max())
        ii, jj = np.nonzero(is_max & (centre >= frac * top))
        lf = np.log(np.maximum(pad, 1e-300))
        out = []
        for a, b in ((1, 0), (0, 1)):
            lo = lf[ii + 1 - a, jj + 1 - b]
            mid = lf[ii + 1, jj + 1]
            hi = lf[ii + 1 + a, jj + 1 + b]
            den = lo - 2 * mid + hi
            sh = np.where(den < 0, 0.5 * (lo - hi) / np.where(den < 0, den, -1.0), 0.0)
            out.append(np.clip(sh, -1, 1))
        v = np.stack([self.vx[ii] + out[0] * self.dv, self.vy[jj] + out[1] * self.dv], axis=1)
        rv, rf = rim
        keep = (rf >= np.roll(rf, 1)) & (rf >= np.roll(rf, -1)) & (rf >= frac * top)
        return np.vstack([v, rv[keep]])

    def _rim(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Samples of the weighted map on the edge ``|v| = 1 + s`` of the visible range, where maxima can sit."""
        k = max(64, int(np.ceil(2 * np.pi * self.spec.v_max / self.dv)))
        phi = np.linspace(0.0, np.pi, k)
        v = self.spec.v_max * np.stack([np.cos(phi), np.sin(phi)], 1)
        f, _ = self.value_and_grad(p, v, grad=False)
        return v, f

    def value_and_grad(self, p: np.ndarray, v: np.ndarray, grad: bool = True):
        ph = np.exp(1j * self.k * (v @ p.T))
        a = ph.sum(1)
        r = np.hypot(v[:, 0], v[:, 1])
        w = steering_weight(self.spec.pattern, self.spec.s, r) * (r <= self.spec.v_max)
        f = w * np.abs(a) ** 2 / self.spec.n**2
        if not grad:
            return f, None
        core = -2.0 * self.k * np.imag(np.conj(a)[:, None] * ph) / self.spec.n**2
        g = np.concatenate([(w * v[:, 0])[:, None] * core, (w * v[:, 1])[:, None] * core], axis=1)
        return f, g

    def worst(self, p: np.ndarray) -> float:
        """Worst level (linear), candidate peaks refined by three rounds of local grid search."""
        return self._refined(p, 0.7, 3)

    def worst_db(self, p: np.ndarray) -> float:
        """Worst level (dB) with every candidate within 3 dB refined to high accuracy."""
        return float(10 * np.log10(self._refined(p, 0.5, 7)))

    def _refined(self, p: np.ndarray, frac: float, rounds: int) -> float:
        v = self.peaks(p, frac)
        h = self.dv
        for _ in range(rounds):
            off = np.stack(np.meshgrid(np.linspace(-h, h, 7), np.linspace(-h, h, 7)), -1).reshape(-1, 2)
            trial = (v[:, None, :] + off[None, :, :]).reshape(-1, 2)
            r = np.hypot(trial[:, 0], trial[:, 1])
            trial *= np.minimum(1.0, self.spec.v_max / np.maximum(r, 1e-12))[:, None]
            f, _ = self.value_and_grad(p, trial, grad=False)
            best = np.argmax(f.reshape(len(v), -1), axis=1)
            v = v + off[best]
            h /= 3.0
        f, _ = self.value_and_grad(p, v, grad=False)
        return float(f.max())

    def mean_level_db(self, p: np.ndarray) -> float:
        f = self.profile(p)
        r = np.hypot(self.vx[:, None], self.vy[None, :])
        keep = (r <= self.spec.v_max) & (f > 0)
        return float(10 * np.log10(f[keep].mean()))


def local_minimax(
    ev: PlanarEvaluator,
    p: np.ndarray,
    rho: float = 0.3,
    frac: float = 0.3,
    max_iter: int = 300,
    rho_min: float = 1e-5,
    log=None,
    deadline: float | None = None,
) -> tuple[np.ndarray, float]:
    spec = ev.spec
    n = spec.n
    p = p.copy()
    best = ev.worst(p)
    c = np.zeros(2 * n + 1)
    c[-1] = 1.0
    for it in range(max_iter):
        v = ev.peaks(p, frac)
        f, g = ev.value_and_grad(p, v)
        pairs = np.array(sorted(cKDTree(p).query_pairs(spec.d_min + 2.5 * rho)))
        rows = [sparse.csr_matrix(np.hstack([g, -np.ones((len(f), 1))]))]
        rhs = [-f]
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = p[i] - p[j]
            dist = np.hypot(d[:, 0], d[:, 1])
            unit = d / dist[:, None]
            m = len(pairs)
            r = np.repeat(np.arange(m), 4)
            cols = np.stack([i, j, n + i, n + j], 1).reshape(-1)
            vals = np.stack([-unit[:, 0], unit[:, 0], -unit[:, 1], unit[:, 1]], 1).reshape(-1)
            rows.append(sparse.csr_matrix((vals, (r, cols)), shape=(m, 2 * n + 1)))
            rhs.append(dist - spec.d_min)
        a_ub = sparse.vstack(rows).tocsr()
        b_ub = np.concatenate(rhs)
        lo = np.concatenate([np.maximum(-rho, -p[:, 0]), np.maximum(-rho, -p[:, 1]), [-np.inf]])
        hi = np.concatenate([np.minimum(rho, spec.side - p[:, 0]), np.minimum(rho, spec.side - p[:, 1]), [np.inf]])
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=np.stack([lo, hi], 1), method="highs")
        if res.status != 0:
            rho *= 0.5
            if rho < rho_min:
                break
            continue
        dx = res.x[: 2 * n]
        trial = p + np.stack([dx[:n], dx[n:]], 1)
        value = ev.worst(trial)
        if value < best:
            p, best = trial, value
            rho = min(rho * 1.6, 2.0)
        else:
            rho *= 0.4
        if log and it % 10 == 0:
            log(f"  iter {it}: {10 * np.log10(best):.3f} dB, {len(v)} peaks, rho {rho:.2g}")
        if rho < rho_min or (deadline and time.perf_counter() > deadline):
            break
    return p, best


def search(spec: PlanarSpec, seconds: float, seed: int = 0, starts: int = 2, log=None):
    """Multistart from random Poisson-disc layouts, then perturb-and-reoptimise until time runs out."""
    rng = np.random.default_rng(seed)
    ev = PlanarEvaluator(spec)
    t0 = time.perf_counter()
    best_p, best = None, np.inf
    history = []
    for s in range(starts):
        p, value = local_minimax(ev, poisson_disc(spec, rng))
        if value < best:
            best_p, best = p, value
        history.append((time.perf_counter() - t0, 10 * np.log10(best)))
        if log:
            log(f"start {s + 1}: {10 * np.log10(value):.3f} dB (best {10 * np.log10(best):.3f})")
    cur_p, cur = best_p, best
    runs = starts
    while time.perf_counter() - t0 < seconds:
        q = cur_p.copy()
        idx = rng.choice(spec.n, max(2, spec.n // 20), replace=False)
        q[idx] += rng.normal(0, 0.3 * spec.d_min, (len(idx), 2))
        q = np.clip(q, 0, spec.side)
        q = repair(spec, q, rng)
        p, value = local_minimax(ev, q)
        runs += 1
        if value < cur:
            cur_p, cur = p, value
            if cur < best:
                best_p, best = cur_p, cur
                history.append((time.perf_counter() - t0, 10 * np.log10(best)))
                if log:
                    log(f"{time.perf_counter() - t0:7.0f} s  run {runs}: {10 * np.log10(best):.3f} dB")
    return best_p, ev.worst_db(best_p), history, runs


def repair(spec: PlanarSpec, p: np.ndarray, rng: np.random.Generator | None = None, iters: int = 200) -> np.ndarray:
    """Push apart pairs closer than ``d_min``."""
    p = p.copy()
    for _ in range(iters):
        pairs = np.array(sorted(cKDTree(p).query_pairs(spec.d_min)))
        if not len(pairs):
            return p
        i, j = pairs[:, 0], pairs[:, 1]
        d = p[i] - p[j]
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        push = ((spec.d_min - dist) / 2 + 1e-6)[:, None] * d / dist[:, None]
        np.add.at(p, i, push)
        np.add.at(p, j, -push)
        p = np.clip(p, 0, spec.side)
    return p


def smooth_descent(
    ev: PlanarEvaluator,
    p: np.ndarray,
    exponents: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0),
    iters: int = 150,
    penalty: float = 50.0,
) -> np.ndarray:
    """Pre-shape a layout by minimising a smooth maximum of the sidelobe map with L-BFGS-B.

    The objective is ``(1/q) log sum_j f_j^q`` over every grid point outside
    the main lobe, a soft maximum that tends to the worst level as ``q``
    grows, plus a quadratic penalty on pairs closer than ``d_min``. Its
    gradient is a type-2 non-uniform FFT of the soft-max weights, so one
    evaluation costs about as much as one far-field map. The result is
    pushed back to feasibility; the linear-programming stage finishes it.
    """
    from scipy.optimize import minimize

    spec = ev.spec
    n = spec.n
    shape = (2 * ev.m + 1, 2 * ev.m + 3)
    vx = ev.vx[:, None] * np.ones((1, ev.m + 1))
    vy = np.ones((2 * ev.m + 1, 1)) * ev.vy[None, :]

    def objective(z, q):
        pts = z.reshape(2, n).T
        t = np.mod(ev.k * ev.dv * pts + np.pi, 2 * np.pi) - np.pi
        tx, ty = np.ascontiguousarray(t[:, 0]), np.ascontiguousarray(t[:, 1])
        a = finufft.nufft2d1(tx, ty, np.ones(n, dtype=complex), shape, eps=1e-9, isign=1, nthreads=1)
        a = a[:, ev.m + 1 : 2 * ev.m + 2]
        level = np.abs(a) ** 2 / n**2
        f = ev.w * level
        f[ev._main_lobe(level)] = 0.0
        top = f.max()
        r = (f / top) ** q
        tot = r.sum()
        value = np.log(top) + np.log(tot) / q
        soft = r / (tot * np.maximum(f, 1e-300))
        base = soft * ev.w * np.conj(a) / n**2
        g = np.empty(2 * n)
        for comp, vv in enumerate((vx, vy)):
            coef = np.zeros(shape, dtype=complex)
            coef[:, ev.m + 1 : 2 * ev.m + 2] = base * vv
            s = finufft.nufft2d2(tx, ty, coef, eps=1e-9, isign=1, nthreads=1)
            g[comp * n : (comp + 1) * n] = -2.0 * ev.k * np.imag(s)
        pairs = np.array(sorted(cKDTree(pts).query_pairs(spec.d_min)))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = pts[i] - pts[j]
            dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
            short = spec.d_min - dist
            value += penalty * np.sum(short**2) / spec.d_min**2
            dd = (-2.0 * penalty * short / spec.d_min**2 / dist)[:, None] * d
            gp = np.zeros((n, 2))
            np.add.at(gp, i, dd)
            np.add.at(gp, j, -dd)
            g += gp.T.reshape(-1)
        return value, g

    z = p.T.reshape(-1).copy()
    bounds = [(0.0, spec.side)] * (2 * n)
    for q in exponents:
        res = minimize(objective, z, args=(q,), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": iters})
        z = res.x
    return repair(spec, z.reshape(2, n).T, np.random.default_rng(0))


def steering_check(spec: PlanarSpec, p: np.ndarray, directions: np.ndarray, oversample: int = 6) -> np.ndarray:
    """PSL (dB) with the beam steered to each direction-cosine vector in ``directions``, from the full steered map.

    Independent of :class:`PlanarEvaluator`: the intensity ``|E(u)|^2 |A(u - u0)|^2``
    is sampled over the visible disc, the main lobe is the region reached from
    the beam by never-increasing steps, and the remaining maximum is compared
    with the refined beam peak.
    """
    k = 2.0 * np.pi / spec.wavelength
    du = spec.wavelength / (spec.side * oversample)
    m = int(np.ceil(1.0 / du)) + 1
    u = (np.arange(2 * m + 1) - m) * du
    uu = np.hypot(u[:, None], u[None, :])
    e2 = spec.pattern(np.minimum(uu, 1.0)) ** 2 * (uu <= 1.0)
    t = np.mod(k * du * p + np.pi, 2 * np.pi) - np.pi
    tx, ty = np.ascontiguousarray(t[:, 0]), np.ascontiguousarray(t[:, 1])
    out = []
    for u0 in np.atleast_2d(directions):
        c = np.exp(-1j * k * (p @ u0))
        a = finufft.nufft2d1(tx, ty, c, (2 * m + 1, 2 * m + 1), eps=1e-10, isign=1, nthreads=1)
        inten = e2 * np.abs(a) ** 2
        i0 = np.unravel_index(
            np.argmax(np.where(np.hypot(u[:, None] - u0[0], u[None, :] - u0[1]) < 2 * du, inten, 0)), inten.shape
        )
        mask = np.zeros(inten.shape, dtype=bool)
        stack = [i0]
        mask[i0] = True
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ii, jj = i + di, j + dj
                if (
                    0 <= ii < inten.shape[0]
                    and 0 <= jj < inten.shape[1]
                    and not mask[ii, jj]
                    and inten[ii, jj] <= inten[i, j]
                ):
                    mask[ii, jj] = True
                    stack.append((ii, jj))
        side = np.where(mask, 0.0, inten)
        cand = np.argsort(side.ravel())[-20:]

        def level(points, u0=u0):
            r = np.hypot(points[:, 0], points[:, 1])
            ph = np.exp(1j * k * ((points - u0) @ p.T)).sum(1)
            return spec.pattern(np.minimum(r, 1.0)) ** 2 * (r <= 1.0) * np.abs(ph) ** 2

        def refine(start):
            pt, h = start.copy(), du
            for _ in range(6):
                g = np.stack(np.meshgrid(np.linspace(-h, h, 7), np.linspace(-h, h, 7)), -1).reshape(-1, 2) + pt
                g *= np.minimum(1.0, 1.0 / np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-12))[:, None]
                f = level(g)
                pt = g[int(np.argmax(f))]
                h /= 3.0
            return f.max()

        starts = np.stack([u[np.unravel_index(cand, side.shape)[0]], u[np.unravel_index(cand, side.shape)[1]]], 1)
        best = max(refine(s0) for s0 in starts)
        peak = refine(np.array([u[i0[0]], u[i0[1]]]))
        out.append(10 * np.log10(best / peak))
    return np.array(out)
