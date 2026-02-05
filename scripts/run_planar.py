"""Planar (2-D) aperiodic array over a steering cone, against a published uniform-amplitude and tapered design."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from aopa.patterns import Gaussian  # noqa: E402
from aopa.planar import (  # noqa: E402
    PlanarEvaluator,
    PlanarSpec,
    local_minimax,
    poisson_disc,
    repair,
    smooth_descent,
    steering_check,
)

LAMBDA = 1.55
SPEC = PlanarSpec(
    name="gao2024-625",
    n=625,
    side=50.5 * LAMBDA,
    d_min=LAMBDA,
    steer_deg=40.0,
    pattern=Gaussian(37.5, 3.0103),
    reference={
        "citation": "P. Gao, Z. Chen, Electronics 13, 4422 (2024)",
        "doi": "10.3390/electronics13224422",
        "uniform_db": -12.7,
        "uniform_note": "pseudo-random layout, uniform amplitudes (their step 1)",
        "tapered_db": -17.0,
        "tapered_note": "same layout with excitations optimised per beam by convex optimisation (their step 2)",
        "assumptions": "625 emitters (25 x 25 before randomisation); element pattern Gaussian with 75 deg "
        "half-power width; steering over the full 40 deg cone rather than the three planes tested in the paper",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5400.0)
    parser.add_argument("--warm", help="continue from a stored planar result (JSON)")
    parser.add_argument("--out", default="results/planar")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = PlanarEvaluator(SPEC)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    randoms = [ev.worst_db(poisson_disc(SPEC, rng)) for _ in range(50)]
    print(f"random Poisson-disc layouts: best {min(randoms):.2f}, median {np.median(randoms):.2f} dB", flush=True)
    shaped = []
    previous = json.loads(Path(args.warm).read_text()) if args.warm else None
    if previous:
        p0 = repair(SPEC, np.array(previous["positions_um"]), rng)
        shaped.append((ev.worst(p0), p0))
        print(f"continuing from stored layout: {10 * np.log10(shaped[-1][0]):.3f} dB", flush=True)
    for s in range(0 if previous else args.starts):
        p = smooth_descent(ev, poisson_disc(SPEC, rng))
        shaped.append((ev.worst(p), p))
        print(
            f"smooth start {s + 1}: {10 * np.log10(shaped[-1][0]):.3f} dB ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    shaped.sort(key=lambda t: t[0])
    p = shaped[0][1]
    remaining = args.seconds - (time.perf_counter() - t0)
    p, _ = local_minimax(
        ev, p, max_iter=100_000, frac=0.5, deadline=time.perf_counter() + remaining, log=lambda m: print(m, flush=True)
    )
    angles = np.radians(np.arange(0, 360, 15))
    rim = SPEC.s * np.stack([np.cos(angles), np.sin(angles)], 1)
    check = steering_check(SPEC, p, np.vstack([rim, [[0.0, 0.0]], 0.5 * rim[::2]]))
    rec = {
        "case": SPEC.name,
        "reference": SPEC.reference,
        "worst_db": ev.worst_db(p),
        "mean_level_db": ev.mean_level_db(p),
        "check_directions": np.vstack([rim, [[0.0, 0.0]], 0.5 * rim[::2]]).round(5).tolist(),
        "check_db": check.round(4).tolist(),
        "random_layouts_db": np.round(randoms, 3).tolist(),
        "smooth_starts_db": previous["smooth_starts_db"] if previous else [float(10 * np.log10(v)) for v, _ in shaped],
        "previous_wall_s": previous["wall_s"] if previous else 0.0,
        "feasible": SPEC.feasible(p),
        "wall_s": time.perf_counter() - t0,
        "positions_um": np.round(p, 8).tolist(),
    }
    (out / f"{SPEC.name}.json").write_text(json.dumps(rec))
    print(f"{SPEC.name}: {rec['worst_db']:.3f} dB; direct check worst {check.max():.3f} dB", flush=True)


if __name__ == "__main__":
    main()
