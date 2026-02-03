"""Particle swarm, genetic algorithm and random search on the published cases, scored exactly like this work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from aopa import baselines, cases  # noqa: E402

SETTINGS = {
    "pso": dict(particles=80, iterations=1000),
    "ga": dict(population=80, generations=1000),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=["zang2025-32", "zang2025-64"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=1800.0, help="cap per optimiser run")
    parser.add_argument("--out", default="results/baselines")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        spec = cases.get(name)
        record = {"case": name, "settings": SETTINGS, "cap_s": args.seconds, "runs": []}
        for seed in range(args.seeds):
            for method, fn in (("pso", baselines.particle_swarm), ("ga", baselines.genetic)):
                x, value, trace = fn(spec, seed=seed, seconds=args.seconds, **SETTINGS[method])
                record["runs"].append(
                    {
                        "method": method,
                        "seed": seed,
                        "worst_db": value,
                        "evaluations": (len(trace) - 1) * 80 + 80,
                        "trace": [[round(t, 2), round(v, 4)] for t, v in trace[:: max(1, len(trace) // 200)]]
                        + [[round(trace[-1][0], 2), round(trace[-1][1], 4)]],
                        "positions_um": np.round(x, 8).tolist(),
                    }
                )
                print(f"[{name}] {method} seed {seed}: {value:.3f} dB in {trace[-1][0]:.0f} s", flush=True)
        n_eval = 80_000
        _, best, scores = baselines.random_search(spec, evaluations=n_eval, seed=0)
        record["random"] = {
            "evaluations": n_eval,
            "best_db": best,
            "median_db": float(np.median(scores)),
            "p10_db": float(np.percentile(scores, 10)),
            "histogram": np.histogram(scores, bins=60)[0].tolist(),
            "edges": np.round(np.histogram(scores, bins=60)[1], 3).tolist(),
        }
        print(f"[{name}] random search best of {n_eval}: {best:.3f} dB (median {np.median(scores):.2f})", flush=True)
        (out / f"{name}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
