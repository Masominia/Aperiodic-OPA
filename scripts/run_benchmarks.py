"""Synthesise a layout for each published case and store it with its full evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from aopa import cases  # noqa: E402
from aopa.farfield import Evaluator, beam_metrics, steering_scan  # noqa: E402
from aopa.optimize import search  # noqa: E402
from aopa.robustness import with_band  # noqa: E402

BUDGET = {
    "zang2025-32": 900,
    "zang2025-64": 1800,
    "ma2024-128": 2700,
    "zhuang2018-128": 2700,
    "komljenovic2017-192": 3600,
}


def evaluate(spec, x: np.ndarray) -> dict:
    ev = Evaluator(spec)
    angles = np.linspace(-spec.steer_deg, spec.steer_deg, 2 * int(round(spec.steer_deg)) + 1)
    scan = steering_scan(x, spec, angles)
    return {
        "worst_db": ev.worst_db(x),
        "scan_angles_deg": angles.tolist(),
        "scan_db": np.round(scan, 4).tolist(),
        "scan_worst_db": float(scan.max()),
        "mean_level_db": ev.mean_level_db(x),
        **beam_metrics(x, spec),
        "aperture_um": float(x[-1] - x[0]),
        "gap_min_um": float(np.diff(x).min()),
        "gap_max_um": float(np.diff(x).max()),
        "feasible": spec.feasible(x),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=list(cases.CASES))
    parser.add_argument("--seconds", type=float, help="override the per-case budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--band", type=float, nargs="+", help="judge every layout at these wavelengths (um)")
    parser.add_argument("--warm", action="store_true", help="also start from the stored layout, if any")
    parser.add_argument("--tolerance", type=float, default=0.0, help="design for emitter displacement (um)")
    parser.add_argument("--out", default="results/designs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        spec = cases.get(name)
        if args.band:
            spec = with_band(spec, tuple(args.band))
        budget = args.seconds or BUDGET.get(name, 900)
        path = out / f"{spec.name}.json"
        prev = json.loads(path.read_text()) if path.exists() else None
        initial = [np.array(prev["positions_um"])] if prev and args.warm else None
        start = time.perf_counter()
        res = search(
            spec,
            tolerance=args.tolerance,
            seconds=budget,
            seed=args.seed,
            initial=initial,
            log=lambda m, n=name: print(f"[{n}] {m}", flush=True),
        )
        record = {
            "case": spec.name,
            "wavelengths_um": list(spec.wavelengths),
            "seed": args.seed,
            "budget_s": budget + (prev.get("budget_s", 0) if initial else 0),
            "warm_start": bool(initial),
            "design_tolerance_um": args.tolerance,
            "wall_s": time.perf_counter() - start,
            "local_runs": res.local_runs,
            "history": [[round(t, 1), round(v, 4)] for t, v in res.history],
            "positions_um": np.round(res.x, 8).tolist(),
            "reference": spec.reference,
            **evaluate(spec, res.x),
        }
        if prev:
            if prev["worst_db"] <= record["worst_db"]:
                print(f"[{name}] kept stored {prev['worst_db']:.3f} dB over new {record['worst_db']:.3f} dB")
                continue
        path.write_text(json.dumps(record, indent=1))
        print(f"[{name}] {record['worst_db']:.3f} dB (reported {spec.reference['reported_db']} dB)", flush=True)


if __name__ == "__main__":
    main()
