"""Fabrication, calibration and wavelength tolerance of the stored designs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from aopa import cases, robustness  # noqa: E402

POSITION_ERRORS_UM = (0.05, 0.1, 0.2)
PHASE_ERRORS_DEG = (5.0, 10.0, 20.0)
WAVELENGTHS_UM = np.round(np.linspace(1.50, 1.60, 11), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", default="results/designs")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out", default="results/robustness.json")
    parser.add_argument("--baselines", help="instead, test the best PSO and GA layouts stored in this folder")
    args = parser.parse_args()
    if args.baselines:
        compare_baselines(args.baselines, args.trials)
        return
    report = {}
    for path in sorted(Path(args.designs).glob("*.json")):
        rec = json.loads(path.read_text())
        name = rec["case"]
        spec = cases.get(name.split("@")[0].replace("-band", ""))
        x = np.array(rec["positions_um"])
        entry = {"nominal_db": robustness.worst_with_errors(spec, x)}
        entry["position"] = {
            str(a): _stats(robustness.position_errors(spec, x, a, args.trials, seed=1)) for a in POSITION_ERRORS_UM
        }
        entry["phase"] = {
            str(s): _stats(robustness.phase_errors(spec, x, s, args.trials, seed=2)) for s in PHASE_ERRORS_DEG
        }
        entry["wavelength_um"] = WAVELENGTHS_UM.tolist()
        entry["wavelength_db"] = np.round(robustness.wavelength_sweep(spec, x, WAVELENGTHS_UM), 4).tolist()
        report[path.stem] = entry
        print(path.stem, json.dumps({k: v for k, v in entry.items() if k in ("nominal_db", "position")}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


def compare_baselines(folder: str, trials: int) -> None:
    """Displacement tolerance of the best particle-swarm and genetic-algorithm layouts, for comparison."""
    out = {}
    for path in sorted(Path(folder).glob("*.json")):
        rec = json.loads(path.read_text())
        spec = cases.get(rec["case"])
        out[rec["case"]] = {}
        for method in ("pso", "ga"):
            best = min((r for r in rec["runs"] if r["method"] == method), key=lambda r: r["worst_db"])
            x = np.array(best["positions_um"])
            out[rec["case"]][method] = {
                "nominal_db": robustness.worst_with_errors(spec, x),
                "position": {
                    str(a): _stats(robustness.position_errors(spec, x, a, trials, seed=1)) for a in POSITION_ERRORS_UM
                },
            }
            print(rec["case"], method, json.dumps(out[rec["case"]][method]["position"]), flush=True)
    Path("results/robustness_baselines.json").write_text(json.dumps(out, indent=1))


def _stats(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "std": float(a.std()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


if __name__ == "__main__":
    main()
