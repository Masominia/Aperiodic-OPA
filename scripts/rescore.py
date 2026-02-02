"""Recompute every stored evaluation from the stored positions, with the current evaluator."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from run_benchmarks import evaluate

from aopa import cases
from aopa.robustness import with_band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    args = parser.parse_args()
    root = Path(args.root)
    for folder in ("designs", "channels", "band"):
        for path in sorted((root / folder).glob("*.json")):
            rec = json.loads(path.read_text())
            base = rec["case"].split("@")[0].removesuffix("-band")
            spec = cases.get(base)
            if "n" in rec and rec["n"] != spec.n:
                spec = dataclasses.replace(spec, n=rec["n"], name=f"{base}@{rec['n']}")
            if rec.get("wavelengths_um") and len(rec["wavelengths_um"]) > 1:
                spec = with_band(spec, tuple(rec["wavelengths_um"]))
            old = rec["worst_db"]
            rec.update(evaluate(spec, np.array(rec["positions_um"])))
            path.write_text(json.dumps(rec, indent=1))
            print(f"{folder}/{path.stem}: {old:.3f} -> {rec['worst_db']:.3f} dB, scan {rec['scan_worst_db']:.3f} dB")


if __name__ == "__main__":
    main()
