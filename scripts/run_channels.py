"""How many emitters does this method need to match a published design? Same constraints, fewer channels."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from run_benchmarks import evaluate  # noqa: E402

from aopa import cases  # noqa: E402
from aopa.optimize import search  # noqa: E402

PLAN = {
    "komljenovic2017-192": [64, 80, 96, 128],
    "zhuang2018-128": [64, 80, 96],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(PLAN), action="append")
    parser.add_argument("--seconds", type=float, default=900.0)
    parser.add_argument("--out", default="results/channels")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for base in args.case or list(PLAN):
        for n in PLAN[base]:
            spec = dataclasses.replace(cases.get(base), n=n, name=f"{base}@{n}")
            path = out / f"{spec.name}.json"
            if path.exists():
                continue
            res = search(spec, seconds=args.seconds, seed=0)
            rec = {
                "case": base,
                "n": n,
                "budget_s": args.seconds,
                "local_runs": res.local_runs,
                "positions_um": np.round(res.x, 8).tolist(),
                **evaluate(spec, res.x),
            }
            path.write_text(json.dumps(rec))
            print(f"[{spec.name}] {rec['worst_db']:.3f} dB", flush=True)


if __name__ == "__main__":
    main()
