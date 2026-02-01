"""Build the interactive far-field explorer from the stored designs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aopa import cases

SHORT = {
    "zang2025-32": "Zang 2025 · 32",
    "zang2025-64": "Zang 2025 · 64",
    "ma2024-128": "Ma 2024 · 128",
    "zhuang2018-128": "Zhuang 2018 · 128",
    "komljenovic2017-192": "Komljenovic 2017 · 192",
}


def pattern_dict(p) -> dict:
    name = type(p).__name__
    if name == "Slit":
        return {"type": "slit", "width": p.width}
    if name == "Gaussian":
        return {"type": "gaussian", "angle_deg": p.angle_deg, "drop_db": p.drop_db}
    return {"type": "isotropic"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    parser.add_argument("--template", default="docs/explorer_template.html")
    parser.add_argument("--fragment", default="results/explorer.html")
    parser.add_argument("--page", default="docs/index.html")
    args = parser.parse_args()
    root = Path(args.root)
    out = []
    for name in cases.CASES:
        path = root / "designs" / f"{name}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        spec = cases.get(name)
        x = np.array(r["positions_um"])
        entry = {
            "id": name,
            "short": SHORT[name],
            "title": SHORT[name].replace(" · ", ", ") + " emitters",
            "n": spec.n,
            "steer": spec.steer_deg,
            "lambda": spec.wavelength,
            "gaps": f"{spec.gap_min:.2f}–{spec.gap_max:g}" if np.isfinite(spec.gap_max) else f"≥ {spec.gap_min:.2f}",
            "pattern": pattern_dict(spec.pattern),
            "pattern_label": spec.pattern.describe(),
            "aperture": float(x[-1] - x[0]),
            "ours": np.round(x, 4).tolist(),
            "uniform": np.round(np.linspace(0, x[-1], spec.n), 4).tolist(),
            "ours_db": r["worst_db"],
            "floor_db": r["mean_level_db"],
            "reported_db": spec.reference["reported_db"],
            "reported_note": spec.reference["reported_note"],
            "citation": spec.reference["citation"],
            "assumptions": ("Modelling choices: " + spec.reference["assumptions"] + ".")
            if spec.reference["assumptions"]
            else "",
            "scan_angles": r["scan_angles_deg"],
            "scan_db": r["scan_db"],
            "worst_angle": float(r["scan_angles_deg"][int(np.argmax(r["scan_db"]))]),
        }
        base = root / "baselines" / f"{name}.json"
        if base.exists():
            runs = [q for q in json.loads(base.read_text())["runs"] if q["method"] == "pso"]
            if runs:
                entry["pso"] = np.round(min(runs, key=lambda q: q["worst_db"])["positions_um"], 4).tolist()
        out.append(entry)
    blob = json.dumps({"cases": out}, separators=(",", ":")).replace("</", "<\\/")
    body = Path(args.template).read_text().replace("__DATA__", blob)
    Path(args.fragment).write_text(body)
    head, rest = body.split("<style>", 1)
    style, rest = rest.split("</style>", 1)
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}<style>{style}</style>\n</head>\n<body>\n{rest}\n</body>\n</html>\n"
    )
    Path(args.page).write_text(page)
    print(f"{len(out)} cases; wrote {args.fragment} and {args.page}")


if __name__ == "__main__":
    main()
