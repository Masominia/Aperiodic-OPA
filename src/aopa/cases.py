"""Published aperiodic OPA designs, restated as specifications with the constraints their authors used.

Each entry records the reported worst sidelobe level over the steering
range (negative dB, lower is better) and where it comes from. Where a paper
leaves a constraint open, the choice made here is noted in ``assumptions``.
"""

from __future__ import annotations

import numpy as np

from .patterns import Gaussian, Isotropic, Slit
from .spec import Spec

LAMBDA = 1.55


def _cases() -> dict[str, Spec]:
    return {
        "zang2025-32": Spec(
            name="zang2025-32",
            n=32,
            gap_min=2.0,
            gap_max=5.0,
            aperture=103.0,
            steer_deg=60.0,
            pattern=Slit(0.5, LAMBDA),
            reference={
                "citation": "Z. Zang, J. Wu, Q. Huang, Photonics 12, 210 (2025)",
                "doi": "10.3390/photonics12030210",
                "method": "multi-strategy enhanced particle swarm optimisation",
                "reported_db": -9.12,
                "reported_note": "PSLL from 12.26 dB down to 9.12 dB across the +/-60 deg range",
                "assumptions": "",
            },
        ),
        "zang2025-64": Spec(
            name="zang2025-64",
            n=64,
            gap_min=2.0,
            gap_max=5.0,
            aperture=208.0,
            steer_deg=60.0,
            pattern=Slit(0.5, LAMBDA),
            reference={
                "citation": "Z. Zang, J. Wu, Q. Huang, Photonics 12, 210 (2025)",
                "doi": "10.3390/photonics12030210",
                "method": "multi-strategy enhanced particle swarm optimisation",
                "reported_db": -11.75,
                "reported_note": "PSLL from 13.27 dB down to 11.75 dB across the +/-60 deg range",
                "assumptions": "",
            },
        ),
        "ma2024-128": Spec(
            name="ma2024-128",
            n=128,
            gap_min=5.0,
            gap_max=10.0,
            steer_deg=47.0,
            pattern=Gaussian(48.6, 1.0),
            reference={
                "citation": "Ma et al., Eng. Res. Express (2024)",
                "doi": "10.1088/2631-8695/ad408b",
                "method": "genetic algorithm (population 50, 50 generations)",
                "reported_db": -11.9,
                "reported_note": "side-lobe suppression between 13.4 dB and 11.9 dB over the 94 deg field of view",
                "assumptions": "element pattern modelled as Gaussian with the reported 1 dB width of 97.2 deg; "
                "aperture left free within the gap bounds",
            },
        ),
        "zhuang2018-128": Spec(
            name="zhuang2018-128",
            n=128,
            gap_min=2.67 * LAMBDA,
            aperture=127 * 9.75,
            steer_deg=82.0,
            pattern=Isotropic(),
            reference={
                "citation": "D. Zhuang et al., Opt. Express 26, 19154 (2018)",
                "doi": "10.1364/OE.26.019154",
                "method": "pattern-search-like position optimisation",
                "reported_db": -14.34,
                "reported_note": "peak side-lobe level -14.34 dB over +/-82 deg, isotropic elements",
                "assumptions": "minimum pitch 2.67 wavelengths; aperture fixed by the reported mean pitch of 9.75 um",
            },
        ),
        "komljenovic2017-192": Spec(
            name="komljenovic2017-192",
            n=192,
            gap_min=2.5,
            aperture=191 * 5.2,
            steer_deg=90.0,
            pattern=Isotropic(),
            reference={
                "citation": "T. Komljenovic, R. Helkey, L. Coldren, J. E. Bowers, Opt. Express 25, 2511 (2017)",
                "doi": "10.1364/OE.25.002511",
                "method": "particle swarm optimisation (400 particles)",
                "reported_db": -13.5,
                "reported_note": "side-mode suppression above 13.5 dB over +/-90 deg",
                "assumptions": "aperture fixed by the reported 5.2 um average pitch; isotropic elements",
            },
        ),
    }


CASES = _cases()


def get(name: str) -> Spec:
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; choose from {', '.join(CASES)}")
    return _cases()[name]


def summary() -> list[dict]:
    rows = []
    for name, s in CASES.items():
        rows.append(
            {
                "case": name,
                "elements": s.n,
                "gap_um": f"{s.gap_min:.2f}-{s.gap_max:.0f}" if np.isfinite(s.gap_max) else f">= {s.gap_min:.2f}",
                "aperture_um": f"{s.aperture_min:.0f}"
                if s.aperture_min == s.aperture_max
                else f"{s.aperture_min:.0f}-{s.aperture_max:.0f}",
                "steering_deg": s.steer_deg,
                "pattern": s.pattern.describe(),
                "reported_db": s.reference["reported_db"],
            }
        )
    return rows
