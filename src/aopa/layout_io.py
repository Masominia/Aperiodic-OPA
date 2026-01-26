"""Writing layouts out: CSV of positions and a GDS emitter array for the mask layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_csv(x: np.ndarray, path: str | Path) -> None:
    """One emitter per row: index, centre position and gap to the previous emitter (µm)."""
    x = np.asarray(x, dtype=float)
    gaps = np.concatenate([[np.nan], np.diff(x)])
    rows = ["index,x_um,gap_um"] + [
        f"{i},{xi:.6f},{'' if np.isnan(g) else f'{g:.6f}'}" for i, (xi, g) in enumerate(zip(x, gaps))
    ]
    Path(path).write_text("\n".join(rows) + "\n")


def to_gds(
    x: np.ndarray,
    path: str | Path,
    width: float = 0.5,
    length: float = 500.0,
    layer: int = 1,
    datatype: int = 0,
    cell: str = "OPA_EMITTERS",
    grid: float = 0.001,
) -> Path:
    """Emitter waveguides as rectangles ``width`` x ``length`` (µm) centred on each position, snapped to ``grid``.

    The emitters run along y, stacked across x at the optimised positions, as
    in a 1-D grating-emitter OPA. Needs ``gdstk`` (``pip install aperiodic-opa[gds]``).
    """
    import gdstk

    x = np.round(np.asarray(x, dtype=float) / grid) * grid
    lib = gdstk.Library(unit=1e-6, precision=grid * 1e-6)
    top = lib.new_cell(cell)
    for xi in x:
        top.add(gdstk.rectangle((xi - width / 2, 0.0), (xi + width / 2, length), layer=layer, datatype=datatype))
    path = Path(path)
    lib.write_gds(str(path))
    return path
