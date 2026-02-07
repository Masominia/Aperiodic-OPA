import numpy as np
import pytest

from aopa.layout_io import to_csv, to_gds


def test_csv_round_trip(tmp_path):
    x = np.array([0.0, 2.5, 6.0, 9.25])
    to_csv(x, tmp_path / "l.csv")
    rows = (tmp_path / "l.csv").read_text().strip().splitlines()
    assert rows[0] == "index,x_um,gap_um" and len(rows) == 5
    assert np.allclose([float(r.split(",")[1]) for r in rows[1:]], x)


def test_gds_has_one_emitter_per_position(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    x = np.array([0.0, 2.0, 5.0, 7.5])
    to_gds(x, tmp_path / "a.gds", width=0.5, length=100.0)
    cell = gdstk.read_gds(str(tmp_path / "a.gds")).top_level()[0]
    centres = sorted(np.mean(p.points[:, 0]) for p in cell.polygons)
    assert np.allclose(centres, x)
