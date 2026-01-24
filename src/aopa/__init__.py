"""Minimax synthesis of aperiodic optical phased arrays over a full steering range."""

from .farfield import Evaluator, beam_metrics, steering_scan
from .layout_io import to_csv, to_gds
from .optimize import Result, search
from .patterns import Gaussian, Isotropic, Slit
from .spec import Spec, random_layout, uniform_layout

__all__ = [
    "Evaluator",
    "Gaussian",
    "Isotropic",
    "Result",
    "Slit",
    "Spec",
    "beam_metrics",
    "random_layout",
    "search",
    "steering_scan",
    "to_csv",
    "to_gds",
    "uniform_layout",
]
__version__ = "0.1.0"
