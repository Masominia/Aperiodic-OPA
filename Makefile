PY ?= python
export OMP_NUM_THREADS = 1

.PHONY: install test lint benchmarks baselines channels band planar robustness report explorer all

install:
	$(PY) -m pip install -e ".[dev,gds]"

test:
	$(PY) -m pytest

lint:
	ruff check src scripts tests examples
	ruff format --check src scripts tests examples

benchmarks:
	$(PY) scripts/run_benchmarks.py

baselines:
	$(PY) scripts/run_baselines.py zang2025-32 zang2025-64 --seeds 3 --seconds 900

channels:
	$(PY) scripts/run_channels.py --seconds 600

band:
	$(PY) scripts/run_benchmarks.py zang2025-32 zang2025-64 --band 1.50 1.55 1.60 --seconds 600 --out results/band

planar:
	$(PY) scripts/run_planar.py --seconds 3600

robustness:
	$(PY) scripts/run_robustness.py

report:
	$(PY) scripts/make_report.py

explorer:
	$(PY) scripts/export_explorer.py

all: benchmarks baselines channels band planar robustness report explorer
