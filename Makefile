# Development targets for DiGiTerra. Run from the project root.
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install dev lint lint-prose test check build-bundle

install:
	$(PIP) install -r requirements.txt

dev: install
	$(PIP) install -r requirements-dev.txt
	npm ci

lint:
	npm run lint
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) scripts/check_language.py --standard a

# Standard B, the prose rules, reports 34 items and does not gate yet.
lint-prose:
	$(PYTHON) scripts/check_language.py --standard b

test:
	MPLBACKEND=Agg $(PYTHON) -m pytest tests/ -q

check: lint test
	npm run check-bundle
	$(PYTHON) scripts/check_requirements.py

build-bundle:
	npm run build
