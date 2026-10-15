PY ?= python
export PYTHONPATH := src

.PHONY: install data analyse report web all test lint

install:
	$(PY) -m pip install -r requirements.txt pytest ruff

data:            ## download the archived narratives and build a monthly sample
	$(PY) scripts/fetch_cfpb.py

analyse:         ## classifier, template letters, themes, urgency, companies
	$(PY) -m complaints

report:          ## figures + reports/findings.md
	$(PY) -m complaints.report

web:             ## docs/index.html with the in-browser classifier
	$(PY) -m complaints.web

all: analyse report web test

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts
