.PHONY: install install-colab install-dev smoke smoke-bootstrap test task2-source-audit task2-split task2-nuisance-source task2-pilot-fixed task2-pilot-uniform task2-pilots

DATA_ROOT ?= /content/hackathon_data
ARTIFACT_ROOT ?= artifacts

install:
	python -m pip install -r requirements.txt
	python -m pip install -e . --no-deps

install-colab:
	python -m pip install -r requirements-colab.txt
	python -m pip install -e . --no-deps

install-dev:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e . --no-deps

smoke:
	python scripts/smoke_check.py --config configs/colab.json

smoke-bootstrap:
	python scripts/smoke_check.py --config configs/colab.json --allow-missing-dependencies

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

task2-source-audit:
	python scripts/build_source_manifest.py --dataset-root $(DATA_ROOT)/raw/sid_set --manifest $(ARTIFACT_ROOT)/task2/source_manifest.csv --report $(ARTIFACT_ROOT)/task2/source_audit.json --expected-csv-rows 20000

task2-split:
	python scripts/assign_splits.py --input $(ARTIFACT_ROOT)/task2/source_manifest.csv --output $(ARTIFACT_ROOT)/task2/source_manifest_split.csv --report $(ARTIFACT_ROOT)/task2/split_report.json --seed 42

task2-nuisance-source:
	python scripts/audit_nuisance.py --manifest $(ARTIFACT_ROOT)/task2/source_manifest_split.csv --output $(ARTIFACT_ROOT)/task2/source_nuisance.json --seed 42

task2-pilot-fixed:
	python scripts/build_matched_clean.py --source-manifest $(ARTIFACT_ROOT)/task2/source_manifest_split.csv --output-root $(ARTIFACT_ROOT)/task2/matched_candidates --output-manifest $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv --report $(ARTIFACT_ROOT)/task2/fixed_q96_report.json --policy fixed_q96 --seed 42 --limit-per-label 1000
	python scripts/audit_nuisance.py --manifest $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv --output $(ARTIFACT_ROOT)/task2/fixed_q96_nuisance.json --seed 42

task2-pilot-uniform:
	python scripts/build_matched_clean.py --source-manifest $(ARTIFACT_ROOT)/task2/source_manifest_split.csv --output-root $(ARTIFACT_ROOT)/task2/matched_candidates --output-manifest $(ARTIFACT_ROOT)/task2/uniform_q95_q100_manifest.csv --report $(ARTIFACT_ROOT)/task2/uniform_q95_q100_report.json --policy uniform_q95_q100 --seed 42 --limit-per-label 1000
	python scripts/audit_nuisance.py --manifest $(ARTIFACT_ROOT)/task2/uniform_q95_q100_manifest.csv --output $(ARTIFACT_ROOT)/task2/uniform_q95_q100_nuisance.json --seed 42

task2-pilots: task2-pilot-fixed task2-pilot-uniform
