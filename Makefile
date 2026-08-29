.PHONY: install install-colab install-dev smoke smoke-bootstrap test task2-source-audit task2-split task2-nuisance-source task2-pilot-fixed task2-pilot-uniform task2-pilots task4-stage-a-fixed task4-stage-a-uniform task4-stage-a-pilots task4-compare task5-evaluate task6-rine task6-compare task7-extract task7-train task7-compare task8-extract task8-train task8-compare

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

TASK4_SEED ?= 42
TASK4_BATCH_SIZE ?= 8
PREDICTIONS ?= artifacts/task4/fixed_q96/seed_42/best_clean_predictions.csv
EVALUATION_OUTPUT ?= artifacts/task5/fixed_q96/seed_42

task4-stage-a-fixed:
	python scripts/train_clip_baseline.py --manifest $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv --matching-policy fixed_q96 --output-root $(ARTIFACT_ROOT)/task4 --cache-root /content/clip_embedding_cache --seed $(TASK4_SEED) --physical-batch-size $(TASK4_BATCH_SIZE)

task4-stage-a-uniform:
	python scripts/train_clip_baseline.py --manifest $(ARTIFACT_ROOT)/task2/uniform_q95_q100_manifest.csv --matching-policy uniform_q95_q100 --output-root $(ARTIFACT_ROOT)/task4 --cache-root /content/clip_embedding_cache --seed $(TASK4_SEED) --physical-batch-size $(TASK4_BATCH_SIZE)

task4-stage-a-pilots: task4-stage-a-fixed task4-stage-a-uniform

task4-compare:
	python scripts/compare_stage_a.py --task4-root $(ARTIFACT_ROOT)/task4 --output $(ARTIFACT_ROOT)/task4/policy_comparison.json

task5-evaluate:
	python scripts/evaluate_predictions.py --predictions $(PREDICTIONS) --output $(EVALUATION_OUTPUT)

task6-rine:
	python scripts/train_rine_baseline.py --manifest $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv --matching-policy fixed_q96 --output-root $(ARTIFACT_ROOT)/task6 --cache-root /content/rine_feature_cache --seed $(TASK4_SEED) --physical-batch-size 4

task6-compare:
	python scripts/compare_stage_a_rine.py --stage-a-root $(ARTIFACT_ROOT)/task4 --stage-b-root $(ARTIFACT_ROOT)/task6 --output $(ARTIFACT_ROOT)/task6/stage_a_vs_rine.json

FREQUENCY_VARIANT ?= magnitude

task7-extract:
	python scripts/extract_frequency_features.py --manifest $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv --output $(ARTIFACT_ROOT)/task7/frequency_features.csv --report $(ARTIFACT_ROOT)/task7/extraction_report.json --cache-root /content/frequency_feature_cache --matching-policy fixed_q96 --workers 4

task7-train:
	python scripts/train_frequency_baseline.py --features $(ARTIFACT_ROOT)/task7/frequency_features.csv --output $(ARTIFACT_ROOT)/task7/$(FREQUENCY_VARIANT)/seed_$(TASK4_SEED) --variant $(FREQUENCY_VARIANT) --seed $(TASK4_SEED)

task7-compare:
	python scripts/compare_frequency_variants.py --task7-root $(ARTIFACT_ROOT)/task7 --output $(ARTIFACT_ROOT)/task7/variant_comparison.json

AUXILIARY_VARIANT ?= rgb

task8-extract:
	python scripts/extract_auxiliary_features.py --manifest $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv --output $(ARTIFACT_ROOT)/task8/auxiliary_features.csv --report $(ARTIFACT_ROOT)/task8/extraction_report.json --cache-root /content/auxiliary_feature_cache --matching-policy fixed_q96

task8-train:
	python scripts/train_auxiliary_baseline.py --features $(ARTIFACT_ROOT)/task8/auxiliary_features.csv --output $(ARTIFACT_ROOT)/task8/$(AUXILIARY_VARIANT)/seed_$(TASK4_SEED) --variant $(AUXILIARY_VARIANT) --seed $(TASK4_SEED)

task8-compare:
	python scripts/compare_color_variants.py --task8-root $(ARTIFACT_ROOT)/task8 --output $(ARTIFACT_ROOT)/task8/color_comparison.json
