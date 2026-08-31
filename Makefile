.PHONY: install install-colab install-dev smoke smoke-bootstrap test task2-source-audit task2-split task2-nuisance-source task2-pilot-fixed task2-pilot-uniform task2-pilots task3-test task3-fixture task4-stage-a-fixed task4-stage-a-uniform task4-stage-a-pilots task4-compare task5-evaluate task6-rine task6-compare task7-extract task7-train task7-compare task8-extract task8-train task8-compare task8b-extract-genimage task8b-inventory task8b-manifest task8b-readiness task8b-matched task8b-prepare task8b-prnu-references task8b-prnu-validate task8b-decision task8b-v2-prnu-validate robustness-test robustness-prepare robustness-stage-a-evaluate robustness-rine-evaluate robustness-rine-train robustness-frequency-extract robustness-lab-extract robustness-fusion-train robustness-prnu-v2-extract robustness-prnu-v2-train robustness-prnu-v2-fusion robustness-prnu-v2-compare task9-test task9-extract task9-run task9-matrix task9-compare task9-robustness-test task9-robustness-materialize task9-robustness-evaluate task9-robustness-compare

DATA_ROOT ?= /content/hackathon_data
ARTIFACT_ROOT ?= artifacts
# Override after Task 2 selects a matched-clean policy; fixed Q96 is the practical fixture default.
TASK2_SELECTED_MANIFEST ?= $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv

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

task3-test:
	PYTHONPATH=src python -m unittest tests.test_config tests.test_benchmark_transforms tests.test_transform_materialization tests.test_preprocessing tests.test_controlled_sampler tests.test_safe_transforms -v

task3-fixture:
	python scripts/materialize_transforms.py --input-manifest $(TASK2_SELECTED_MANIFEST) --output-root $(ARTIFACT_ROOT)/task3/variants --output-manifest $(ARTIFACT_ROOT)/task3/transform_manifest.csv --report $(ARTIFACT_ROOT)/task3/transform_report.json --config configs/colab.json

ROBUSTNESS_ROOT ?= $(ARTIFACT_ROOT)/robustness
ROBUSTNESS_CLEAN_MANIFEST ?= $(ROBUSTNESS_ROOT)/manifests/dev_clean_manifest.csv
ROBUSTNESS_TRANSFORM_MANIFEST ?= $(ROBUSTNESS_ROOT)/manifests/transform_manifest.csv
ROBUSTNESS_COMBINED_MANIFEST ?= $(ROBUSTNESS_ROOT)/manifests/combined_manifest.csv
ROBUSTNESS_SEED ?= 42
ROBUSTNESS_BATCH_SIZE ?= 4
STAGE_A_CHECKPOINT ?=
RINE_CHECKPOINT ?=
CONTROLLED_RINE_CHECKPOINT ?= $(ROBUSTNESS_ROOT)/train-controlled-rine/seed_$(ROBUSTNESS_SEED)/best_50_50.pt
ROBUSTNESS_FUSION_VARIANT ?= frequency
PRNU_V2_RUNTIME_TABLE ?= $(ROBUSTNESS_ROOT)/features/prnu_v2_runtime_features.csv
PRNU_V2_CROP_SIZE ?= 256

robustness-test:
	PYTHONPATH=src python -m unittest tests.test_robustness_training tests.test_prnu_runtime_v2 tests.test_evaluation tests.test_controlled_sampler -v

robustness-prepare:
	python scripts/prepare_robustness_manifest.py --input-manifest $(TASK2_SELECTED_MANIFEST) --output-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --report $(ROBUSTNESS_ROOT)/manifests/clean_manifest_report.json
	python scripts/materialize_transforms.py --input-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --output-root $(ROBUSTNESS_ROOT)/variants --output-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --report $(ROBUSTNESS_ROOT)/manifests/transform_validation_report.json --config configs/colab.json
	python scripts/combine_robustness_manifest.py --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --output-manifest $(ROBUSTNESS_COMBINED_MANIFEST) --report $(ROBUSTNESS_ROOT)/manifests/combined_manifest_report.json --config configs/colab.json

robustness-stage-a-evaluate:
	python scripts/run_robustness.py evaluate-stage-a --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --checkpoint $(STAGE_A_CHECKPOINT) --output-root $(ROBUSTNESS_ROOT) --seed $(ROBUSTNESS_SEED) --physical-batch-size $(ROBUSTNESS_BATCH_SIZE)

robustness-rine-evaluate:
	python scripts/run_robustness.py evaluate-rine --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --checkpoint $(RINE_CHECKPOINT) --output-root $(ROBUSTNESS_ROOT) --seed $(ROBUSTNESS_SEED) --physical-batch-size $(ROBUSTNESS_BATCH_SIZE)

robustness-rine-train:
	python scripts/run_robustness.py train-controlled-rine --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --output-root $(ROBUSTNESS_ROOT) --seed $(ROBUSTNESS_SEED) --physical-batch-size $(ROBUSTNESS_BATCH_SIZE)

robustness-frequency-extract:
	python scripts/extract_frequency_features.py --manifest $(ROBUSTNESS_COMBINED_MANIFEST) --output $(ROBUSTNESS_ROOT)/features/frequency_features.csv --report $(ROBUSTNESS_ROOT)/features/frequency_extraction_report.json --cache-root /content/robustness_frequency_cache --matching-policy fixed_q96 --workers 4

robustness-lab-extract:
	python scripts/extract_auxiliary_features.py --manifest $(ROBUSTNESS_COMBINED_MANIFEST) --output $(ROBUSTNESS_ROOT)/features/auxiliary_features.csv --report $(ROBUSTNESS_ROOT)/features/auxiliary_extraction_report.json --cache-root /content/robustness_auxiliary_cache --matching-policy fixed_q96 --families color

robustness-fusion-train:
	python scripts/run_robustness_fusion.py --variant $(ROBUSTNESS_FUSION_VARIANT) --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --parent-checkpoint $(CONTROLLED_RINE_CHECKPOINT) --frequency-table $(ROBUSTNESS_ROOT)/features/frequency_features.csv --auxiliary-table $(ROBUSTNESS_ROOT)/features/auxiliary_features.csv --output-root $(ROBUSTNESS_ROOT) --seed $(ROBUSTNESS_SEED) --physical-batch-size $(ROBUSTNESS_BATCH_SIZE)

robustness-prnu-v2-extract:
	python scripts/extract_prnu_runtime_v2.py --manifest $(ROBUSTNESS_COMBINED_MANIFEST) --output $(PRNU_V2_RUNTIME_TABLE) --report $(ROBUSTNESS_ROOT)/features/prnu_v2_runtime_extraction_report.json --cache-root /content/robustness_prnu_v2_cache --matching-policy fixed_q96

robustness-prnu-v2-train:
	python scripts/train_prnu_runtime_v2.py --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --features $(PRNU_V2_RUNTIME_TABLE) --output-root $(ROBUSTNESS_ROOT) --seed $(ROBUSTNESS_SEED)

robustness-prnu-v2-fusion:
	python scripts/run_robustness_fusion.py --variant prnu_v2 --clean-manifest $(ROBUSTNESS_CLEAN_MANIFEST) --transform-manifest $(ROBUSTNESS_TRANSFORM_MANIFEST) --parent-checkpoint $(CONTROLLED_RINE_CHECKPOINT) --prnu-table $(PRNU_V2_RUNTIME_TABLE) --output-root $(ROBUSTNESS_ROOT) --seed $(ROBUSTNESS_SEED) --physical-batch-size $(ROBUSTNESS_BATCH_SIZE)

robustness-prnu-v2-compare:
	python scripts/compare_robustness_candidate.py --parent-root $(ROBUSTNESS_ROOT)/train-controlled-rine --candidate-root $(ROBUSTNESS_ROOT)/rine_prnu_v2 --candidate-name rine_prnu_v2 --output $(ROBUSTNESS_ROOT)/reports/prnu_v2

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

TASK8B_DATA_ROOT ?= $(DATA_ROOT)/raw/task8b
TASK8B_ARTIFACT_ROOT ?= $(ARTIFACT_ROOT)/task8b
TASK8B_V2_ARTIFACT_ROOT ?= $(ARTIFACT_ROOT)/task8b_v2
TASK8B_MANIFEST ?= $(TASK8B_ARTIFACT_ROOT)/manifests/source_manifest_split.csv
GENIMAGE_ARCHIVE ?=
GENIMAGE_GENERATOR ?=
GENIMAGE_LIMIT ?= 200

task8b-extract-genimage:
	python scripts/extract_task8b_genimage_sample.py --archive "$(GENIMAGE_ARCHIVE)" --generator "$(GENIMAGE_GENERATOR)" --output-root $(TASK8B_DATA_ROOT)/genimage_ai --report "$(TASK8B_ARTIFACT_ROOT)/audits/extract_$(GENIMAGE_GENERATOR).json" --limit $(GENIMAGE_LIMIT)

task8b-inventory:
	python scripts/prepare_task8b_inventory.py --task8b-root $(TASK8B_DATA_ROOT) --output $(TASK8B_DATA_ROOT)/sources.csv --report $(TASK8B_ARTIFACT_ROOT)/audits/inventory_preparation.json --generators ADM GLIDE Midjourney Wukong

task8b-manifest:
	python scripts/build_task8b_manifest.py --dataset-root $(TASK8B_DATA_ROOT) --inventory $(TASK8B_DATA_ROOT)/sources.csv --manifest $(TASK8B_MANIFEST) --audit-report $(TASK8B_ARTIFACT_ROOT)/audits/source_audit.json --split-report $(TASK8B_ARTIFACT_ROOT)/audits/split_report.json

task8b-readiness:
	python scripts/audit_task8b_readiness.py --manifest $(TASK8B_MANIFEST) --output $(TASK8B_ARTIFACT_ROOT)/audits/readiness_report.json --require-source-ready

task8b-matched:
	python scripts/build_task8b_matched_views.py --source-manifest $(TASK8B_MANIFEST) --output-root $(TASK8B_ARTIFACT_ROOT)/matched_views/images --output-manifest $(TASK8B_ARTIFACT_ROOT)/manifests/matched_manifest.csv --report $(TASK8B_ARTIFACT_ROOT)/audits/matched_view_report.json --size 256 --perceptual-distance 1 --overwrite
	python scripts/audit_task8b_readiness.py --manifest $(TASK8B_ARTIFACT_ROOT)/manifests/matched_manifest.csv --output $(TASK8B_ARTIFACT_ROOT)/audits/matched_readiness_report.json --require-training-ready

task8b-prepare: task8b-manifest task8b-readiness

task8b-prnu-references: task8b-readiness
	python scripts/build_task8b_prnu_references.py --manifest $(TASK8B_MANIFEST) --output-root $(TASK8B_ARTIFACT_ROOT)/fingerprints --report $(TASK8B_ARTIFACT_ROOT)/audits/prnu_reference_report.json

task8b-prnu-validate:
	python scripts/validate_task8b_prnu.py --manifest $(TASK8B_MANIFEST) --output $(TASK8B_ARTIFACT_ROOT)/audits/prnu_signal_validation.json

task8b-v2-prnu-validate:
	python scripts/validate_task8b_prnu_v2.py --manifest $(TASK8B_MANIFEST) --artifact-root $(TASK8B_V2_ARTIFACT_ROOT) --reference-images-per-device 25 --crop-size $(PRNU_V2_CROP_SIZE) --wavelet db2 --wavelet-levels 4 --edge-keep-quantile 0.75 --maximum-shift 8 --minimum-auc 0.60

task8b-decision:
	python scripts/decide_task8b.py --matched-readiness $(TASK8B_ARTIFACT_ROOT)/audits/matched_readiness_report.json --prnu-validation $(TASK8B_ARTIFACT_ROOT)/audits/prnu_signal_validation.json --output $(TASK8B_ARTIFACT_ROOT)/reports/retention_decision.json

TASK9_MANIFEST ?= $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv
TASK9_OUTPUT_ROOT ?= $(ARTIFACT_ROOT)/task9
TASK9_GLOBAL_CACHE ?= /content/rine_feature_cache
TASK9_PATCH_CACHE ?= /content/texture_patch_cache
TASK9_CACHE_PAYLOAD ?= /content/texture_patch_cache/task9_cache_payload.json
TASK9_VARIANT ?= global_only
TASK9_SEED ?= 42
TASK9_DEVICE ?= cpu

task9-test:
	PYTHONPATH=src python -m unittest tests.test_config tests.test_features_texture tests.test_texture_model tests.test_texture_extraction tests.test_texture_training tests.test_texture_gate -v

# Frozen global/patch feature extraction only needs to run once; every one of
# the nine variant/seed training runs below reuses the same cached tensors.
task9-extract:
	python scripts/extract_texture_features.py --manifest $(TASK9_MANIFEST) --cache-payload $(TASK9_CACHE_PAYLOAD) --global-cache-root $(TASK9_GLOBAL_CACHE) --patch-cache-root $(TASK9_PATCH_CACHE)

task9-run: task9-extract
	python scripts/train_texture_pilot.py --cached-features $(TASK9_CACHE_PAYLOAD) --variant $(TASK9_VARIANT) --seed $(TASK9_SEED) --output-root $(TASK9_OUTPUT_ROOT) --device $(TASK9_DEVICE)

# Explicit sequential loop over every configured variant and seed. This never
# runs the nine trainings concurrently: simultaneous GPU extraction/training
# would duplicate frozen-CLIP memory on the shared Colab GPU and race on the
# shared /content feature caches.
task9-matrix: task9-extract
	for variant in global_only local_only global_local; do \
		for seed in 42 43 44; do \
			python scripts/train_texture_pilot.py --cached-features $(TASK9_CACHE_PAYLOAD) --variant $$variant --seed $$seed --output-root $(TASK9_OUTPUT_ROOT) --device $(TASK9_DEVICE); \
		done; \
	done

# Only meaningful after all nine task9-matrix runs have completed.
task9-compare:
	python scripts/compare_texture_pilot.py --output-root $(TASK9_OUTPUT_ROOT)

TASK9_ROBUSTNESS_INPUT_MANIFEST ?= $(TASK9_MANIFEST)
TASK9_ROBUSTNESS_IMAGE_ROOT ?= /content/task9_robustness_stage1/images
TASK9_ROBUSTNESS_OUTPUT_ROOT ?= $(ARTIFACT_ROOT)/task9/clean_pilot_v1/robustness_stage1_v1
TASK9_ROBUSTNESS_MANIFEST ?= $(TASK9_ROBUSTNESS_OUTPUT_ROOT)/materialization/transformed_selection_val.csv
TASK9_ROBUSTNESS_REPORT ?= $(TASK9_ROBUSTNESS_OUTPUT_ROOT)/materialization/materialization_report.json
TASK9_ROBUSTNESS_CACHE_ROOT ?= /content/task9_robustness_stage1/cache
TASK9_ROBUSTNESS_PREDICTIONS_ROOT ?= $(TASK9_ROBUSTNESS_OUTPUT_ROOT)/predictions
TASK9_ROBUSTNESS_CLEAN_ROOT ?= $(TASK9_OUTPUT_ROOT)/clean_pilot_v1
TASK9_ROBUSTNESS_CONTROLLED_RINE_ROOT ?= $(ROBUSTNESS_ROOT)/train-controlled-rine
TASK9_ROBUSTNESS_DEVICE ?= cpu

task9-robustness-test:
	PYTHONPATH=src python -m unittest tests.test_texture_robustness -v

task9-robustness-materialize:
	python scripts/materialize_texture_robustness.py --input-manifest $(TASK9_ROBUSTNESS_INPUT_MANIFEST) --output-root $(TASK9_ROBUSTNESS_IMAGE_ROOT) --output-manifest $(TASK9_ROBUSTNESS_MANIFEST) --report $(TASK9_ROBUSTNESS_REPORT)

# Only meaningful after task9-robustness-materialize and all nine clean
# task9-matrix runs have completed.
task9-robustness-evaluate:
	python scripts/evaluate_texture_robustness.py --transformed-manifest $(TASK9_ROBUSTNESS_MANIFEST) --materialization-report $(TASK9_ROBUSTNESS_REPORT) --clean-experiment-root $(TASK9_ROBUSTNESS_CLEAN_ROOT) --cache-root $(TASK9_ROBUSTNESS_CACHE_ROOT) --output-root $(TASK9_ROBUSTNESS_PREDICTIONS_ROOT) --device $(TASK9_ROBUSTNESS_DEVICE)

# Only meaningful after task9-robustness-evaluate has produced all 81
# prediction slices and the retained controlled-RINE artifacts are restored
# under TASK9_ROBUSTNESS_CONTROLLED_RINE_ROOT.
task9-robustness-compare:
	python scripts/compare_texture_robustness.py --clean-experiment-root $(TASK9_ROBUSTNESS_CLEAN_ROOT) --robustness-root $(TASK9_ROBUSTNESS_PREDICTIONS_ROOT) --controlled-rine-root $(TASK9_ROBUSTNESS_CONTROLLED_RINE_ROOT) --output-root $(TASK9_ROBUSTNESS_OUTPUT_ROOT)
