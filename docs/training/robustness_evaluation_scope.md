# Robustness Evaluation and Retraining Scope

## 1. Purpose

Tasks 1-8 are complete. The next milestone is to measure and improve robustness
before Task 9 model integration and Task 10 packaging.

This run answers three questions:

1. How well do the existing clean-trained CLIP and RINE checkpoints survive each
   independent Task 3 transformation?
2. Does controlled clean-or-one-transform retraining improve the locked 50/50
   score without causing unacceptable class-specific regressions?
3. Do the retained frequency magnitude/residual and Lab candidates add value to
   RINE under the same robustness protocol?

The competition `final_test` remains sealed throughout this milestone.

## 2. Fixed Evaluation Contract

- The task remains binary: `authentic` versus `ai_generated`.
- Mixed-origin, AI-edited, composited, inpainted-authentic, and ambiguous samples
  remain excluded.
- Every robustness image is derived directly from its fixed-Q96 matched-clean
  parent.
- Each robustness row contains exactly one transform and one parameter setting.
- Transforms are never chained, blended, or overlaid.
- Both labels use identical transform settings and sampling probabilities.
- All views derived from one `source_id` remain in the same split.
- Model and feature retention decisions use `selection_val` only.
- The primary selection score is:

  `locked_score = 0.50 x clean_accuracy + 0.50 x mean_independent_transform_accuracy`

- Per-class authentic accuracy (R.Acc.) and AI-generated accuracy (F.Acc.) remain
  mandatory guardrails.

## 3. Inputs and Preconditions

The run consumes the completed outputs of Tasks 1-8:

- the frozen fixed-Q96 matched-clean manifest and source-group splits;
- Task 3 deterministic transform configuration and provenance fields;
- existing frozen-CLIP Stage A and RINE Stage B checkpoints;
- the Task 5 evaluation and reporting contract;
- retained frequency magnitude/residual features from Task 7;
- retained Lab features from Task 8; and
- seeds `42`, `43`, and `44`.

Before GPU work starts:

1. Run the repository smoke check in the Colab runtime.
2. Verify the fixed-Q96 manifest hash and split counts.
3. Materialize or validate every Task 3 `selection_val` transform cell.
4. Confirm that every variant points directly to a matched-clean parent.
5. Confirm zero `final_test` rows are readable by training, selection, feature
   fitting, threshold selection, or calibration code.
6. Record the Git commit, configuration hash, dependency versions, device, and
   checkpoint hashes in the run metadata.

Any failed provenance, leakage, class-balance, or transform-independence check
stops the run before model scoring.

## 4. Robustness Cells

Evaluate clean data and every cell below separately. Do not average severities
before per-cell metrics have been recorded.

| Transform | Independent parameter cells |
|---|---|
| JPEG compression | quality 90, 70, 50, 30 |
| Gaussian blur | sigma 0.5, 1.0, 2.0 |
| Resize and restore | scale 0.5, 0.25 with locked bilinear settings |
| Gaussian noise | sigma 0.02, 0.05, 0.10 on normalized pixels |
| Color jitter | locked brightness, contrast, and saturation ranges |
| Center crop | 80% crop |

The robustness mean is the unweighted mean across the declared independent
cells. It must not be weighted by the number of materialized files in a cell.

## 5. Experiment Sequence

### Phase A - Evaluate existing checkpoints

Score the current clean-trained checkpoints without updating weights:

| Run | Training action | Purpose |
|---|---|---|
| Frozen CLIP Stage A | None | Mandatory clean and robustness baseline |
| Existing RINE Stage B | None | Measure whether the clean improvement survives redistribution |

This phase establishes the pre-retraining baseline and must finish before any
controlled-training result is used for comparison.

### Phase B - Controlled RINE retraining

Retrain only the RINE layer-importance projection and binary head with the CLIP
backbone frozen. Run seeds `42`, `43`, and `44`.

For every epoch, the Task 3 sampler must:

1. allocate 50% of draws to matched-clean views and 50% to transformed views;
2. balance `authentic` and `ai_generated` within both halves;
3. sample transformed draws uniformly across the declared transform cells; and
4. apply no undocumented augmentation after the selected view.

Compare controlled RINE against existing clean-trained RINE and frozen CLIP.

### Phase C - Incremental auxiliary fusion

Use the same splits, seeds, sampler, checkpoint-selection rule, and evaluation
cells for every candidate.

| Candidate | Trainable components | Run condition |
|---|---|---|
| RINE + frequency | Frequency projection and fusion head only | Always run after controlled RINE |
| RINE + Lab | Lab projection and fusion head only | Always run after controlled RINE |
| RINE + frequency + Lab | Both projections and fusion head only | Run only if each individual addition passes its retention gate |

The CLIP backbone, deterministic frequency extractor, and deterministic Lab
extractor remain frozen. Lab training must use identical color-jitter sampling
for both labels.

Do not rerun the dropped frequency phase representation, RGB-only vector, fixed
RGB+Lab concatenation, chromatic aberration, radial distortion, or a full CLIP
fine-tune. Task 8B remains a separate closed physical-signal track and contributes
no feature to this robustness matrix.

### Phase D - Select the pre-Task-9 model

Aggregate the three seeds only after preserving each seed-level report. Select a
candidate only when it:

- strictly improves the mean locked 50/50 score over its direct parent;
- does not reduce R.Acc. or F.Acc. by more than 1.0 percentage point;
- has no unexplained transform-cell collapse;
- preserves balanced label coverage and feature validity; and
- meets the recorded inference-cost budget for the pre-Task-9 pipeline.

If no auxiliary candidate passes, controlled RINE becomes the input to Task 9.
Complexity is not retained for a clean-only gain.

## 6. Required Metrics and Reports

Record the following for every model, seed, and evaluation cell:

- sample count and class count;
- overall accuracy;
- R.Acc. and F.Acc.;
- confusion matrix;
- balanced accuracy;
- loss and uncalibrated probability diagnostics;
- transform name and exact parameter;
- inference latency and peak memory where measurable; and
- input manifest, checkpoint, configuration, and code hashes.

Each candidate summary must include:

- clean accuracy;
- mean independent-transform accuracy;
- locked 50/50 score;
- worst-cell accuracy and its transform;
- mean and standard deviation across seeds;
- per-class deltas against its direct parent; and
- a final `retain`, `reject`, or `blocked` decision with the gate evidence.

Calibration is diagnostic during this milestone. Final temperature fitting and
public-output calibration belong to Task 10 after the architecture is frozen.

## 7. Artifact Layout

Use a new robustness namespace so previous clean milestones are immutable:

```text
artifacts/robustness/
  manifests/
    selection_transform_manifest.csv
    transform_validation_report.json
  existing_stage_a/seed_<seed>/
  existing_rine/seed_<seed>/
  controlled_rine/seed_<seed>/
  rine_frequency/seed_<seed>/
  rine_lab/seed_<seed>/
  rine_frequency_lab/seed_<seed>/
  reports/
    seed_comparison.csv
    cell_comparison.csv
    retention_decisions.json
    robustness_summary.json
```

Every run directory must contain resolved configuration, environment metadata,
checkpoint metadata, predictions, per-cell metrics, and a completion marker.
Write the completion marker only after all expected files pass validation and
are copied to durable Drive storage.

Resume logic may skip only a run whose completion marker and input hashes match.
Partial or hash-mismatched runs must be rerun into a new directory, not silently
combined with earlier output.

## 8. Execution Order and Stop Conditions

Run in this order:

1. validate the transform manifest and sealed-split boundary;
2. evaluate existing Stage A and RINE checkpoints;
3. retrain and evaluate controlled RINE for all three seeds;
4. train and evaluate RINE + frequency;
5. train and evaluate RINE + Lab;
6. conditionally train and evaluate the combined auxiliary model;
7. generate the seed-level and aggregate retention reports; and
8. freeze the selected pre-Task-9 checkpoint and evidence bundle.

Stop and investigate if:

- a transform cell is missing, chained, or derived from another robustness row;
- the two labels receive different transform distributions;
- a source crosses split boundaries;
- any selection process reads `final_test`;
- a run cannot reproduce its manifest or configuration hash;
- a feature-validity mask becomes label-dependent; or
- results are incomplete for any required seed or transform cell.

## 9. Handoff to Tasks 9 and 10

This milestone is complete when:

- all required model/seed/cell combinations have validated reports;
- the clean and robustness metrics are available separately;
- the locked 50/50 score is calculated from the declared cells;
- every candidate has an evidence-backed retention decision;
- one pre-Task-9 checkpoint is frozen; and
- the competition `final_test` remains unopened.

Task 9 compares the selected global model with global-plus-patch aggregation
under the same robustness protocol and a fixed latency budget. Notebook 08 has
completed the reference-free PRNU-v2 readiness and binary usefulness gate;
fusion was rejected after two of three seeds collapsed and no combined
texture+PRNU candidate is authorized. Task 10 begins after Task 9 records its
retention decision; it owns
calibration, packaging, the directory inference contract, and the one-time
sealed final-test execution.

## 10. Implemented Execution Interface

The repository exposes the milestone through `notebooks/07_robustness_rerun.ipynb`
and these Make targets:

```text
make robustness-test
make robustness-prepare
make robustness-stage-a-evaluate ROBUSTNESS_SEED=42 STAGE_A_CHECKPOINT=<path>
make robustness-rine-evaluate ROBUSTNESS_SEED=42 RINE_CHECKPOINT=<path>
make robustness-rine-train ROBUSTNESS_SEED=42
make robustness-frequency-extract
make robustness-lab-extract
make robustness-fusion-train ROBUSTNESS_FUSION_VARIANT=frequency ROBUSTNESS_SEED=42
make robustness-fusion-train ROBUSTNESS_FUSION_VARIANT=lab ROBUSTNESS_SEED=42
```

Run the model commands for seeds 42, 43, and 44. The combined
`frequency_lab` fusion target is allowed only after the individual frequency and
Lab comparison reports both say `retain`. Use
`scripts/compare_robustness_candidate.py` to apply the locked improvement and
per-class regression gates.

`robustness-prepare` first creates a development-only parent manifest containing
`seed_train` and `selection_val`; only then does it materialize Task 3 views. The
combined extraction manifest therefore contains no `final_test` rows. The Lab
target runs the color extractor only, avoiding unnecessary PRNU and optics work.

Local verification covers the contracts, all 14 cells, the controlled sampler,
JSON-cell reporting, frozen-parent fusion, and a real one-epoch synthetic
cached-feature run that writes a `best_50_50.pt` checkpoint. Full SID_Set
materialization, CLIP/RINE extraction, and three-seed metrics require the primary
fixed-Q96 data and clean checkpoints in the prepared Colab/Drive runtime; those
artifacts are intentionally not stored in this checkout.

## 11. Out of Scope

- Task 9 patch aggregation implementation or training;
- Task 10 calibration, packaging, and sealed final-test execution;
- new datasets or changed split assignments;
- chained or adversarial transformations;
- mixed-origin or AI-edited classification;
- PRNU or optical-feature reopening;
- self-training or pseudo-labeling; and
- full-backbone CLIP fine-tuning.
