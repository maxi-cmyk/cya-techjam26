# Task 9 Texture Robustness Stage 1 Design

## Status

Approved design for the continuation authorized by the real Task 9 clean gate. The clean pilot remains immutable at commit `c14702b`; its verified decision is `continue_to_robustness_design`.

This specification defines a frozen-checkpoint, inference-only robustness screen. It does not authorize transformed training, calibration changes, final-test access, or final model retention.

## Objective

Determine whether the clean-pilot `global_local` texture head preserves an advantage over its internal `global_only` ablation and improves on the retained controlled-RINE parent under the texture-sensitive subset of the existing Task 3 benchmark contract, without hiding class-specific or single-cell regressions.

The experiment evaluates all clean-trained variants and seeds:

- variants: `global_only`, `local_only`, `global_local`
- seeds: `42`, `43`, `44`

The retained controlled-RINE run is the authoritative parent comparator. `global_local` must beat both controlled RINE and its internal `global_only` ablation; `local_only` is diagnostic. Controlled RINE has `1.0` clean accuracy and a `0.9981` locked 50/50 score across the full 14-cell Task 3 matrix. Stage 1 must recompute its nine-cell subset from persisted per-sample predictions instead of substituting that full-matrix aggregate.

## Data and leakage boundary

The sole input population is the fixed-Q96 matched-clean `selection_val` split used by the clean pilot. The continuation must reject `seed_train`, `self_train_pool`, `final_test`, source-original, benchmark-derived, Task 8B, or otherwise nonconforming rows before writing outputs.

Every transformed row must derive directly from its matched-clean parent through the existing deterministic Task 3 engine. Transform chaining is forbidden. Parent identity, label, split, clean hash, transform parameters, transform seed, storage format, and output hash must remain auditable.

Existing best-clean checkpoints are read-only. Training, fine-tuning, threshold changes, and recalibration are forbidden. The clean calibration and threshold are reused unchanged.

## Staged transform matrix

Stage 1 contains exactly these nine predeclared Task 3 cells:

| Family | Cell IDs |
|---|---|
| JPEG | `jpeg_q90`, `jpeg_q70`, `jpeg_q50`, `jpeg_q30` |
| Gaussian blur | `blur_sigma_0.5`, `blur_sigma_1.0`, `blur_sigma_2.0` |
| Resize | `resize_scale_0.5`, `resize_scale_0.25` |

Noise, color jitter, and center crop are deferred. A Stage-1 pass authorizes their evaluation in a later full-robustness continuation; it does not retain texture by itself.

## Architecture and data flow

1. Restore the fixed-Q96 Task 2 bundle, all nine completed clean Task 9 runs, and the hash-verified three-seed controlled-RINE prediction artifacts from the completed Task 3 robustness evaluation.
2. Select and validate only matched-clean `selection_val` parents.
3. Materialize the nine Stage-1 cells directly from clean parents using the existing Task 3 materializer.
4. Verify cell counts, labels, sample/parent identity, hashes, split integrity, and the no-chaining contract.
5. Recompute the Laplacian/Sobel energy map and up to four non-overlapping 112-pixel patch positions from each transformed image. Clean-image patch coordinates must not be reused.
6. Extract one frozen global representation and the transformed image's patch representations once per transformed image.
7. Bind feature caches to the transformed image hash, parent hash, cell contract, resolved model revision, preprocessing version, texture extractor version, matching policy, patch size, and patch count.
8. Reuse the immutable transformed feature caches across variants and seeds.
9. Load each existing best-clean checkpoint without mutation and emit predictions for every variant/seed/cell slice.
10. Require all `9 cells × 3 variants × 3 seeds = 81` slices before comparison.
11. Require the 27 controlled-RINE seed-cell partitions for the same nine cells and three seeds from the three persisted per-seed `predictions.csv` files, then compare `global_local` with both controlled RINE and `global_only`, report `local_only` diagnostically, apply the predeclared gate, and publish a hashed completion manifest last.

Materializing transformed images before extraction is intentional: it uses the existing provenance contract, permits independent verification, and makes interruption recovery deterministic. Large images and feature caches stay under `/content`; durable predictions and reports are copied to Drive.

## Artifact contract

Publish under a new subtree without changing the clean experiment:

```text
artifacts/task9/clean_pilot_v1/robustness_stage1_v1/
├── materialization/
│   ├── transformed_selection_val.csv
│   └── materialization_report.json
├── predictions/
│   └── <variant>/seed_<seed>/<cell_id>.csv
├── reports/
│   ├── per_cell_metrics.csv
│   ├── per_seed_robustness.csv
│   ├── robustness_comparison.json
│   ├── latency_and_memory.json
│   └── failure_analysis.csv
└── metadata/
    └── artifact_manifest.json
```

Each prediction row records sample ID, parent ID, label, cell ID and parameters, variant, seed, logit, probability, prediction, paired clean prediction, checkpoint hash, and input/feature provenance hashes.

`artifact_manifest.json` is the completion marker and is published last. It records the experiment status and decision plus hashes for every durable input and output required to reproduce the decision.

The manifest also records the source root and hashes of the controlled-RINE prediction and metric artifacts. Comparator artifacts remain read-only and are not republished as Task 9 outputs.

## Metrics

For every variant, seed, and transform cell, report:

- overall accuracy;
- authentic accuracy and false-positive rate;
- AI-generated accuracy and false-negative rate;
- confusion matrix;
- expected calibration error under unchanged clean calibration;
- corrected and introduced errors relative to both `global_only` and controlled RINE;
- inference latency and peak GPU memory;
- patch availability and selected-patch stability diagnostics.

Cells receive equal macro weight:

```text
clean_accuracy(variant) = mean clean accuracy across seeds

robustness_accuracy(variant) =
    mean accuracy across the 9 cells and 3 seeds

locked_50_50_score(variant) =
    0.5 * clean_accuracy(variant)
    + 0.5 * robustness_accuracy(variant)
```

Report deterministic paired bootstrap confidence intervals for accuracy deltas. Confidence intervals are diagnostic; the predeclared inequalities below control the decision.

## Decision gate

Emit `retain_texture_for_full_robustness` only when every condition holds:

```text
global_local locked_50_50_score > global_only locked_50_50_score

global_local robustness_accuracy >= global_only robustness_accuracy

global_local locked_50_50_score > controlled_rine locked_50_50_score

global_local robustness_accuracy >= controlled_rine robustness_accuracy

mean robust authentic accuracy delta versus each comparator >= -0.01

mean robust AI-generated accuracy delta versus each comparator >= -0.01

for every transform cell, after averaging across seeds:
    overall accuracy delta versus each comparator >= -0.03
    authentic accuracy delta versus each comparator >= -0.03
    AI-generated accuracy delta versus each comparator >= -0.03
```

Otherwise emit `reject_texture_robustness_stage1` and retain controlled RINE without the texture addition. Missing, mismatched, or unverifiable controlled-RINE artifacts are a blocked prerequisite, never a texture pass or rejection.

A pass authorizes a separately controlled evaluation of the remaining Task 3 noise, color-jitter, and crop cells. It does not authorize final retention, calibration changes, transformed training, packaging, or final-test evaluation.

## Recovery and failure behavior

- Publish prediction slices atomically.
- Skip a completed slice only when its complete artifact set and hashes validate against the exact experiment contract.
- Recompute missing, malformed, partial, non-finite, or hash-mismatched slices.
- Do not run comparison until all 81 texture slices and all 27 controlled-RINE seed-cell partitions validate and sample/label sets align.
- Reject any clean/transformed parent mismatch, forbidden split, transform-chain evidence, checkpoint mismatch, model/preprocessing mismatch, or cache-contract mismatch.
- Reject non-finite features, logits, probabilities, metrics, latency, or memory measurements.
- Never treat a partially copied Drive directory as complete.
- A fresh Colab runtime may rematerialize images and features, restore verified prediction slices from Drive, and resume remaining slices.

## Verification requirements

Automated tests must prove:

- only fixed-Q96 matched-clean `selection_val` parents are accepted;
- exactly the nine approved cell IDs are present and each derives directly from clean;
- transforms and transformed-image patch selection are deterministic;
- clean patch coordinates cannot be reused;
- checkpoints and clean artifacts remain unchanged;
- feature-cache identity includes every material transform/model/texture contract field;
- all 81 texture slices and 27 controlled-RINE seed-cell partitions are required and aligned before comparison;
- the 50/50 score uses equal macro weighting across cells and seeds;
- the controlled-RINE nine-cell score is recomputed from hash-verified per-sample predictions rather than copied from its 14-cell aggregate;
- improvement, aggregate class tolerances, and each worst-cell condition fail independently against either controlling comparator;
- resume accepts only hash-verified completed slices;
- corrupt, partial, mismatched, non-finite, and forbidden-split inputs fail closed;
- no normal continuation path can read `final_test`;
- a dependency-free fixture smoke covers materialization, extraction, frozen evaluation, comparison, and atomic publication.

The Colab launcher must remain thin: it may stage inputs, invoke repository scripts, manage `/content` caches, resume verified work, and copy completed artifacts to Drive, but it must not inline transform, model, inference, metric, or gate logic.

## Deferred work

- Gaussian noise cells: `noise_sigma_0.02`, `noise_sigma_0.05`, `noise_sigma_0.1`
- Color jitter: `color_jitter_0.2`
- Center crop: `center_crop_0.8`
- Controlled transformed training or fine-tuning
- Temperature or threshold changes
- Final feature retention, packaging, or sealed `final_test` evaluation
