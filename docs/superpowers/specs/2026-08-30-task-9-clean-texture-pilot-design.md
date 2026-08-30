# Task 9 Clean Texture Pilot Design

## Purpose

Task 9 is a bounded, retention-gated experiment that tests whether selected local-detail patches add useful authentic-versus-AI evidence beyond the existing global RINE representation. RINE remains the expected final model unless the local path passes the clean pilot gate. A negative result is valid: the project records that the fixed-budget texture path was tested and rejected rather than retaining unjustified complexity.

## Scope

The pilot uses only the fixed-Q96 matched-clean manifest used for the recorded RINE result:

- `seed_train` trains the small Task 9 heads.
- `selection_val` compares the candidates and decides whether the experiment continues.
- `self_train_pool`, sealed `final_test`, source-original images, Task 3 robustness variants, and Task 8B data are excluded.

Matched-clean is a processing state, not a class. Both `authentic` and `ai_generated` rows are included.

Task 3 transformed training and robustness evaluation are a separate continuation authorized only if this clean pilot passes. They are not implemented as part of this specification.

## Fixed Experiment Contract

- Primary global representation: frozen-CLIP RINE features from layers 6, 12, 18, and 24.
- Local selector: the existing deterministic multi-scale Laplacian/Sobel selector.
- Patch budget: at most four non-overlapping patches per image.
- Source patch size: 336 by 336 pixels.
- Patch encoder: the same frozen CLIP vision backbone used by the global model.
- Patch aggregation: one lightweight learned scorer followed by masked softmax and a weighted average.
- Seeds: 42, 43, and 44 for every model variant.
- Maximum live encoding cost: one global view plus four local views per image.
- Texture energy selects where to look and is never used as an authenticity score.
- No CLIP parameter may become trainable.

Images smaller than a 336-pixel patch are symmetrically zero-padded only as needed to form a valid patch. An odd padding remainder is placed on the right or bottom, matching the Task 3 preprocessing convention. Images that provide fewer than four valid patches retain only the available patches; the model masks absent positions and never duplicates patches to fill the budget.

## Architecture

For each matched-clean image, the global branch reads the existing RINE representation. The local branch selects up to four source-resolution patches before CLIP input conversion, converts each patch with the locked CLIP preprocessing, and obtains one final projected CLIP embedding per patch.

The local aggregator scores each available patch independently, applies a masked softmax across available patches, and computes their weighted average. It must reject a sample with no valid patch embedding rather than manufacture local evidence. Deterministic padding should normally guarantee at least one valid patch for every decodable image.

Three independently trained variants share the same splits, cached inputs, optimization contract, and seeds:

1. `global_only`: RINE global representation to a binary classifier.
2. `local_only`: aggregated patch representation to a binary classifier.
3. `global_local`: projected RINE global representation concatenated with the aggregated local representation, followed by a binary fusion head.

The global-only variant is retrained through the same Task 9 training driver so comparisons do not mix training implementations. The existing recorded RINE result remains an external consistency check.

## Extraction and Cache Flow

Frozen feature extraction occurs before head training, not during the first epoch:

1. Load and validate the fixed-Q96 manifest and permitted splits.
2. Reuse existing global RINE features only when their immutable cache contract matches.
3. Decode each image once for deterministic patch selection and crop construction.
4. Encode available patches with frozen CLIP.
5. Save coordinates, availability masks, patch embeddings, and cache metadata under Colab `/content`.
6. Train all variants and seeds from fixed cached tensors without rerunning CLIP each epoch.

A patch cache key includes the image SHA-256, ordered patch coordinates, CLIP identifier, immutable resolved CLIP revision, preprocessing version, and Task 9 extractor version. Any change creates a different key. Large embedding caches remain disposable under `/content` and are not copied to Drive by default.

## Clean Pilot Gate

The `global_local` candidate proceeds to a separately designed Task 3 robustness continuation only when all of the following hold on `selection_val`:

- Its mean clean accuracy across seeds 42, 43, and 44 is greater than the Task 9 `global_only` mean.
- Neither mean authentic accuracy nor mean AI-generated accuracy regresses by more than 1.0 percentage point relative to `global_only`.
- Paired prediction analysis shows at least one `selection_val` error made by `global_only` that is corrected by `global_local`; the exact corrected and introduced error counts are reported.
- Extraction and inference latency, peak memory, and the fixed maximum of five encoded views are recorded.

Latency and memory are reported during this clean screening gate but do not receive a post-hoc numeric pass threshold. If the accuracy conditions pass, the robustness-continuation design must set the target-machine resource gate before transformed training begins.

If any accuracy condition fails, the decision is `reject_texture_clean_gate`: retain RINE, record Task 9 as tested, and do not perform transformed Task 9 training. Passing the clean gate does not retain Task 9; it authorizes the later robustness experiment, whose locked 50/50 score determines final retention.

## Shared Drive Artifact Contract

No Task 9 directories are manually pre-created. Writers call `mkdir(parents=True, exist_ok=True)` immediately before publishing an artifact. The shared Drive layout is:

```text
task9/
└── clean_pilot_v1/
    ├── global_only/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── local_only/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── global_local/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── seed_44/
    ├── comparison/
    │   ├── global_local_comparison.json
    │   ├── per_seed_metrics.csv
    │   └── latency_comparison.json
    └── metadata/
        ├── extraction_report.json
        ├── resolved_config.json
        ├── run_metadata.json
        └── artifact_manifest.json
```

Each seed directory is created only when its run starts:

```text
seed_42/
├── checkpoints/
│   ├── best_clean.pt
│   └── latest.pt
├── predictions/
│   └── selection_val.csv
├── reports/
│   ├── metrics.json
│   └── training_history.json
└── metadata/
    └── run_metadata.json
```

Experiment-level metadata records the Git commit, input-manifest path and SHA-256, resolved configuration, CLIP identifier and immutable revision, preprocessing and extractor versions, environment information, and hashes of published artifacts. Seed metadata records the variant, seed, optimization history, checkpoint selection, and completion state.

Artifact publication uses a temporary sibling file followed by an atomic replacement where supported. A completed run is never silently overwritten; replacement requires an explicit overwrite option. One designated Task 9 owner publishes canonical results into `clean_pilot_v1` so simultaneous teammates cannot collide.

## Failure Handling

- Reject manifests containing unauthorized splits before extraction starts.
- Reject missing hashes, labels, paths, or incompatible cached feature dimensions.
- Fail if the resolved CLIP revision is unavailable or mutable.
- Fail if any CLIP parameter is trainable.
- Reject non-finite embeddings, attention scores, losses, logits, or metrics.
- Reject cache entries whose metadata does not exactly match the requested extraction contract.
- Do not publish final reports or manifests after a partial extraction or failed training run.
- Do not read or emit predictions for `final_test`.
- Record a decodable-image failure with its sample identifier and stop the canonical pilot rather than silently changing the evaluated population.

## Verification

Unit and fixture tests cover:

- Stable selection of four or fewer non-overlapping patches.
- Deterministic symmetric padding and odd-remainder placement.
- Patch crop, tensor, mask, global-feature, and embedding dimensions.
- Masked attention weights that are finite, nonnegative, zero for absent patches, and sum to one across available patches.
- Rejection of a sample with no available patch embedding.
- Frozen CLIP invariants.
- Shape validation for all three model variants.
- Cache-key changes when image content, coordinates, CLIP revision, preprocessing version, or extractor version changes.
- Cache reuse when the complete immutable contract matches.
- All three variants running seeds 42, 43, and 44.
- Automatic artifact-directory creation and overwrite refusal.
- `final_test` access refusal.
- Deterministic clean-gate keep/reject decisions from fixture predictions.

A smoke fixture performs extraction, trains all three variants with a minimal epoch budget, writes the artifact contract to a temporary root, and produces the comparison decision without network access or pretrained-weight downloads.

## Implementation Boundaries

The implementation is divided into five reviewable units:

1. Freeze Task 9 configuration, types, cache schema, and artifact schema.
2. Build deterministic patch-view preparation and frozen-CLIP cache extraction.
3. Implement masked attention plus the three model variants.
4. Implement cached-feature training, comparison, metrics, and artifact publication.
5. Add CLI/Make targets, a thin Colab launcher, planning documentation, and full verification.

Task 8/8B feature extractors and training modules are outside Task 9 ownership. Task 3 transformation code is consumed only by a future robustness continuation and is not modified by this pilot.
