# Plan

Build the detector in evidence-gated milestones: first make the data and evaluation contract reproducible, then establish the frozen-CLIP baseline, and only then add forensic signals one at a time. The first usable release does not depend on every experimental branch; each addition remains only if it improves the locked 50/50 score without unacceptable class-specific regression.

## Scope

- In: binary authentic-versus-fully-synthetic classification; data manifests and leakage controls; independent robustness transforms; frozen CLIP/RINE training; deterministic frequency, texture, PRNU, color, and optics ablations; calibration; C2PA synthetic early exit; local directory inference; reproducible metrics and reports.
- Out: mixed or AI-edited images, face swaps, localization, chained evaluation transforms, online APIs in the inference path, concurrent service infrastructure, and production deployment.

## Action items

[ ] **Task 1 — Create the project skeleton and freeze shared configuration**

  - [ ] Add `src/`, `scripts/`, `configs/`, `tests/`, and `artifacts/` boundaries without committing downloaded images, caches, or checkpoints.
  - [ ] Define one versioned configuration schema for paths, seeds, model identifier/input size, preprocessing, transform cells, feature flags, optimization, and evaluation thresholds.
  - [ ] Add environment/dependency files for PyTorch, torchvision, the CLIP loader, OpenCV, NumPy/SciPy, pandas, scikit-learn, Pillow, scikit-image, and C2PA bindings.
  - [ ] Add ignore rules for datasets, feature caches, run outputs, model weights, secrets, and notebook checkpoints.
  - [ ] Record package versions, Git commit, resolved configuration, and random seed in every future run directory.
  - [ ] Define a minimal smoke command that loads the configuration and checks CPU/GPU availability without downloading or training anything.
  - [ ] Complete when a fresh environment can validate the configuration and import the planned stack.

[ ] **Task 2 — Reconcile the supplied SID data with the agreed dataset contract**

  - [ ] Preserve `hackathon_data/raw/sid_set/` as immutable source bytes and verify the reported 20,000-image inventory, labels, corruption results, and exact-duplicate findings.
  - [ ] Keep only `real` and `full_synthetic`; assert that SID label `2` and all mixed/tampered/ambiguous rows are absent.
  - [ ] Treat the existing `cleaned/sid_set/` 336×336 set as a smoke-test artifact, not automatically as the canonical training view: its real-only recompression is label-dependent and its resize may remove native low-level evidence.
  - [ ] Build a manifest containing stable `source_id`, hashes, label, source path, dimensions, format, generator metadata when available, processing history, and feature-eligibility fields from [training.md](training.md).
  - [ ] Run exact and perceptual duplicate checks before splitting; group all duplicates and all derivatives of one source together.
  - [ ] Create canonical matched-clean derivatives from the immutable originals using the same encoder, quality distribution, chroma subsampling, color conversion, and metadata policy for both labels.
  - [ ] Compare fixed Q96 with Q95–Q100 using nuisance-only audits; freeze the policy before model comparison.
  - [ ] Produce a source-grouped `seed_train`, `selection_val`, and sealed `final_test`; create a `self_train_pool` only if enough independent sources remain.
  - [ ] Complete when the manifest hash, split hash, class counts, duplicate report, and nuisance-bias report are saved and reproducible.

[ ] **Task 3 — Implement preprocessing and the independent-transform contract**

  - [ ] Separate three concepts in code: offline matched-clean construction, training augmentation policy, and deterministic evaluation transforms.
  - [ ] Implement JPEG Q90/Q70/Q50/Q30, blur sigma 0.5/1.0/2.0, resize round trips 0.5x/0.25x, Gaussian noise sigma 0.02/0.05/0.10, color jitter within +/-20%, and center crop retaining 80%.
  - [ ] Ensure every benchmark variant is created directly from its matched-clean parent with exactly one transform and one parameter cell.
  - [ ] Pin resize library/version, bilinear interpolation, antialiasing, dimension rounding, RGB/dtype handling, and exact restoration dimensions; retain resize outputs losslessly.
  - [ ] Log the parent, transform, realized parameter, seed, extractor/preprocessing version, and output hash for every materialized variant.
  - [ ] Implement the primary controlled sampler as 50% clean and 50% transformed, with balanced labels and uniform transform-cell selection.
  - [ ] Add SAFE as a separately named training-policy ablation: crop-based model input plus training-only flip/jitter/rotation/mask. Do not apply SAFE to validation/test or silently combine it with the controlled benchmark sampler.
  - [ ] Prefer native-resolution parents and crop to the locked CLIP input size; specify and test deterministic handling for images smaller than the required crop.
  - [ ] Add tests proving that evaluation variants contain one transform only, resize outputs match parent dimensions, stochastic rows reproduce from seeds, and both labels receive identical sampling distributions.
  - [ ] Complete when a small fixture dataset can regenerate byte-identical deterministic variants and statistically matched stochastic policies.

[ ] **Task 4 — Build the frozen-CLIP Stage A baseline**

  - [ ] Load the vision tower only and lock the exact CLIP variant/input size before producing caches.
  - [ ] Freeze the backbone and train a linear classifier first, followed by a small MLP only if the linear baseline is stable.
  - [ ] Extract embeddings from the exact received training views; never run matched normalization or robustness probing inside inference.
  - [ ] Cache fixed-view embeddings using image hash, model revision, preprocessing version, and view identifier as the key.
  - [ ] Run a representative throughput/VRAM/cache-size pilot and choose the physical microbatch from evidence; use accumulation toward an effective batch size of 32 when practical.
  - [ ] Train with the documented AdamW/BCE starting values and at least three seeds for the retained candidate.
  - [ ] Save latest, best-clean, best-robustness, and best-50/50 checkpoints with their full run configuration.
  - [ ] Complete when clean and every independent transform cell produce reproducible R.Acc., F.Acc., accuracy, confusion matrices, and logits.

[ ] **Task 5 — Build the locked evaluation and reporting harness**

  - [ ] Compute clean accuracy and mean accuracy across all independent transform-and-parameter cells using the agreed 50/50 formula.
  - [ ] Report R.Acc., F.Acc., confusion matrix, ECE, false-positive rate, and false-negative rate for clean data and each transform row.
  - [ ] Add generator/checkpoint/source breakdowns and an `unknown` metadata bucket without changing the binary public label.
  - [ ] Add bootstrap confidence intervals for retained model comparisons and any future fast-track precision/coverage claim.
  - [ ] Make `selection_val` the only source for checkpoint, hyperparameter, feature-retention, and calibration decisions.
  - [ ] Prevent normal experiment commands from reading `final_test`; require an explicit final-evaluation mode after the architecture is frozen.
  - [ ] Generate machine-readable metrics plus the robustness table needed for the report/demo.
  - [ ] Complete when a deliberately constant or class-collapsed predictor is exposed by the per-label metrics and cannot look successful through aggregate accuracy alone.

[ ] **Task 6 — Add the RINE-style Stage B representation**

  - [ ] Expose the selected intermediate CLIP representations without unfreezing the backbone.
  - [ ] Train only the layer-importance estimator and binary head on the same manifest, views, seeds, and sampler as Stage A.
  - [ ] Compare plain final-layer CLIP against RINE-style fusion on clean, transformed, and available generator/source strata.
  - [ ] Measure added cache size, extraction time, training time, and peak memory.
  - [ ] Retain Stage B only if it improves the predeclared selection criterion without breaching R.Acc./F.Acc. regression limits.
  - [ ] Complete when Stage A versus Stage B is a reproducible, apples-to-apples ablation with a recorded keep/drop decision.

[ ] **Task 7 — Implement deterministic Stage 1 frequency features**

  - [ ] Extract FFT/DCT log-magnitude summaries, radial/angular power, periodic-peak prominence, residual autocorrelation, and local neighboring-pixel dependencies.
  - [ ] Keep generator paradigm, checkpoint, decoder/tokenizer, and upsampling metadata as evaluation strata rather than prediction targets.
  - [ ] Fit feature scaling on `seed_train` only and cache versioned vectors with validity indicators.
  - [ ] Train a frequency-only classifier and then a small projection fused with the retained CLIP representation.
  - [ ] Compare magnitude and bounded phase-spectrum variants, especially on JPEG and resize rows.
  - [ ] Audit overlap with compression/file-source nuisance features and remove generator-specific cues that fail held-out evaluation.
  - [ ] Keep the Stage 1 early exit disabled; consider enabling synthetic-only exit only after locked precision, confidence-interval, coverage, authentic, held-out-family, and transform gates pass.
  - [ ] Complete when frequency-only and incremental-fusion tables support a keep/drop decision and the default inference path still falls through to Stage 2.

[ ] **Task 8 — Add auxiliary feature families one at a time**

  - [ ] Implement RGB/Lab global and local standardized inter-channel correlations with low-variance masks, coverage, and numerical guards.
  - [ ] Implement the single-image PRNU-coherence residual summaries as an experimental proxy; do not present it as camera identification.
  - [ ] Implement radial chromatic-aberration fitting with support/confidence and optional radial-distortion fitting only on eligible scenes.
  - [ ] Derive eligibility from native dimensions, provenance/processing history, and extractor support; do not infer physical-signal eligibility from resolution alone.
  - [ ] Mark the resized/recompressed SID cleaned set ineligible for native PRNU/optics claims unless a separate validation proves signal preservation.
  - [ ] Normalize each family on `seed_train`, zero-fill only after normalization, and always pair missing values with an explicit mask.
  - [ ] Train only each family projection and the fusion head while CLIP remains frozen; run RGB-only, Lab-only, PRNU-only, CA-only, eligible-distortion-only, and incremental-fusion ablations.
  - [ ] Audit feature missingness, validity, and confidence by label, dataset, and transform so they cannot become label shortcuts.
  - [ ] Complete when every family has a documented physical limitation, coverage report, latency measurement, and locked keep/drop decision.

[ ] **Task 9 — Add the texture-aware local-detail path under a fixed budget**

  - [ ] Generate multi-scale Laplacian/Sobel energy maps and select fixed top-k non-overlapping patches before CLIP input conversion.
  - [ ] Keep a global image view so patch selection cannot discard semantic context.
  - [ ] Benchmark shared frozen-CLIP patch encoding against a small shared patch head if repeated CLIP encodes exceed the measured compute budget.
  - [ ] Train soft attention/aggregation and fusion weights; never use smoothness, sharpness, edge density, LBP, GLCM, or OCR confidence as fixed verdict rules.
  - [ ] Tune patch size, patch count, energy scale, and aggregation only on `selection_val`.
  - [ ] Measure global-only, local-only, and combined performance on clean and resize-0.5x/0.25x, including authentic false-positive changes.
  - [ ] Measure redundancy with frequency and PRNU outputs before counting the texture path as additional evidence.
  - [ ] Complete when its accuracy benefit justifies the approximately `1 + k` view-encoding cost and the fixed inference budget is recorded.

[ ] **Task 10 — Freeze, calibrate, package, and only then attempt optional improvements**

  - [ ] Select the final retained feature set and checkpoint using the locked 50/50 criterion and predeclared class-regression tolerances.
  - [ ] Fit one temperature on clean `selection_val` logits and reuse it unchanged for transformed data and inference; keep the binary threshold fixed unless the challenge specifies otherwise.
  - [ ] Add Stage 0 C2PA verification so only a valid signed AI-generation claim can early-exit; absence, invalidity, or an authenticity claim must fall through.
  - [ ] Implement synchronous directory inference that processes each received image once and writes only `{"image_path": ..., "pred": ...}` to the public JSON.
  - [ ] Add handling/tests for corrupt files, unsupported modes, alpha channels, grayscale input, very small images, extreme aspect ratios, and deterministic ordering.
  - [ ] Measure final model size, latency, peak memory, and disk/cache requirements on the target machine.
  - [ ] Run the sealed `final_test` once after weights, calibration, threshold, and feature flags are frozen; generate robustness, ablation, generalization, and error-analysis artifacts.
  - [ ] Attempt SAFE-versus-controlled training, adapters/final-block fine-tuning, self-training, ConvNeXt-Tiny, or external datasets only as separately versioned experiments after the baseline package works.
  - [ ] Reject and roll back any optional experiment that fails the score, per-label, held-out-source, coverage, or resource gates.
  - [ ] Complete when a fresh environment can reproduce training/evaluation artifacts and run the local JSON inference contract without external APIs.

## Open questions

- Which exact CLIP checkpoint/input size should be frozen first: ViT-L/14@336 as currently designed, or a 224-pixel variant that better matches the supplied SAFE crop example and lowers compute?
- Can the untouched `hackathon_data/raw/sid_set/` bytes be made available to every trainer, or must the first baseline use the existing 336×336 cleaned handoff while the canonical dataset is rebuilt elsewhere?
- Should the primary training policy remain clean-or-one-controlled-transform, with SAFE strictly an ablation as recommended here, or should SAFE become the default training-only composite policy after the Stage A comparison?
