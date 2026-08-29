# Plan

Build the detector in evidence-gated milestones: first make the data and evaluation contract reproducible, then establish the frozen-CLIP baseline, and only then add forensic signals one at a time. The first usable release does not depend on every experimental branch; each addition remains only if it improves the locked 50/50 score without unacceptable class-specific regression.

## Scope

- In: binary authentic-versus-fully-synthetic classification; data manifests and leakage controls; independent robustness transforms; frozen CLIP/RINE training; deterministic frequency, texture, PRNU, color, and optics ablations; calibration; C2PA synthetic early exit; local directory inference; reproducible metrics and reports.
- Out: mixed or AI-edited images, face swaps, localization, chained evaluation transforms, online APIs in the inference path, concurrent service infrastructure, and production deployment.

## Google Colab and Drive workflow

Google Colab is the primary training runtime. Google Drive is durable storage; the Colab VM's `/content` filesystem is disposable, faster working storage.

The supplied dataset folder is `https://drive.google.com/drive/folders/1c-IVvAiHlApA49CtU3QQH9XqQDmkbO8U`. Before the first session, add it as a shortcut at `My Drive/hackathon_data`. The folder ID is recorded in `configs/colab.json`, while the mounted runtime consumes the stable shortcut path.

Expected paths are frozen in `configs/colab.json`:

```text
/content/drive/MyDrive/hackathon_data              immutable and cleaned data in Drive
/content/drive/MyDrive/cya-techjam26/artifacts     persistent checkpoints and metrics
/content/cya-techjam26                             remote runtime repository checkout
/content/hackathon_data                            active local dataset copy
/content/cya-techjam26/artifacts                   active local run outputs
```

At the start of every Colab session:

1. Open a notebook through the official VS Code Colab extension and select **Kernel -> Colab -> Auto Connect**.
2. Select a GPU runtime, then use **Colab: Mount Google Drive to Server...** or run:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

3. Clone the repository into `/content/cya-techjam26`, or pull the existing disposable checkout. Local VS Code files are not assumed to exist on the remote runtime.
4. Install without replacing Colab's matched PyTorch/CUDA build:

   ```bash
   cd /content/cya-techjam26
   make install-colab
   ```

5. Copy the active dataset archive/subset from Drive to `/content/hackathon_data` and extract it locally. Do not train by repeatedly reading thousands of small images from the mounted Drive folder.
6. Run `make smoke`. Do not begin extraction or training unless configuration, imports, and CUDA checks pass.
7. Keep embeddings, feature caches, temporary transforms, and frequent logs under `/content` during the run.
8. Copy completed checkpoints, resolved configuration, metrics, environment metadata, and reports to the Drive artifact path. Copy rather than move so an interrupted Drive operation cannot remove the local result.

Every notebook remains a thin launcher for versioned modules in `src/`. A runtime reset requires rerunning installation, local data staging, and smoke checks; durable artifacts already copied to Drive survive the reset.

## Dependencies and parallel teammate tracks

Tasks are ordered by integration dependency, but implementation does not have to be fully sequential. Teammates should own separate module paths to avoid merge conflicts.

| Workstream | Can start | Hard dependency before integration | Can run alongside | Suggested ownership boundary |
|---|---|---|---|---|
| Task 2 data contract | Now | Task 1 complete | Tasks 3, 5, and standalone parts of 4/7/8/9 | `src/cya_detector/data/`, Task 2 scripts/notebook |
| Task 3 transform engine | Now | Frozen config/schema from Task 1 | Tasks 2 and 5 | `src/cya_detector/transforms/`, transform tests |
| Task 4 CLIP loader/head skeleton | Now | Real training waits for Tasks 2 and 3 | Tasks 2, 3, and 5 | `src/cya_detector/models/clip_baseline.py` |
| Task 5 evaluation harness | Now, using fixtures | Real tables wait for Tasks 2–4 | Tasks 2, 3, 4, and feature extraction | `src/cya_detector/evaluation/`, reporting tests |
| Task 7 frequency extraction | Now, using fixtures | Fusion/retention waits for Tasks 4 and 5 | Tasks 2, 3, 5, 8, and 9 | `src/cya_detector/features/frequency.py` |
| Task 8 color features | Now, using fixtures | Fusion/retention waits for Tasks 4 and 5 | Frequency, PRNU, optics, and texture tracks | `src/cya_detector/features/color.py` |
| Task 8 PRNU features | Now, using fixtures | Physical claims wait for Task 2 native-data audit | Frequency, color, optics, and texture tracks | `src/cya_detector/features/prnu.py` |
| Task 8 optical features | Now, using fixtures | Physical claims wait for Task 2 eligibility audit | Frequency, color, PRNU, and texture tracks | `src/cya_detector/features/optics.py` |
| Task 9 patch selector | Now, using fixtures | Learned aggregation waits for Task 4 | Tasks 2, 3, 5, 7, and 8 | `src/cya_detector/features/texture.py` |
| Task 6 RINE integration | After Stage A runs | Tasks 4 and 5 | Later auxiliary fusion experiments | `src/cya_detector/models/rine.py` |
| Task 10 packaging | After feature retention | Selected outputs from Tasks 4–9 | Documentation/demo preparation only | inference CLI, calibration, release tests |

Safe work available immediately for other teammates: Task 3, Task 5, the Task 4 model-loader skeleton, Task 7 deterministic extraction, and the separate Task 8/9 extractor modules. Do not start real model selection, calibration, or final-test evaluation until the required upstream gates pass.

## Action items

[x] **Task 1 — Create the project skeleton and freeze shared configuration**

  - [x] Add `src/`, `scripts/`, `configs/`, `tests/`, `notebooks/`, and `artifacts/` boundaries without committing downloaded images, caches, or checkpoints.
  - [x] Define schema-versioned `configs/colab.json` for paths, seed, CLIP identifier/input size, preprocessing, exact transform cells, feature flags, optimization, and evaluation thresholds.
  - [x] Add full, Colab-safe, and development dependency files covering PyTorch, torchvision, Transformers CLIP, OpenCV, NumPy/SciPy, pandas, scikit-learn, Pillow, scikit-image, and C2PA bindings.
  - [x] Add ignore rules for datasets, feature caches, run outputs, model weights, secrets, and notebook checkpoints.
  - [x] Add run metadata helpers that record package versions, Git commit, resolved configuration, platform, Python version, timestamp, and seed.
  - [x] Add `make smoke` for strict installed-environment validation and `make smoke-bootstrap` for dependency-free repository/configuration validation.
  - [x] Add configuration and metadata unit tests plus a Colab notebook workflow note.

Task 1 implementation is complete. The first connected Colab GPU session must still run `make install-colab`, `make smoke`, and record the assigned accelerator before Task 2 data work begins.

[x] **Task 2 — Reconcile the supplied SID data with the agreed dataset contract** *(verified in Colab on the full SID handoff)*

  - [x] Implement immutable-source inventory, corruption checking, dimensions/formats, SHA-256, color-aware perceptual hashes, and optional strict 20,000-row verification.
  - [x] Implement fail-closed SID label mapping; keep only `real`/`authentic` and `synthetic`/`ai_generated`, and exclude label `2`, tampered, mixed, edited, and ambiguous rows.
  - [x] Keep the existing `cleaned/sid_set/` 336×336 real-only recompressed handoff outside the canonical-data builder.
  - [x] Implement the source-original manifest fields from [training.md](../training/training.md), including stable IDs, source paths, generator/capture metadata, C2PA status, and eligibility flags.
  - [x] Implement exact and near-duplicate grouping, deterministic primary selection, grouped splitting, and mandatory review/exclusion for cross-label duplicate groups.
  - [x] Implement source-byte C2PA scanning with remote/OCSP fetching disabled; matched derivation refuses unchecked/error sources unless an explicit fixture-only override is used.
  - [x] Implement deterministic fixed-Q96 and uniform-Q95–Q100 matched-clean builders with the same Pillow encoder, RGB conversion, 4:4:4 subsampling, metadata stripping, and no resize for both labels.
  - [x] Implement nuisance-only label-predictiveness audits and 1,000-per-label policy pilots.
  - [x] Implement deterministic 60/25/7.5/7.5 grouped splits and persist source/output manifest hashes.
  - [x] Add `01_task2_data_contract.ipynb`, Make targets, and fixture tests covering filtering, corruption, duplicates, cross-label conflicts, splitting, matched encoding, and nuisance metrics.
  - [x] Run the Task 2 notebook against `/content/hackathon_data/raw/sid_set` in Colab and resolve any failed source-audit gate.
  - [x] Preserve both matched-policy pilot reports for the Stage A comparison; do not select from nuisance accuracy alone.
  - [x] Complete when real-data manifests/reports are copied to Drive and their hashes, class counts, C2PA statuses, duplicate findings, and nuisance results have been reviewed.

Task 2's recorded Colab run processed 20,000 rows with 10,000 images per source label, zero corrupt files, zero cross-label duplicate groups, 106 duplicate groups, and 19,882 eligible primary images. C2PA returned `no_manifest` for all inputs without dependency or scan failures. The split contains 11,929 `seed_train`, 4,971 `self_train_pool`, 1,491 `selection_val`, and 1,491 sealed `final_test` images. Source nuisance ROC-AUC was 0.737; the 2,000-image fixed-Q96 and uniform-Q95–100 pilots produced nuisance ROC-AUC values of 0.710 and 0.692 respectively. These diagnostics do not select the matching policy: Task 4 compares both with frozen CLIP. Generated pilot JPEGs live on disposable `/content`, so a fresh runtime reruns `make task2-pilots` while retaining the reports in Drive.

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

[ ] **Task 4 — Build the frozen-CLIP Stage A baseline** *(implementation complete; real Colab runs pending)*

  - [x] Load the vision tower only and lock the exact CLIP variant/input size before producing caches; record the resolved model commit in cache keys.
  - [x] Freeze the backbone and train a linear classifier first, with the MLP available only through an explicit ablation flag.
  - [x] Extract embeddings from the exact received training views; never run matched normalization or robustness probing inside inference.
  - [x] Cache fixed-view embeddings using image hash, resolved model revision, preprocessing version, and view identifier as the key.
  - [x] Report throughput, peak VRAM, cache size, and accumulation toward effective batch size 32; select the physical microbatch in Colab.
  - [ ] Train with AdamW/BCE across seeds 42, 43, and 44 for both Task 2 matching candidates, then retain one policy on `selection_val`.
  - [x] Save latest and best-clean checkpoints now; create best-robustness and best-50/50 only when Task 3 cells exist rather than fabricating scores.
  - [ ] Complete when clean and every independent transform cell produce reproducible R.Acc., F.Acc., accuracy, confusion matrices, and logits.

[ ] **Task 5 — Build the locked evaluation and reporting harness** *(implementation complete; real robustness tables wait for Tasks 3–4)*

  - [x] Compute clean accuracy and mean accuracy across all independent transform-and-parameter cells using the agreed 50/50 formula.
  - [x] Report R.Acc., F.Acc., confusion matrix, ECE, false-positive rate, and false-negative rate for clean data and each transform row.
  - [x] Add generator/checkpoint/source breakdowns and an `unknown` metadata bucket without changing the binary public label.
  - [x] Add deterministic stratified bootstrap confidence intervals for retained model comparisons and future fast-track claims.
  - [x] Make `selection_val` the only source for ordinary checkpoint, hyperparameter, feature-retention, and calibration decisions.
  - [x] Prevent normal experiment commands from reading `final_test`; require both explicit final-evaluation and architecture-frozen flags.
  - [x] Generate machine-readable JSON metrics and a CSV robustness table.
  - [x] Verify with fixtures that a constant or class-collapsed predictor is exposed by per-label metrics.
  - [ ] Complete the integration gate when real Task 4 logits cover clean data and every independent Task 3 transform cell.

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

- Should the primary training policy remain clean-or-one-controlled-transform, with SAFE strictly an ablation as recommended here, or should SAFE become the default training-only composite policy after the Stage A comparison?
