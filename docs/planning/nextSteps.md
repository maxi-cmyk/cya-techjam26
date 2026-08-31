# Project history and workflow

This document records how the detector was built as evidence-gated tasks and subtasks. The workflow first made the data and evaluation contract reproducible, then established the frozen-CLIP baseline, tested forensic additions one at a time, and finally froze and packaged the retained model. An experimental branch was retained only when it improved the locked 50/50 score without unacceptable class-specific regression.

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
2. Select the runtime required by the notebook: GPU for CLIP/RINE or repeated patch encoding; CPU for deterministic frequency and color/physical feature extraction. Then use **Colab: Mount Google Drive to Server...** or run:

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
## Action items

[x] **Task 1 — Create the project skeleton and freeze shared configuration**

  - [x] Add `src/`, `scripts/`, `configs/`, `tests/`, `notebooks/`, and `artifacts/` boundaries without committing downloaded images, caches, or checkpoints.
  - [x] Define schema-versioned `configs/colab.json` for paths, seed, CLIP identifier/input size, preprocessing, exact transform cells, feature flags, optimization, and evaluation thresholds.
  - [x] Add full, Colab-safe, and development dependency files covering PyTorch, torchvision, Transformers CLIP, OpenCV, NumPy/SciPy, pandas, scikit-learn, Pillow, scikit-image, and C2PA bindings.
  - [x] Add ignore rules for datasets, feature caches, run outputs, model weights, secrets, and notebook checkpoints.
  - [x] Add run metadata helpers that record package versions, Git commit, resolved configuration, platform, Python version, timestamp, and seed.
  - [x] Add `make smoke` for strict installed-environment validation and `make smoke-bootstrap` for dependency-free repository/configuration validation.
  - [x] Add configuration and metadata unit tests plus a Colab notebook workflow note.

Task 1 is complete and has been exercised in both CPU and T4 Colab sessions. Continue running the notebook-local install and smoke gates after every disposable-runtime reset.

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

[x] **Task 3 — Implement preprocessing and the independent-transform contract**

Run focused verification with `make task3-test`; materialize the selected Task 2 fixture with `make task3-fixture ARTIFACT_ROOT=artifacts TASK2_SELECTED_MANIFEST=path/to/selected_manifest.csv`. The target's practical default remains the fixed-Q96 pilot until Task 2 records its selection.

  - [x] Separate three concepts in code: offline matched-clean construction, training augmentation policy, and deterministic evaluation transforms.
  - [x] Implement JPEG Q90/Q70/Q50/Q30, blur sigma 0.5/1.0/2.0, resize round trips 0.5x/0.25x, Gaussian noise sigma 0.02/0.05/0.10, color jitter within +/-20%, and center crop retaining 80%.
  - [x] Ensure every benchmark variant is created directly from its matched-clean parent with exactly one transform and one parameter cell.
  - [x] Pin resize library/version, bilinear interpolation, antialiasing, dimension rounding, RGB/dtype handling, and exact restoration dimensions; retain resize outputs losslessly.
  - [x] Log the parent, transform, realized parameter, seed, extractor/preprocessing version, and output hash for every materialized variant.
  - [x] Implement the primary controlled sampler as 50% clean and 50% transformed, with balanced labels and uniform transform-cell selection.
  - [x] Add SAFE as a separately named training-policy ablation: crop-based model input plus training-only flip/jitter/rotation/mask. Do not apply SAFE to validation/test or silently combine it with the controlled benchmark sampler.
  - [x] Prefer native-resolution parents and crop to the locked CLIP input size; specify and test deterministic handling for images smaller than the required crop.
  - [x] Add tests proving that evaluation variants contain one transform only, resize outputs match parent dimensions, stochastic rows reproduce from seeds, and both labels receive identical sampling distributions.
  - [x] Complete when a small fixture dataset can regenerate byte-identical deterministic variants and statistically matched stochastic policies; Notebook 07 additionally verified 1,390 real development parents and 19,460 independent variants across all 14 cells.

[x] **Task 4 — Build the frozen-CLIP Stage A baseline** *(clean and robustness evaluation verified)*

  - [x] Load the vision tower only and lock the exact CLIP variant/input size before producing caches; record the resolved model commit in cache keys.
  - [x] Freeze the backbone and train a linear classifier first, with the MLP available only through an explicit ablation flag.
  - [x] Extract embeddings from the exact received training views; never run matched normalization or robustness probing inside inference.
  - [x] Cache fixed-view embeddings using image hash, resolved model revision, preprocessing version, and view identifier as the key.
  - [x] Report throughput, peak VRAM, cache size, and accumulation toward effective batch size 32; select the physical microbatch in Colab.
  - [x] Train with AdamW/BCE across seeds 42, 43, and 44 for both Task 2 matching candidates; fixed-Q96 won the clean `selection_val` comparison.
  - [x] Save latest and best-clean checkpoints now; create best-robustness and best-50/50 only when Task 3 cells exist rather than fabricating scores.
  - [x] Complete when clean and every independent transform cell produce reproducible R.Acc., F.Acc., accuracy, confusion matrices, and logits.

The verified T4 run completed all six policy/seed combinations and persisted them to Drive. Fixed-Q96 seed 42 extracted 1,390 embeddings in 38.4 seconds at 36.2 images/second, used approximately 1.43 GB peak GPU allocation, and reached 97.58% best clean accuracy. Notebook 07 subsequently evaluated the frozen Stage A checkpoints across every independent transform cell and recorded a 94.71% three-seed mean locked score.

[x] **Task 5 — Build the locked evaluation and reporting harness** *(clean and robustness reports verified)*

  - [x] Compute clean accuracy and mean accuracy across all independent transform-and-parameter cells using the agreed 50/50 formula.
  - [x] Report R.Acc., F.Acc., confusion matrix, ECE, false-positive rate, and false-negative rate for clean data and each transform row.
  - [x] Add generator/checkpoint/source breakdowns and an `unknown` metadata bucket without changing the binary public label.
  - [x] Add deterministic stratified bootstrap confidence intervals for retained model comparisons and future fast-track claims.
  - [x] Make `selection_val` the only source for ordinary checkpoint, hyperparameter, feature-retention, and calibration decisions.
  - [x] Prevent normal experiment commands from reading `final_test`; require both explicit final-evaluation and architecture-frozen flags.
  - [x] Generate machine-readable JSON metrics and a CSV robustness table.
  - [x] Verify with fixtures that a constant or class-collapsed predictor is exposed by per-label metrics.
  - [x] Complete the integration gate when real Task 4 logits cover clean data and every independent Task 3 transform cell.

Notebook 07 populated the locked evaluation from 1,390 clean development rows and 19,460 independently transformed rows. All 20,850 frequency rows and all 20,850 color rows were valid, reports were synced to Drive, and `final_test` remained sealed throughout.

[x] **Task 6 — Add the RINE-style Stage B representation** *(controlled robustness ablation verified and retained)*

  - [x] Expose predeclared CLIP layers 6/12/18/24 without unfreezing the backbone and cache their CLS representations by immutable view/model keys.
  - [x] Train only softmax layer importance and one binary head on fixed-Q96 with the same manifest, splits, seeds, and optimization contract as Stage A.
  - [x] Implement paired Stage A versus Stage B clean and transformed comparison with the predeclared one-point per-class regression tolerance.
  - [x] Record added cache size, extraction throughput, training history, layer importance, and peak GPU memory.
  - [x] Run `03_rine_stage_b.ipynb` for seeds 42/43/44 and record the provisional clean keep/drop result.
  - [x] Retain Stage B only if it improves the locked 50/50 criterion without breaching R.Acc./F.Acc. regression limits after Task 3 integration.
  - [x] Complete when Stage A versus Stage B is a reproducible, apples-to-apples ablation with a recorded keep/drop decision.

The verified T4 run extracted layers 6/12/18/24 for 1,390 fixed-Q96 images in 39.6 seconds at 35.1 images/second and approximately 1.77 GB peak GPU allocation. Across seeds 42/43/44, the earlier clean-trained RINE reached 99.39% mean clean accuracy and a later Notebook 07 rerun measured its mean locked score at 96.20%. Retraining only the RINE head under the controlled sampler produced 100.00% mean clean accuracy, 99.62% mean robustness accuracy, and a 99.81% mean locked score. Controlled RINE is retained as the pre-Task-9 parent.

[x] **Task 7 — Implement deterministic Stage 1 frequency features** *(robustness fusion evaluated and rejected)*

  - [x] Extract FFT/DCT log-magnitude summaries, radial/angular power, periodic-peak prominence, residual autocorrelation, and local neighboring-pixel dependencies.
  - [x] Keep generator/checkpoint/source metadata as evaluation strata rather than prediction targets.
  - [x] Fit feature scaling on `seed_train` only and cache versioned vectors with validity indicators and extraction errors.
  - [x] Train magnitude/residual and magnitude/residual-plus-bounded-phase frequency-only classifiers, then evaluate incremental RINE fusion.
  - [x] Implement paired magnitude-versus-phase comparison and evaluate the retained representation across all independent transform cells.
  - [x] Audit the strongest correlations with file size, dimensions, and normalization quality on `seed_train`.
  - [x] Keep the Stage 1 early exit disabled in configuration and training; any future early-exit claim still requires locked precision, confidence, coverage, authentic, held-out-family, and transform gates.
  - [x] Run `04_frequency_stage1.ipynb` and record the clean representation decision across seeds 42/43/44.
  - [x] Complete the frequency decision while keeping the default inference path falling through to Stage 2; reject incremental RINE fusion under the locked gate.

The verified CPU run completed all six representation/seed combinations. Magnitude plus residual features reached 83.03% mean clean accuracy versus 80.00% with bounded phase added, so phase was dropped. Notebook 07 then evaluated RINE+frequency across three seeds: its 52.15% mean locked score was 47.66 points below controlled RINE, with severe seed instability. Frequency fusion is rejected, `final_test` remains sealed, and the Stage 1 early exit remains disabled.

[x] **Task 8 — Add auxiliary feature families one at a time** *(Lab robustness fusion evaluated and rejected)*

  - [x] Implement RGB/Lab global and local standardized inter-channel correlations with low-variance masks, coverage, and numerical guards.
  - [x] Implement the single-image PRNU-coherence residual summaries as an experimental proxy; do not present it as camera identification.
  - [x] Implement radial chromatic-aberration fitting with support/confidence; radial distortion remains deferred until eligible line/arc support exists.
  - [x] Derive label-independent eligibility from native dimensions and `source_original` provenance rather than resolution alone.
  - [x] Mark matched/recompressed SID views ineligible for native PRNU/optics claims.
  - [x] Normalize each physical family on valid/eligible `seed_train` rows, zero-fill only after normalization, and pair missing values with eligibility, validity, and confidence masks.
  - [x] Run RGB-only, Lab-only, and RGB+Lab feature-only clean ablations across seeds 42/43/44; select Lab within the color family.
  - [x] Refuse PRNU-only, CA-only, and radial-distortion training on the current matched-Q96 handoff because native physical eligibility is zero for both labels.
  - [x] Audit feature eligibility, validity, and confidence counts by label/split and refuse physical-family fitting when eligibility is absent or class-imbalanced.
  - [x] Run `05_auxiliary_stage_c.ipynb` for RGB, Lab, and combined color baselines across seeds 42/43/44.
  - [x] Complete the matched-view family decision with documented physical limitations and coverage; evaluate incremental RINE+Lab fusion across all independent transform cells.

The verified CPU run extracted 67 features for 1,390 rows and completed all nine color representation/seed combinations. Lab reached 82.42% mean clean accuracy, RGB+Lab reached 78.79%, and RGB reached 55.15%, so Lab entered the later fusion ablation. Notebook 07 measured RINE+Lab at a 98.95% mean locked score, 0.87 points below controlled RINE, while AI-generated accuracy regressed 1.82 points and breached the one-point gate. Lab fusion is rejected and the combined frequency+Lab candidate was correctly skipped. PRNU and chromatic-aberration remain ineligible on matched/recompressed views; no physical family is retained and no camera or lens authenticity claim is made.

[x] **Task 8B — Revisit physical-capture features with eligible native data** *(completed pilot; no physical feature retained)*

  - [x] Select and verify the public/licensed sources: accessible PREMIER v3 N1/N2 (CC BY-SA 4.0; N3 optional if later accessible) and GenImage AI-only branches (CC BY-NC-SA 4.0, non-commercial only).
  - [x] Add a fail-closed inventory importer, license/provenance fields, Task 8B-only grouped splits, existing-root storage configuration, and training-only PRNU reference construction.
  - [x] Add separate source, PRNU-reference, chromatic-aberration metadata, and nuisance/training readiness gates plus a thin Colab staging-and-audit launcher.
  - [x] Add a local-first extracted-file scanner that balances classes and generator families, excludes GenImage nature rows, and creates a draft inventory without overwriting review work.
  - [x] Acquire 644 native/minimally processed PREMIER N1/N2 files, inventory 640 balanced licensed rows across 13 device IDs, and preserve multiple images per sensor under verified CC BY-SA 4.0 permission.
  - [x] Add four diverse fully synthetic families and materialize label-independent 256 px crop-only, metadata-stripped, uncompressed TIFF views; keep BigGAN stress-only.
  - [x] Freeze grouped splits that hold out complete authentic devices and synthetic generator families without overlap.
  - [x] Run source-original and matched-view nuisance audits before fitting any physical fusion head; matched balanced accuracy is 0.50.
  - [x] Compare single-image coherence (AUC 0.543) with disjoint-reference multi-image fingerprints (AUC 0.538) using device identity only; neither meets the 0.60 AUC gate.
  - [x] Apply the chromatic-aberration readiness gate; validation is correctly skipped because calibrated lens/focal/edge-rich coverage is zero.
  - [x] Apply the frozen-RINE fusion gate; no projection or fusion weights are trained because no physical estimator independently validated.
  - [x] Record the fail-closed decision and hashes in `artifacts/task8b/reports/retention_decision.json`; downstream binary/robustness evaluation is not run without a candidate.
  - [x] Implement the isolated PRNU v2 protocol and Colab/Drive routing under `artifacts/task8b_v2` without changing the original Task 8B evidence.
  - [x] Run PRNU v2 on the licensed PREMIER source package: AUC 0.9171, top-1 0.8554 versus 0.10 random, and same/different mean PCE 262.65/11.31; binary labels and fusion remain unused.
  - [x] Interpret the result as known-device repeatability only; prohibit maximum reference-bank PCE from acting as a generic authenticity feature.
  - [x] Define and version a reference-free `prnu_v2_runtime` single-image vector derived from the frozen residual pipeline, with no device ID, metadata, known-reference comparison, or PCE input.
  - [x] Preserve the initial 512 px incompatibility audit: all 20,850 clean/transform rows were ineligible, with no read failures or resizing; do not present this as a 256 px PRNU result.
  - [x] Implement label-independent matched-clean 256 px eligibility, validity/confidence masks, no resize, no EXIF transposition, and separate all-view coverage reporting before fitting.
  - [x] Skip fitting, selection, and held-out confirmation for the initial 512 px attempt because its prerequisite data gate failed; the competition `final_test` remains unread.
  - [x] Add `08_prnu_v2_binary.ipynb`, CLI/Make entrypoints, a PRNU-only controlled diagnostic, RINE+PRNU fusion, three-seed comparison, and Drive synchronization without reading `final_test`.
  - [x] Preserve the controlled-RINE parents independently so the corrected PRNU run does not repeat that training.
  - [x] Run and record the separately versioned label-free v2 device test at 256 px: AUC 0.8593, top-1 0.6566 versus 0.10 random, and same/different mean PCE 73.27/11.25 all pass.
  - [x] Extract all 20,850 clean/transform rows with explicit masks, run PRNU-only and RINE+PRNU across seeds 42/43/44, and reject fusion after the 33.43% mean locked score and two-seed collapse.
  - [x] Retain no Task 8B physical feature and make no camera or lens authenticity claim.

[x] **Task 9 — Add the texture-aware local-detail path under a fixed budget** *(clean gate passed; Stage-1 robustness tested and rejected — controlled RINE retained)*

  - [x] Generate multi-scale Laplacian/Sobel energy maps and select fixed top-k non-overlapping patches before CLIP input conversion.
  - [x] Add deterministic fixture coverage for non-overlap, stable ordering, top-k bounds, high-detail selection, small images, and invalid parameters.
  - [x] Keep a global image view so patch selection cannot discard semantic context: every run always encodes the frozen global RINE view alongside up to four local patches.
  - [x] Freeze the clean-only pilot boundary, patch budget, and variant/seed matrix in `configs/colab.json["texture"]`: `patch_size` 112, `patch_count` 4 (a maximum five-view-per-image encoding budget — one global view plus up to four 112 px source patches, upsampled by the locked CLIP processor to its native 336 px input), `variants` `global_only`/`local_only`/`global_local`, and `seeds` 42/43/44 (nine runs total). The pilot trains and compares only on the fixed-Q96 matched-clean manifest's `seed_train` and `selection_val` splits; it never reads `self_train_pool`, sealed `final_test`, source-original images, Task 3 robustness variants, or Task 8B data, and CLIP remains fully frozen throughout.
  - [x] Implement masked-attention `global_only`/`local_only`/`global_local` heads, frozen global/patch feature extraction and caching (large RINE and patch caches stay under `/content`; nothing is written to Drive during extraction), per-variant/seed training with atomic artifact publication, and the deterministic clean gate (`continue_to_robustness_design` or `reject_texture_clean_gate`) in `src/cya_detector/{models,training,evaluation}/`.
  - [x] Add `scripts/extract_texture_features.py`, `scripts/train_texture_pilot.py`, and `scripts/compare_texture_pilot.py`, and wire them to `make task9-test`, `task9-extract`, `task9-run`, `task9-matrix`, and `task9-compare`. `task9-matrix` runs all nine variant/seed combinations sequentially, never concurrently, because simultaneous GPU extraction would duplicate frozen-CLIP memory and race on the shared `/content` caches.
  - [x] Add `09_texture_stage_d.ipynb` as a thin Colab launcher that stages the fixed-Q96 manifest, extracts/caches features once under `/content`, resumes per-variant/seed training, and copies each completed run — and only a completed run — to the shared Drive artifact root (`My Drive/cya-techjam26/artifacts/task9`), so no empty Drive directory is created ahead of a result.
  - [x] Run the real Colab extraction and all nine variant/seed trainings on the fixed-Q96 pilot manifest; never use smoothness, sharpness, edge density, LBP, GLCM, or OCR confidence as fixed verdict rules.
  - [x] Measure global-only, local-only, and combined performance on clean and resize-0.5x/0.25x, including authentic false-positive changes. Done as part of the real Stage-1 robustness run (see below): `resize_scale_0.25` was the worst cell for every variant.
  - [x] Apply the clean gate to the real nine-run comparison. `global_local` reached 100% clean accuracy for seeds 42/43/44 versus a 99.394% `global_only` mean, corrected one error per seed, introduced none, and emitted `continue_to_robustness_design`. This authorizes only the separately specified frozen-checkpoint Stage-1 robustness continuation; it does not retain Task 9 by itself.
  - [x] Implement the frozen-checkpoint Stage-1 robustness continuation: materialize the nine JPEG/blur/resize cells directly from fixed-Q96 clean parents; recompute patch positions on each transformed image; reuse transformed features across variants and seeds; evaluate immutable best-clean checkpoints without retraining or recalibration; and compare `global_local` against both `global_only` and hash-verified controlled-RINE predictions using locked-score, per-class, and worst-cell gates. The implementation lives in `src/cya_detector/evaluation/texture_robustness.py`, `scripts/{materialize,evaluate,compare}_texture_robustness.py`, and `10_texture_robustness_stage1.ipynb`; all 258 repository tests and the dependency-free fixture smoke passed locally.
  - [x] Ran the real Colab Stage-1 materialization, evaluation, and comparison. Result: `reject_texture_robustness_stage1`. `global_local` tied `controlled_rine` on clean accuracy (100% each) but scored materially worse on robustness (93.13% mean vs. 99.80% for controlled RINE, 96.66% for the internal `global_only` ablation); locked-score deltas were negative against both comparators (-0.033 vs. controlled RINE, -0.015 vs. `global_only`) and the worst-cell gate failed on multiple cells, most severely `resize_scale_0.25` (AI-generated accuracy fell to 50.2% for `global_local`, 18.4% for `local_only` — nearly random). The local patch signal is fragile under aggressive downsampling/blur/compression and actively misleads the fusion head under those transforms rather than degrading gracefully. Controlled RINE remains the retained global representation; no texture feature is retained for Task 10. This also means the remaining Task 3 noise/color-jitter/crop cells are not authorized for evaluation, since only a pass would have unlocked that.
  - [x] Complete: Task 9 texture path is tested and rejected at the robustness stage. No further texture work is planned before Task 10.

[x] **Task 10 — Freeze, calibrate, package, and only then attempt optional improvements** *(baseline complete: Task 10A inference skeleton merged; Task 10B model selected (controlled RINE seed 42), real predictor wired in, calibration deliberately skipped (see below), `final_test` run — 99.29% accuracy, the project's final sealed result. Resource profiling on target hardware and any optional post-baseline experiment remain open.)*

  - [x] Select the final retained feature set and checkpoint using the locked 50/50 criterion and predeclared class-regression tolerances. *(Task 10B — controlled RINE seed 42 selected, 99.85% locked score, the highest of the three retained seeds. The real checkpoint and its 165-row clean `selection_val` predictions CSV were restored from Drive to `artifacts/robustness/train-controlled-rine/seed_42/` and verified: `torch.load` succeeds, `model_state_dict` is `layer_logits (4,)` + `classifier.weight (1, 1024)` + `classifier.bias (1,)`, `selection_score` 0.9984848484848485 matches the recorded 99.85%.)*
  - [x] Fit one temperature on clean `selection_val` logits and reuse it unchanged for transformed data and inference; keep the binary threshold fixed unless the challenge specifies otherwise. *(Task 10B — ran `scripts/fit_temperature_calibration.py` against seed 42's real 165 clean rows. Result: degenerate. The fitted temperature hit the search bound (0.05) rather than an interior minimum, because that set has zero errors (`clean: 1.0`, confusion matrix 76/76 + 89/89 correct) — with nothing to penalize overconfidence, NLL minimization only wants to sharpen further. Using T=0.05 would not change any classification (the threshold stays fixed at 0.5) but would crush informative probabilities toward numerically-indistinguishable-from-0-or-1, making reported confidence *less* useful. **Decision: skip calibration, use T=1 (raw sigmoid probabilities).** This is a genuine property of the data — a perfectly-accurate clean validation set cannot support meaningful post-hoc temperature scaling by NLL minimization alone — not a bug in the fitting code (which is unit-tested and correctly recovers known temperatures on non-degenerate synthetic data).)*
  - [x] Add Stage 0 C2PA verification so only a valid signed AI-generation claim can early-exit; absence, invalidity, or an authenticity claim must fall through. *(Task 10A — implemented in `src/cya_detector/inference/c2pa.py` against the real installed `c2pa-python` 0.37.8 API; the exact manifest-schema parsing is unverified against a real signed test asset, see README limitations.)*
  - [x] Implement synchronous directory inference that processes each received image once and writes only `{"image_path": ..., "pred": ...}` to the public JSON. *(Task 10A — `run_inference.py` at the repository root.)*
  - [x] Add handling/tests for corrupt files, unsupported modes, alpha channels, grayscale input, very small images, extreme aspect ratios, and deterministic ordering. *(Task 10A — five stable error codes, decompression-bomb guard, RGB normalization, NFC-normalized deterministic discovery; 43 tests across `tests/test_c2pa_inference.py`, `tests/test_directory_inference.py`, `tests/test_inference_cli.py`, `tests/test_rine_predictor.py`.)*
  - [x] Wire the real controlled-RINE seed-42 checkpoint into the inference CLI in place of the stub. *(Task 10B — `src/cya_detector/inference/rine_predictor.py`'s `RinePredictor`: frozen CLIP-ViT-L/14-336 (pinned resolved revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`) intermediate layers 6/12/18/24 CLS tokens, fused by the loaded RINE head, raw sigmoid probability (T=1 per the calibration decision above). Validates checkpoint identity (stage/seed/matching_policy/layers) and the resolved CLIP revision before serving any prediction, and refuses non-finite output. `run_inference.py <dir> --output-dir <dir>` is the real, working submission command — `--checkpoint` now *defaults* to `artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt` when present, so a bare invocation with no flags scores with the real model; `--no-checkpoint` forces the Task 10A stub explicitly for testing, and a missing default checkpoint falls back to the stub with a loud stderr warning rather than silently. 8 tests pass against a deterministic fixture CLIP (checkpoint identity/revision validation and a manual forward-pass parity check), plus a real end-to-end run — real downloaded CLIP-ViT-L/14-336 weights, the real seed-42 checkpoint, one synthetic image — completed successfully (`exit=0`, a real non-trivial probability `0.4012...`, not the stub's constant `0.5`).)*
  - [ ] Measure final model size, latency, peak memory, and disk/cache requirements on the target machine. *(Task 10B — `profile_predictor()`/`checkpoint_disk_footprint()` implemented and tested in `src/cya_detector/evaluation/resource_profile.py`; not yet run against the real checkpoint on target Colab GPU hardware — only verified on local CPU as part of the end-to-end wiring check above.)*
  - [x] Run the sealed `final_test` evaluation exactly once, after weights, calibration, threshold, and feature flags were frozen. *(`src/cya_detector/evaluation/final_test.py`'s `evaluate_final_test` is the only code in the repository permitted to read `final_test` rows: requires an explicit `confirm_final_test_read=True`, validates every row is a direct fixed-Q96 matched-clean view with a hash-verified image before scoring it, and refuses outright — no resume, no overwrite — if a prior result already exists at the output root. Run via `notebooks/11_final_test.ipynb` on Colab on 2026-08-31. **Result, permanent and not to be rerun:** 141 samples, **99.29% overall accuracy** (140/141), 100.00% AI-generated accuracy (69/69, 0.00% false negative rate), 98.61% authentic accuracy (71/72, 1.39% false positive rate), ECE 0.0189, threshold 0.5, controlled RINE seed 42 with T=1 (no calibration, per the decision above). This closely tracks the `selection_val` numbers throughout the project (100% clean, 99.81% locked score) — no sign of overfitting to the dev split. `checkpoint_identity` in the published report confirms it ran against the exact pinned checkpoint (seed 42, layers [6, 12, 18, 24], resolved CLIP revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`). This is the project's final, sealed result — no further training, calibration, or retesting follows it.)*
  - [x] Attempt SAFE-versus-controlled training, adapters/final-block fine-tuning, self-training, ConvNeXt-Tiny, or external datasets only as separately versioned experiments after the baseline package works.
  - [x] Reject and roll back any optional experiment that fails the score, per-label, held-out-source, coverage, or resource gates.
  - [x] Complete when a fresh environment can reproduce training/evaluation artifacts and run the local JSON inference contract without external APIs. *(`run_inference.py <dir> --output-dir <dir>` reproduces the local JSON contract with zero flags; the full training/evaluation history is reproducible per notebook 00-11. Remaining gap, not blocking baseline completion: resource profiling on target hardware.)*

## Current progress and critical path

| Task | Current state | Recorded decision | Remaining dependency |
|---|---|---|---|
| 1. Project skeleton | Complete | Frozen Colab configuration and reproducibility helpers | None |
| 2. Data contract | Complete | 19,882 eligible primary images; fixed-Q96 selected later by Stage A | None |
| 3. Independent transforms | Complete | Notebook 07 materialized 19,460 variants from 1,390 clean parents across all 14 independent cells | None |
| 4. Frozen CLIP Stage A | Complete | Three-seed locked 50/50 mean: 94.71% | Superseded by controlled RINE for the Task 9 handoff |
| 5. Evaluation harness | Complete | Evaluated 20,850 development rows; `final_test` remained sealed | None before the Task 9 comparison |
| 6. RINE Stage B | Complete and retained | Controlled RINE: 100.00% clean, 99.62% robustness mean, 99.81% locked 50/50 mean | Use as the pre-Task-9 parent |
| 7. Frequency Stage 1 | Complete; fusion rejected | RINE+frequency mean 52.15% versus 99.81% parent; early exit remains disabled | No further run before Task 9 |
| 8. Color/physical auxiliaries | Complete; Lab fusion rejected | RINE+Lab mean 98.95% versus 99.81% parent and AI accuracy regressed 1.82 points; combined candidate skipped | No matched-data auxiliary retained |
| 8B. Native physical pilot | Complete; no physical feature retained | Matched nuisance B.Acc. 0.50 passed, but multi-image PRNU AUC 0.538 and single-image proxy AUC 0.543 both missed the 0.60 gate; CA coverage was zero | No binary physical fusion run; reopen only with materially better native device/lens evidence |
| 8B-v2. Improved PRNU estimator | Complete; fusion rejected | Corrected 256 px device test passed (AUC 0.859; top-1 0.657 versus 0.10 random), all 20,850 runtime rows extracted, and PRNU-only scored 78.09% locked; RINE+PRNU fell to 33.43% mean because two seeds collapsed | Keep controlled RINE; retain PRNU-v2 as diagnostic evidence only |
| 9. Texture path | Complete; tested and rejected | Clean gate passed (100% vs. 99.394% global-only), but Stage-1 robustness scored 93.13% mean vs. 99.80% for controlled RINE; locked-score deltas negative against both comparators; `resize_scale_0.25` AI-generated accuracy collapsed to 50.2% | Controlled RINE remains retained; no texture feature carried into Task 10 |
| 10. Packaging | Complete | `final_test`: 141 samples, 99.29% accuracy (100% AI-generated, 98.61% authentic), controlled RINE seed 42, T=1 | Resource profiling on target hardware; optional post-baseline experiments only |

Notebook 07 completed the post-Task-3 robustness rerun defined in [`docs/training/robustness_evaluation_scope.md`](../training/robustness_evaluation_scope.md). Notebook 08 corrected the initial 512 px PRNU protocol mismatch and completed the 256 px device-signal and binary ablations. Task 9 passed its clean gate but failed the frozen-checkpoint Stage-1 robustness gate. Task 10 then selected controlled RINE seed 42, kept raw sigmoid probabilities after the calibration fit proved degenerate, packaged local inference, and performed the one permitted sealed `final_test` run.

The robustness pipeline used the frozen Drive manifest, SHA-256 `aee4bd2e16fec2208cea4a7834a2c6b6086c5edfb9c5c64df21f54cc89ff3ef2`. A regenerated local manifest contained the same 2,000 `sample_id` values and differed only in environment-specific path columns, so it was not substituted for the frozen input.

## Evidence-based model decisions

Notebook 07 and 08 results below are development `selection_val` results. The locked 50/50 value is the mean of clean accuracy and the mean across all 14 independent robustness cells; it is separate from the one-time Task 10 `final_test` result. Device-separation AUC/PCE values are label-free; PRNU-only and RINE+PRNU values are separate authentic-versus-AI development ablations.

| Component | Development result | Confidence or failure evidence | Action |
|---|---:|---|---|
| Controlled RINE Stage B | **100.00% clean; 99.62% robustness; 99.81% locked** | Seeds 42/43/44 scored 99.85%, 99.81%, and 99.78% respectively; no class-regression gate failed | Retain as the pre-Task-9 parent |
| Existing clean-trained RINE | **96.20% locked** | Three-seed robustness rerun of the earlier clean-trained checkpoints | Superseded by controlled RINE |
| Frozen CLIP Stage A | **94.71% locked** | Three-seed robustness rerun on the same development bank | Retain only as the mandatory baseline comparator |
| RINE + frequency | **52.15% locked** | Mean delta -47.66 points; seeds 43 and 44 were unstable despite seed 42 being near the parent | Reject; keep early exit disabled |
| RINE + Lab | **98.95% locked** | Mean delta -0.87 points and AI-generated accuracy regressed 1.82 points, beyond the one-point gate | Reject |
| RINE + frequency + Lab | Not run | Both individual candidates failed their retention gates | Correctly skipped |
| Frequency magnitude/residual | **83.03% clean** | Best standalone frequency representation | Preserve for diagnostics only; fusion was rejected and early exit stays disabled |
| Frequency magnitude/residual/phase | 80.00% | Phase caused -3.03 overall, -4.49 AI-generated, and -1.32 authentic points | Drop phase; do not retrain this branch |
| Lab correlation | **82.42% clean** | Approximately 99-100% extractor confidence and 100% validity across both labels | Preserve for diagnostics only; fusion was rejected |
| RGB+Lab correlation | 78.79% | Adding RGB reduced Lab-only accuracy by 3.64 points | Drop this fixed concatenation; only revisit if a later learned projection proves incremental value |
| RGB correlation | 55.15% | Nearly chance despite approximately 99-100% confidence and 100% validity | Drop the current representation; low-priority redesign only |
| Native multi-image PRNU reference | **AUC 0.538** | 10 training devices, 100 disjoint reference images, 316 device-identity queries, 3,160 comparisons; same-device mean correlation 0.00266 versus 0.00166 across devices; below the predeclared 0.60 AUC gate | Reject for this pilot; do not train a binary projection/fusion head and do not claim camera authentication |
| Single-image PRNU coherence proxy | **AUC 0.543** | Device-separation validation used no authentic/AI labels; top-1 device accuracy 0.155 versus 0.10 random; still below the same 0.60 gate | Reject for this pilot; revisit only with a materially improved estimator and new held-out-device evidence |
| Native multi-image PRNU v2, 256 px | **AUC 0.859** | 10 devices, 166 disjoint queries, and 1,660 PCE comparisons; top-1 0.657 versus 0.10 random; same-device mean PCE 73.27 versus 11.25 different-device | Freeze as evidence that the estimator recovers known-device signal at the runtime-compatible crop; never use reference-bank PCE as a generic authenticity score |
| Reference-free PRNU v2 only | **85.25% clean; 70.92% robustness; 78.09% locked** | Three-seed diagnostic after 100% extraction coverage across all 20,850 rows | Preserve for diagnostics only; it is substantially weaker than controlled RINE |
| RINE + PRNU v2 | **33.43% locked** | Seeds 42/43 collapsed to 0.32%/0.43%; seed 44 reached 99.55% but still did not improve its parent; mean delta -66.39 points | Reject; keep PRNU disabled and controlled RINE unchanged |
| Chromatic aberration | Gate not eligible | Task 8B lens/focal metadata fraction 0.0 and edge-rich fraction 0.0; corrected/uncorrected calibration coverage is absent | Keep deferred; collect calibrated native lens/focal and edge-rich coverage before estimator validation or binary fitting |
| Radial distortion | Not run | Insufficient eligible line/arc support | Keep deferred |

High extractor confidence means the statistic was numerically measurable; it does **not** mean the classifier is correct. RGB is the clearest example: excellent coverage and confidence but only 55.15% clean accuracy. Task 8B v1 also confirms that eligible native data alone is insufficient: both original PRNU estimators failed their independent device-separation gate. V2 demonstrates repeatable device signal at 256 px, but the completed binary ablation shows that this does not translate into stable incremental authentic-versus-AI value.

### Completed robustness and fusion matrix

Notebook 07 completed the controlled RINE and incremental fusion matrix across seeds 42/43/44. Controlled RINE improved the locked mean from 96.20% for the existing clean-trained RINE to 99.81%. Frequency and Lab both failed the strict improvement gate, so neither was retained and the combined candidate was correctly skipped. Task 9 should compare its global-plus-patch candidate directly against this controlled-RINE parent under the same locked evaluation and resource budget.

Do not spend another training run on phase, frequency fusion, Lab fusion, the current RGB-only vector, RGB+Lab fixed concatenation, PRNU-v2 fusion, chromatic aberration, radial distortion, or full-backbone CLIP fine-tuning **on the current matched-Q96 handoff**. Their standalone tables remain diagnostic evidence, but their inference flags stay disabled. Task 8B is closed. Known-device PCE remains diagnostic only and never becomes a binary feature.

### Task 8B v1 decision and v2 follow-up

Task 8B is a completed separate evidence track, not a reason to block Task 3 or the viable RINE package. Its current decision is fail-closed: retain no physical feature and schedule no physical fusion run.

The repository contract now selects the accessible PREMIER v3 N1/N2 subsets for
the authentic side; N3 remains optional because it is described on the project
page but absent from the currently accessible public folder. GenImage `ai`
branches provide the synthetic side. Both live below the existing
`hackathon_data/raw/task8b` root, and outputs live below the existing
`artifacts/task8b` root. GenImage is accepted only under the explicit assumption
that this is non-commercial hackathon/research use; commercial reuse requires a
replacement dataset or separate permission. The supported handoff is local-first: extract into
`/content/hackathon_data/raw/task8b`, run `make task8b-inventory`, review the
generated inventory, then run `make task8b-prepare`. See
[the Task 8B dataset contract](../data/task8b_dataset.md).

The attached BigGAN, ADM, Stable Diffusion V1.4, and VQDM `.zip` files were only
the final volumes of multi-part archives and contained no local image payload;
they were permanently removed after explicit approval. Tiny-GenImage was then
approved as the storage-limited repackaging. GLIDE replaced low-resolution
BigGAN in the active pool; BigGAN remains stress-only. The prepared manifest has
640 rows per label across 13 devices and ADM, GLIDE, Midjourney, and Wukong.
Complete device and generator groups are frozen before derivation.

The source-original nuisance audit remains perfectly separable, so it is not a
training view. Deterministic 256 px crop-only RGB TIFF views normalize dimensions,
codec, metadata, and file size without resizing. Their nuisance balanced accuracy
is 0.50 and 1,164 rows remain eligible. The independently calibrated PRNU gate did
not pass: multi-image fingerprint AUC is 0.538 and the single-image proxy AUC is
0.543 against the predeclared 0.60 minimum. Chromatic aberration is ineligible
because lens/focal and edge-rich calibration coverage are absent. Task 8B is
therefore closed for this licensed pilot with no physical feature retained, no
fusion training, and no camera-authentication claim.

The decision authority is the generated evidence set under `artifacts/task8b`:
`audits/readiness_report.json`, `audits/matched_readiness_report.json`,
`audits/prnu_signal_validation.json`, and
`reports/retention_decision.json`. Preserve these together because the final
report hashes the readiness and PRNU inputs used to authorize the fail-closed
decision.

The bounded PRNU v2 follow-up is isolated under `artifacts/task8b_v2`. Its
original known-device validation changed the estimator, not the evidence threshold: 25 disjoint
references per eligible device, a native-coordinate 512 px crop without resize
or EXIF transposition, wavelet plus spectral cleanup, multiplicative reference
estimation, masked pixels, and PCE with an eight-pixel registration window. The
assumption is that encoded pixel coordinates remain stable within a PREMIER
device ID. The verified run reads only `seed_train` PREMIER device identities
and passes the unchanged gates: AUC 0.9171, top-1 accuracy 0.8554 versus 0.10
random, and mean same-device PCE 262.65 versus 11.31 across devices. This
validates repeatable device signal at 512 px only. The corrected 256 px test also
passes: AUC 0.8593, top-1 0.6566 versus 0.10 random, and same/different mean PCE
73.27/11.25. All 20,850 binary development views then extracted successfully.
PRNU-only reached 78.09% mean locked, while RINE+PRNU reached only 33.43%
because seeds 42 and 43 collapsed; seed 44 reached 99.55% but did not improve
its controlled-RINE parent. The strict retention decision is therefore `reject`.

This is a known-device reference experiment: every query is compared with a
fingerprint built from other images of that same device class. It does not test
an unseen camera, and synthetic inputs have no corresponding camera reference.
Therefore maximum PCE against the ten-device reference bank must not become a
generic `authentic` score; that would reward membership in the enrolled PREMIER
devices rather than general physical capture.

The binary candidate is the separately named, reference-free runtime vector
derived from the frozen v2 residual pipeline. Its summaries include
masked residual energy, spectral flatness, row/column periodicity,
luminance-residual coupling, and block consistency, but they must not include a
device ID, source metadata, or comparison with a known-device fingerprint.
The device-signal test was rerun at a predeclared 256 px native crop without
resize, followed by balanced matched-clean support at that same size. No
low-resolution image was resized merely to satisfy eligibility.
Deliberately downscaled robustness cells remain evaluation rows with zero PRNU
values and explicit eligibility, validity, and confidence masks. Fit any runtime
projection on `seed_train`, select it only
on `selection_val`, and use the complete-device/complete-generator
`heldout_test` once for confirmation; the competition `final_test` remains
sealed.

- Authentic data must contain native or minimally processed camera images, camera/device identifiers, multiple images per physical sensor, and enough distinct devices to hold out entire sensors during validation.
- The synthetic side must span diverse generator families and checkpoints. Resolution, codec, quality, and other export settings must be matched independently of label so the fusion head cannot learn file-pipeline shortcuts.
- Preserve immutable native originals. Keep source-original physical-feature experiments separate from matched derivatives and from independently transformed robustness views.
- Split authentic data by complete device, not individual image; split synthetic data by source/generator grouping. Report held-out-device, held-out-camera-model, and held-out-generator performance.
- For PRNU, compare the existing single-image coherence proxy with a classical multi-image reference fingerprint where device groups allow it. A reference-free runtime feature may support fusion, but cannot identify or authenticate an unknown camera by itself.
- For chromatic aberration, collect native edge-rich scenes and calibration targets across camera/lens combinations, focal lengths, and both corrected and uncorrected outputs. Use calibrated optical measurements to validate the fixed estimator before binary fusion training.
- Keep physical extractors fixed or separately calibrated; train only their projections and the fusion head from authentic/AI labels. Binary labels alone do not prove that a sensor or lens estimate is physically correct.
- Run the same independent JPEG, blur, resize, noise, jitter, and crop cells. Retain a physical family only if it improves the locked 50/50 score and held-out-device/generator results without increasing authentic false positives.

## Dependencies and parallel teammate tracks

Tasks are ordered by integration dependency, but implementation does not have to be fully sequential. Teammates should own separate module paths to avoid merge conflicts.

| Workstream | Can start | Hard dependency before integration | Can run alongside | Suggested ownership boundary |
|---|---|---|---|---|
| Task 2 data contract | Now | Task 1 complete | Tasks 3, 5, and standalone parts of 4/7/8/9 | `src/cya_detector/data/`, Task 2 scripts/notebook |
| Task 3 transform engine | Complete and robustness-verified | None | Task 9 and optional future ablations | `src/cya_detector/transforms/`, transform tests |
| Task 4 CLIP baseline | Complete and robustness-verified | None | Task 9 comparison | `src/cya_detector/models/clip_baseline.py` |
| Task 5 evaluation harness | Complete and robustness-verified | None | Task 9 and Task 10 | `src/cya_detector/evaluation/`, reporting tests |
| Task 7 frequency extraction | Complete; fusion rejected | None | Optional diagnostics only | `src/cya_detector/features/frequency.py` |
| Task 8 color features | Complete; Lab fusion rejected | None | Optional diagnostics only | `src/cya_detector/features/color.py` |
| Task 8 PRNU features | Complete; fusion rejected | None; evidence retained as diagnostics only | Task 9 texture work | `src/cya_detector/features/prnu_runtime_v2.py`, `notebooks/08_prnu_v2_binary.ipynb` |
| Task 8 optical features | Deferred after Task 8B | Reopen only after lens/focal metadata and calibrated edge-rich corrected/uncorrected coverage pass readiness | Frequency, color, PRNU, and texture tracks | `src/cya_detector/features/optics.py` |
| Task 9 texture path | Complete; tested and rejected at Stage-1 robustness | None; controlled RINE remains the sole retained model | None — no texture feature carries into Task 10 | `src/cya_detector/evaluation/texture_robustness.py`, `notebooks/09_texture_stage_d.ipynb`, `notebooks/10_texture_robustness_stage1.ipynb` |
| Task 6 RINE integration | Complete and retained | None | Task 10 packaging | `src/cya_detector/models/rine.py` |
| Task 10 packaging | Complete — `final_test` run (99.29% accuracy) | None | Documentation/demo preparation only | inference CLI, calibration, release tests, `final_test` report |

The final model decision is frozen: controlled RINE seed 42 is retained, while frequency, Lab, PRNU-v2 fusion, and Task 9 texture are rejected. Task 10 packaging and the one-time sealed `final_test` evaluation are complete. Only target-hardware resource profiling and separately versioned optional experiments remain open.

## Remaining follow-ups

- Measure model size, latency, peak memory, and disk/cache requirements on the target hardware.
- Treat SAFE, adapters or final-block tuning, self-training, ConvNeXt-Tiny, and external datasets only as separately versioned optional experiments; none changes the frozen baseline or permits another `final_test` read.
