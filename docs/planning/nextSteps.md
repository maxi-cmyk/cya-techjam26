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

## Current progress and critical path

| Task | Current state | Recorded decision | Remaining dependency |
|---|---|---|---|
| 1. Project skeleton | Complete | Frozen Colab configuration and reproducibility helpers | None |
| 2. Data contract | Complete | 19,882 eligible primary images; fixed-Q96 selected later by Stage A | None |
| 3. Independent transforms | Complete | No chained transforms; each cell derives directly from matched clean; 82 focused tests pass | Ready for the robustness rerun |
| 4. Frozen CLIP Stage A | Clean milestone complete | Fixed-Q96, 97.58% clean | Re-evaluate on Task 3 cells |
| 5. Evaluation harness | Clean milestone complete | Final-test lock and 50/50 formula implemented | Populate robustness tables from Task 3 |
| 6. RINE Stage B | Clean milestone complete | Provisionally retain, 99.39% clean | Confirm with Task 3 and class-regression gates |
| 7. Frequency Stage 1 | Clean milestone complete | Retain magnitude/residual at 83.03%; drop phase; early exit remains disabled | Robustness and incremental fusion after Task 3 |
| 8. Color/physical auxiliaries | Current matched-data milestone complete | Lab wins the color-only ablation at 82.42%; matched-Q96 remains ineligible for PRNU/CA | Color robustness/fusion after Task 3; PRNU follows the separate 8B-v2 gate |
| 8B. Native physical pilot | Complete; no physical feature retained | Matched nuisance B.Acc. 0.50 passed, but multi-image PRNU AUC 0.538 and single-image proxy AUC 0.543 both missed the 0.60 gate; CA coverage was zero | No binary physical fusion run; reopen only with materially better native device/lens evidence |
| 8B-v2. Improved PRNU estimator | Device-signal gate passed; binary usefulness pending | AUC 0.917; top-1 0.855 vs 0.10 random; same-device mean PCE 262.65 vs 11.31 different-device | Freeze the reference estimator; audit 512 px support and define a reference-free runtime vector before any binary ablation |
| 9. Texture path | Patch selector complete | Deterministic top-k non-overlapping Laplacian/Sobel selection passes fixtures | Implement global+patch encoding and benchmark cost |
| 10. Packaging | Not started | Final-test remains sealed | Wait for Tasks 3, 6-9 retention decisions |

The critical path is now the post-Task-3 robustness rerun defined in [`docs/training/robustness_evaluation_scope.md`](../training/robustness_evaluation_scope.md). Task 9 model integration waits for its pre-Task-9 retention decision, and no calibration or final-test run is valid before Tasks 9 and 10 freeze the final architecture.

## Evidence-based model decisions

Classifier accuracy values below are clean `selection_val` results. They are useful for pruning experiments, but they are not the final challenge score until the new robustness rerun populates every independent Task 3 cell. Task 8B PRNU values are label-free device-separation AUCs, not authentic-versus-AI accuracy.

| Component | Clean result | Confidence or failure evidence | Action |
|---|---:|---|---|
| RINE Stage B | **99.39%** | +1.82 points over Stage A; +2.63 authentic and +1.12 AI-generated points; no clean class collapse | Provisionally retain and retrain its head under the controlled Task 3 policy |
| Frozen CLIP Stage A | **97.58%** | Strong clean baseline on fixed-Q96 | Retain as the mandatory baseline and robustness comparator |
| Frequency magnitude/residual | **83.03%** | Best frequency representation | Keep only as an auxiliary fusion candidate; do not enable early exit |
| Frequency magnitude/residual/phase | 80.00% | Phase caused -3.03 overall, -4.49 AI-generated, and -1.32 authentic points | Drop phase; do not retrain this branch |
| Lab correlation | **82.42%** | Approximately 99-100% extractor confidence and 100% validity across both labels | Keep as the color-family fusion candidate and train with matched color-jitter distributions |
| RGB+Lab correlation | 78.79% | Adding RGB reduced Lab-only accuracy by 3.64 points | Drop this fixed concatenation; only revisit if a later learned projection proves incremental value |
| RGB correlation | 55.15% | Nearly chance despite approximately 99-100% confidence and 100% validity | Drop the current representation; low-priority redesign only |
| Native multi-image PRNU reference | **AUC 0.538** | 10 training devices, 100 disjoint reference images, 316 device-identity queries, 3,160 comparisons; same-device mean correlation 0.00266 versus 0.00166 across devices; below the predeclared 0.60 AUC gate | Reject for this pilot; do not train a binary projection/fusion head and do not claim camera authentication |
| Single-image PRNU coherence proxy | **AUC 0.543** | Device-separation validation used no authentic/AI labels; top-1 device accuracy 0.155 versus 0.10 random; still below the same 0.60 gate | Reject for this pilot; revisit only with a materially improved estimator and new held-out-device evidence |
| Native multi-image PRNU v2 | **AUC 0.917** | 10 devices, 250 disjoint reference images, 166 queries, and 1,660 PCE comparisons; top-1 0.855 versus 0.10 random; same-device mean PCE 262.65 versus 11.31 different-device | Freeze as evidence that the extraction method recovers known-device signal; do not use reference-bank PCE as a generic authenticity score or retain PRNU until a reference-free locked binary ablation passes |
| Chromatic aberration | Gate not eligible | Task 8B lens/focal metadata fraction 0.0 and edge-rich fraction 0.0; corrected/uncorrected calibration coverage is absent | Keep deferred; collect calibrated native lens/focal and edge-rich coverage before estimator validation or binary fitting |
| Radial distortion | Not run | Insufficient eligible line/arc support | Keep deferred |

High extractor confidence means the statistic was numerically measurable; it does **not** mean the classifier is correct. RGB is the clearest example: excellent coverage and confidence but only 55.15% clean accuracy. Task 8B v1 also confirms that eligible native data alone is insufficient: both original PRNU estimators failed their independent device-separation gate. V2 now demonstrates repeatable device signal, but that is still not evidence that it improves authentic-versus-AI classification.

### Required retraining and fusion matrix

Task 3 is merged and validated. Run this matrix now:

1. RINE with the backbone frozen and its head retrained under the clean-or-one-controlled-transform sampler, seeds 42/43/44.
2. RINE plus magnitude/residual frequency, training only the frequency projection and fusion head.
3. RINE plus Lab, training only the Lab projection and fusion head with identical jitter sampling for both labels.
4. RINE plus magnitude/residual frequency plus Lab, but only if both individual additions pass the 50/50 and per-class gates.
5. Global RINE plus the Task 9 texture aggregator under a fixed patch budget, including latency and resize robustness.
6. RINE plus a reference-free PRNU-v2 runtime vector only after its 512 px class-support audit, nuisance gate, and standalone binary diagnostic pass. The known-device PCE score is not an inference feature.

Do not spend another training run on phase, the current RGB-only vector, RGB+Lab fixed concatenation, chromatic aberration, radial distortion, or full-backbone CLIP fine-tuning **on the current matched-Q96 handoff**. Task 8B v1 remains closed. PRNU v2 passed its independent device-signal gate, but it may enter only a separately predeclared binary usefulness ablation—not fusion retention—after Task 3 establishes the locked robustness evaluation. Full CLIP fine-tuning remains optional until the frozen pipeline is complete.

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
implementation changes the estimator, not the evidence threshold: 25 disjoint
references per eligible device, a native-coordinate 512 px crop without resize
or EXIF transposition, wavelet plus spectral cleanup, multiplicative reference
estimation, masked pixels, and PCE with an eight-pixel registration window. The
assumption is that encoded pixel coordinates remain stable within a PREMIER
device ID. The verified run reads only `seed_train` PREMIER device identities
and passes the unchanged gates: AUC 0.9171, top-1 accuracy 0.8554 versus 0.10
random, and mean same-device PCE 262.65 versus 11.31 across devices. This
validates repeatable device signal only; no binary fusion is authorized until a
separately locked usefulness ablation passes.

This is a known-device reference experiment: every query is compared with a
fingerprint built from other images of that same device class. It does not test
an unseen camera, and synthetic inputs have no corresponding camera reference.
Therefore maximum PCE against the ten-device reference bank must not become a
generic `authentic` score; that would reward membership in the enrolled PREMIER
devices rather than general physical capture.

The next candidate must be a separately named, reference-free runtime vector
derived from the frozen v2 residual pipeline. Candidate summaries may include
masked residual energy, spectral flatness, row/column periodicity,
luminance-residual coupling, and block consistency, but they must not include a
device ID, source metadata, or comparison with a known-device fingerprint.
Before fitting it, audit both labels for native crop support at 512 px without
resize. If support is insufficient or class-imbalanced, acquire additional
licensed 512 px-or-larger synthetic exports or stop; do not resize low-resolution
generators merely to satisfy eligibility. Build label-independent native crops,
rerun the nuisance-only audit, and require balanced eligibility before reading
authentic/AI labels. Fit any runtime projection on `seed_train`, select it only
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
| Task 3 transform engine | Complete | Frozen config/schema from Task 1 | Tasks 2 and 5 | `src/cya_detector/transforms/`, transform tests |
| Task 4 CLIP loader/head skeleton | Now | Real training waits for Tasks 2 and 3 | Tasks 2, 3, and 5 | `src/cya_detector/models/clip_baseline.py` |
| Task 5 evaluation harness | Now, using fixtures | Real tables wait for Tasks 2–4 | Tasks 2, 3, 4, and feature extraction | `src/cya_detector/evaluation/`, reporting tests |
| Task 7 frequency extraction | Now, using fixtures | Fusion/retention waits for Tasks 4 and 5 | Tasks 2, 3, 5, 8, and 9 | `src/cya_detector/features/frequency.py` |
| Task 8 color features | Now, using fixtures | Fusion/retention waits for Tasks 4 and 5 | Frequency, PRNU, optics, and texture tracks | `src/cya_detector/features/color.py` |
| Task 8 PRNU features | V1 closed; v2 known-device signal validated | Reference-free feature definition, 512 px class-support audit, nuisance gate, then predeclared binary/robustness ablation | Frequency, color, optics, and texture tracks | `src/cya_detector/features/prnu_reference_v2.py` |
| Task 8 optical features | Deferred after Task 8B | Reopen only after lens/focal metadata and calibrated edge-rich corrected/uncorrected coverage pass readiness | Frequency, color, PRNU, and texture tracks | `src/cya_detector/features/optics.py` |
| Task 9 patch selector | Now, using fixtures | Learned aggregation waits for Task 4 | Tasks 2, 3, 5, 7, and 8 | `src/cya_detector/features/texture.py` |
| Task 6 RINE integration | After Stage A runs | Tasks 4 and 5 | Later auxiliary fusion experiments | `src/cya_detector/models/rine.py` |
| Task 10 packaging | After feature retention | Selected outputs from Tasks 4–9 | Documentation/demo preparation only | inference CLI, calibration, release tests |

The robustness rerun is the active gate. Task 9 global-plus-patch implementation may proceed only after the pre-Task-9 model decision is frozen. A Task 10 inference-CLI skeleton may be developed without selecting weights or reading `final_test`, but calibration, architecture freeze, and final evaluation remain blocked.

## How to proceed from here

1. **Task 3 complete.** Fixture tests prove one transform per row, parent/child split integrity, deterministic seeded outputs, exact resize restoration dimensions, and matched transform distributions across labels.
2. **Materialize the selection robustness views.** Produce every JPEG, blur, resize, noise, color-jitter, and crop parameter cell directly from fixed-Q96 `matched_clean`, never from another transformed image.
3. **Populate the locked evaluation.** Score Stage A and RINE on clean plus every Task 3 cell, then compute R.Acc., F.Acc., per-cell accuracy, confusion matrices, calibration diagnostics, robustness mean, and the 50% clean / 50% robustness score.
4. **Re-test retained auxiliary candidates.** Use magnitude/residual for frequency and Lab for color. Keep phase dropped. Evaluate CLIP/RINE-only, each feature-only diagnostic, and incremental RINE+frequency, RINE+Lab, and RINE+frequency+Lab fusion across seeds 42/43/44.
5. **Preserve Task 8B v1 and qualify a runtime PRNU-v2 candidate.** Keep the v1 evidence unchanged and freeze the passing v2 reference protocol. Do not use maximum PCE against its ten known devices as authenticity evidence. First define a reference-free single-image vector, audit native 512 px support across both labels, build label-independent crop-only views, and pass the nuisance and balanced-eligibility gates. Only then compare RINE-only, PRNU-only diagnostic, and RINE+PRNU across seeds 42/43/44 on clean plus every Task 3 transform. Retain it only if the locked 50/50 score strictly improves and neither class regresses by more than the configured one point. Chromatic aberration and radial distortion remain deferred under their existing data gates.
6. **Complete Task 9 in parallel.** Preserve the global CLIP view, encode a fixed patch budget, train only the aggregator/fusion head, and compare global-only, local-only, and combined accuracy plus latency. Resize 0.5x/0.25x results wait for Task 3.
7. **Freeze and package only after those gates.** Select the architecture by the locked 50/50 score and per-class regression limits, fit temperature once on clean `selection_val`, implement the directory JSON contract, and then run sealed `final_test` once.

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
  - [ ] Complete when a small fixture dataset can regenerate byte-identical deterministic variants and statistically matched stochastic policies.

[ ] **Task 4 — Build the frozen-CLIP Stage A baseline** *(clean Stage A verified; robustness integration waits for Task 3)*

  - [x] Load the vision tower only and lock the exact CLIP variant/input size before producing caches; record the resolved model commit in cache keys.
  - [x] Freeze the backbone and train a linear classifier first, with the MLP available only through an explicit ablation flag.
  - [x] Extract embeddings from the exact received training views; never run matched normalization or robustness probing inside inference.
  - [x] Cache fixed-view embeddings using image hash, resolved model revision, preprocessing version, and view identifier as the key.
  - [x] Report throughput, peak VRAM, cache size, and accumulation toward effective batch size 32; select the physical microbatch in Colab.
  - [x] Train with AdamW/BCE across seeds 42, 43, and 44 for both Task 2 matching candidates; fixed-Q96 won the clean `selection_val` comparison.
  - [x] Save latest and best-clean checkpoints now; create best-robustness and best-50/50 only when Task 3 cells exist rather than fabricating scores.
  - [ ] Complete when clean and every independent transform cell produce reproducible R.Acc., F.Acc., accuracy, confusion matrices, and logits.

The verified T4 run completed all six policy/seed combinations and persisted them to Drive. Fixed-Q96 seed 42 extracted 1,390 embeddings in 38.4 seconds at 36.2 images/second, used approximately 1.43 GB peak GPU allocation, and reached 97.58% best clean accuracy. Fixed-Q96 is the provisional clean-only policy; the locked policy decision is revisited once Task 3 enables the 50/50 score.

[ ] **Task 5 — Build the locked evaluation and reporting harness** *(clean reports verified; robustness tables wait for Task 3)*

  - [x] Compute clean accuracy and mean accuracy across all independent transform-and-parameter cells using the agreed 50/50 formula.
  - [x] Report R.Acc., F.Acc., confusion matrix, ECE, false-positive rate, and false-negative rate for clean data and each transform row.
  - [x] Add generator/checkpoint/source breakdowns and an `unknown` metadata bucket without changing the binary public label.
  - [x] Add deterministic stratified bootstrap confidence intervals for retained model comparisons and future fast-track claims.
  - [x] Make `selection_val` the only source for ordinary checkpoint, hyperparameter, feature-retention, and calibration decisions.
  - [x] Prevent normal experiment commands from reading `final_test`; require both explicit final-evaluation and architecture-frozen flags.
  - [x] Generate machine-readable JSON metrics and a CSV robustness table.
  - [x] Verify with fixtures that a constant or class-collapsed predictor is exposed by per-label metrics.
  - [ ] Complete the integration gate when real Task 4 logits cover clean data and every independent Task 3 transform cell.

Task 5 clean reports for both policies and all three seeds were generated and copied to Drive. The final-test lock remained in place. No robustness mean or 50/50 score is claimed before Task 3 produces the independent transform cells.

[ ] **Task 6 — Add the RINE-style Stage B representation** *(clean ablation verified and provisionally retained; robustness waits for Task 3)*

  - [x] Expose predeclared CLIP layers 6/12/18/24 without unfreezing the backbone and cache their CLS representations by immutable view/model keys.
  - [x] Train only softmax layer importance and one binary head on fixed-Q96 with the same manifest, splits, seeds, and optimization contract as Stage A.
  - [x] Implement paired Stage A versus Stage B clean comparison with the predeclared one-point per-class regression tolerance; transformed comparison waits for Task 3.
  - [x] Record added cache size, extraction throughput, training history, layer importance, and peak GPU memory.
  - [x] Run `03_rine_stage_b.ipynb` for seeds 42/43/44 and record the provisional clean keep/drop result.
  - [ ] Retain Stage B only if it improves the locked 50/50 criterion without breaching R.Acc./F.Acc. regression limits after Task 3 integration.
  - [ ] Complete when Stage A versus Stage B is a reproducible, apples-to-apples ablation with a recorded keep/drop decision.

The verified T4 run extracted layers 6/12/18/24 for 1,390 fixed-Q96 images in 39.6 seconds at 35.1 images/second and approximately 1.77 GB peak GPU allocation. Across seeds 42/43/44, RINE reached 99.39% mean clean accuracy versus Stage A's 97.58%, with +2.63-point authentic and +1.12-point AI-generated mean accuracy changes. The provisional clean decision is `retain`; the final decision remains pending Task 3's robustness cells and the locked 50/50 score.

[x] **Task 7 — Implement deterministic Stage 1 frequency features** *(clean family decision verified; robustness/fusion waits for Task 3)*

  - [x] Extract FFT/DCT log-magnitude summaries, radial/angular power, periodic-peak prominence, residual autocorrelation, and local neighboring-pixel dependencies.
  - [x] Keep generator/checkpoint/source metadata as evaluation strata rather than prediction targets.
  - [x] Fit feature scaling on `seed_train` only and cache versioned vectors with validity indicators and extraction errors.
  - [x] Train magnitude/residual and magnitude/residual-plus-bounded-phase frequency-only classifiers; RINE fusion waits for clean representation selection and Task 3 integration.
  - [x] Implement paired magnitude-versus-phase comparison; JPEG and resize rows wait for Task 3.
  - [x] Audit the strongest correlations with file size, dimensions, and normalization quality on `seed_train`.
  - [x] Keep the Stage 1 early exit disabled in configuration and training; any future early-exit claim still requires locked precision, confidence, coverage, authentic, held-out-family, and transform gates.
  - [x] Run `04_frequency_stage1.ipynb` and record the clean representation decision across seeds 42/43/44.
  - [x] Complete the clean frequency-only decision while keeping the default inference path falling through to Stage 2; incremental fusion remains a Task 3 integration item.

The verified CPU run completed all six representation/seed combinations. Magnitude plus residual features reached 83.03% mean clean accuracy versus 80.00% with bounded phase added. The phase features were dropped, magnitude is the retained clean representation, `final_test` remained sealed, and the Stage 1 early exit remained disabled.

[x] **Task 8 — Add auxiliary feature families one at a time** *(current matched-data milestone verified; robustness/fusion waits for Task 3)*

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
  - [x] Complete the clean matched-view family decision with documented physical limitations and coverage; incremental RINE fusion and color-jitter robustness remain Task 3 integration work.

The verified CPU run extracted 67 features for 1,390 rows and completed all nine color representation/seed combinations without notebook errors. Lab reached 82.42% mean clean accuracy, RGB+Lab reached 78.79%, and RGB reached 55.15%, so Lab is the selected color representation for the later fusion ablation. RGB and Lab validity was effectively complete for both labels. PRNU and chromatic-aberration eligibility was 0% for both labels because the evaluated views are matched/recompressed rather than native originals; neither family was trained or retained, radial distortion remains deferred, and no camera or lens authenticity claim is made.

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
  - [ ] Define and version a reference-free `prnu_v2_runtime` single-image vector derived from the frozen residual pipeline, with no device ID, metadata, or known-reference comparison.
  - [ ] Audit authentic and AI-generated rows for balanced native 512 px crop support; acquire additional licensed high-resolution generator exports or stop if the gate fails, and never upsample rows into eligibility.
  - [ ] Materialize label-independent native crop-only views and pass a nuisance-only gate before fitting the runtime PRNU diagnostic.
  - [ ] Fit on `seed_train`, select on `selection_val`, and confirm once on grouped `heldout_test`; never read the competition `final_test` for this decision.
  - [ ] After Task 3, run PRNU-only, RINE-only, and RINE+PRNU across seeds 42/43/44 and every independent transform; retain only on a strict locked-50/50 improvement with at most one-point regression for either class.
  - [x] Retain no Task 8B physical feature and make no camera or lens authenticity claim.

[ ] **Task 9 — Add the texture-aware local-detail path under a fixed budget**

  - [x] Generate multi-scale Laplacian/Sobel energy maps and select fixed top-k non-overlapping patches before CLIP input conversion.
  - [x] Add deterministic fixture coverage for non-overlap, stable ordering, top-k bounds, high-detail selection, small images, and invalid parameters.
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
