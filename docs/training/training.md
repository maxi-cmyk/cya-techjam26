# Training runbook and recorded outcomes

This document records the operational training contract and the resulting model decisions for the binary `authentic` versus `ai_generated` detector. It implements the product boundary in [PRD.md](../product/PRD.md), the architecture in [design.md](../architecture/design.md), and the components in [models.md](../architecture/models.md) and [techStack.md](../architecture/techStack.md). Historical procedures remain here for reproducibility; the retained baseline is already frozen.

## 1. Training Objectives and Non-Negotiable Rules

The model produces one logit and one public value, `pred = sigmoid(logit)`. Temperature fitting was attempted after checkpoint selection but rejected as degenerate, so the shipped baseline uses `T=1` and threshold `0.5`; `pred` is a raw model probability-like score, not a separately calibrated probability guarantee.

Training must preserve these rules:

- Use only fully authentic source images and fully AI-generated images.
- Exclude mixed-origin, AI-edited, face-swapped, composited, inpainted-authentic, and ambiguous-provenance samples.
- Keep every clean source and all variants derived from it in one split.
- A transformed sample receives exactly one benchmark transform and one parameter setting.
- Never chain, mix, blend, or overlay benchmark transformations.
- Apply the same transform distributions to both labels.
- Keep the public task binary. Generator/decoder metadata is for sampling and evaluation, not an output class.
- Keep Stage 1's frequency fast-track disabled unless the locked enablement gate passes.
- Treat absent PRNU, color, or optical evidence as neutral.
- Select checkpoints using the agreed score:

`selection_score = 0.50 × clean_accuracy + 0.50 × mean_independent_transform_accuracy`

R.Acc. and F.Acc. remain mandatory guardrails even when the aggregate score improves.

### 1.1 Trainability audit

The architecture is trainable, but the word **parameter** must be used carefully. The benchmark transform values are augmentation settings, not learned weights. Likewise, most forensic extractors are fixed algorithms whose outputs are learned by a small downstream projection/fusion head.

| Component | What is learned | Supervision currently available | Feasibility decision |
|---|---|---|---|
| Frozen CLIP baseline | Linear binary head | Binary authentic/AI labels | **Completed baseline**: 94.71% three-seed mean locked score; superseded by controlled RINE |
| RINE-style layer fusion | Layer-importance estimator and binary head | Binary authentic/AI labels | **Completed and retained**: 100.00% clean, 99.62% robustness, and 99.81% mean locked score with CLIP frozen; seed 42 is packaged |
| Texture path | Patch aggregation/attention and fusion weights | Binary labels on clean views; frozen-checkpoint robustness evaluation | **Completed and rejected**: clean gate passed, but `global_local` reached 93.13% Stage-1 robustness versus 99.80% controlled RINE |
| Frequency path | Feature projection, fusion weights, and a frequency-only classifier | Binary labels plus generator/decoder metadata for stratified validation | **Completed and rejected for fusion**: 52.15% mean locked versus 99.81% parent; extractor outputs remain diagnostic only |
| Reference-free PRNU-v2 path | Feature projection and fusion weights | Binary labels on clean and independent-transform rows | **Completed and rejected**: 78.09% PRNU-only locked score; RINE+PRNU 33.43% mean versus 99.81% parent after two-seed collapse; known-device fingerprints/PCE remain separate evidence and never become classifier inputs |
| RGB/Lab correlation path | Feature projection and fusion weights | Binary labels on matched jitter distributions | **Completed and rejected for fusion**: Lab was the best clean color vector, but RINE+Lab scored 98.95% locked and breached the AI-generated regression gate |
| Chromatic-aberration/radial-distortion path | Feature projection and fusion weights | Requires independently calibrated optical evidence before binary fusion | **Not eligible on current data**: calibrated lens/focal/edge-rich coverage is absent; radial distortion also lacks line/arc support |
| Stage 1 early exit | Frequency-only scorer and a fixed threshold selected on validation | Binary labels, held-out generator strata, and independent transforms | **Disabled**: the frequency branch did not pass its retention and enablement gates |
| Temperature scaling | One scalar temperature | Clean `selection_val` logits and labels | **Attempted and not retained**: the error-free 165-row clean set drove the fit to the 0.05 lower bound; baseline remains `T=1` |
| Selective CLIP fine-tuning | Adapters or final vision blocks | Binary labels | **Conditional** on an actual GPU/VRAM/time check; not required for the viable baseline |
| ConvNeXt-Tiny comparison | Full comparison model | Binary labels | **Possible but optional** and lower priority than proving the frozen-CLIP pipeline |
| C2PA verification | Nothing | Signed provenance manifest | **Not a training task** |
| JPEG/blur/resize/noise/jitter/crop values | Nothing | Fixed by the challenge protocol | **Not trainable**; they are sampled augmentation/evaluation settings |

The authentic/AI label is enough to learn whether a deterministic auxiliary vector adds predictive value. It is not enough to claim that the underlying PRNU, chromatic-aberration, or lens-distortion estimate is physically correct. Those extractors must therefore remain confidence-masked experimental inputs and pass the locked ablations before shipping.

### 1.2 Dataset support by signal

The named datasets do not support every feature equally:

| Data source | Suitable use | Restriction |
|---|---|---|
| SID_Set | Primary high-resolution binary pool after filtering | Keep only authentic and fully synthetic rows; exclude its tampered class entirely |
| PREMIER v3 N1/N2 accessible; N3 optional | Task 8B native authentic pool for device-grouped PRNU work | Accept only licensed native/minimally processed images with device IDs; preserve RAW/HEIC but defer them until decoding is pinned |
| GenImage | Task 8B synthetic pool only | The completed pilot used AI branches under the non-commercial research assumption; preserve generator grouping and never treat derivatives as independent sources |
| WildFake | Optional future generator-stratified experiment | Not acquired or used in the frozen baseline; verify the exact hierarchy and licenses before any use |
| CIFAKE | Optional low-resolution stress test | Not used in the frozen baseline; 32x32 sources are ineligible for physical or primary texture claims, and upscaling cannot recreate those signals |
| COCO plus DALL-E benchmark subsets | Optional external comparison | Not used for fitting, calibration, or final model selection |
| Chameleon | Optional external stress test | Not used for training or retention decisions |

Every row must include a per-feature eligibility mask derived from source resolution, provenance, processing history when known, and extractor support. Eligibility rules are fixed before model comparison and applied identically to both labels. Low-resolution images may still train the CLIP head, but they must not teach the fusion head that a missing physical feature implies either class.

High resolution does not prove that sensor or lens traces survived. Social-media and ImageNet-derived authentic files may already have been resized, denoised, sharpened, corrected, or re-encoded. Until a provenance-preserving authentic subset with near-native camera exports is identified, PRNU and optics are **unvalidated research ablations**, not dependable training signals. A classifier improvement on source-confounded web data is insufficient evidence to retain them.

For Task 8B specifically, accessible PREMIER N1/N2 supplies the authentic side
(N3 may be added if it later becomes accessible under the same license), and only
GenImage `ai` branches supply the synthetic side. GenImage `nature` rows are not
eligible because they lack the device-grouped native provenance required by this
track. This selection assumes non-commercial hackathon/research use under
GenImage's CC BY-NC-SA 4.0 terms. Task 8B uses separate grouped splits and never
relabels or merges its held-out rows into the competition `final_test`.

Task 8B source readiness and training readiness are separate gates. A valid
native collection may pass license, provenance, device, generator, and leakage
checks while still failing the nuisance-only classifier threshold. In that case,
reference fingerprints may be prepared for controlled PRNU comparison, but no
binary physical-fusion head may be fitted until label-independent export matching
reduces the nuisance balanced accuracy to the predeclared limit.

The completed Task 8B pilot meets that matched-view gate at 0.50 balanced
accuracy using 256 px crop-only, uncompressed RGB TIFF views. This does not by
itself authorize fusion training. The separate seed-train-only device validation
reaches AUC 0.538 for multi-image fingerprints and 0.543 for the single-image
proxy, both below the frozen 0.60 minimum. CA is also ineligible because calibrated
lens/focal/edge-rich coverage is absent. Consequently no PRNU/CA projection or
fusion weights are trained, RINE remains unchanged, and no physical feature is
retained by the original pilot. The corrected reference-free PRNU-v2 follow-up
then achieved 78.09% locked alone, while RINE+PRNU-v2 fell to 33.43% after two
seeds collapsed. PRNU-v2 was therefore also rejected and remains diagnostic.

### 1.3 Current readiness status

Training, robustness selection, packaging, and the one-time sealed final evaluation are complete. The retained model is controlled RINE seed 42 over frozen CLIP-ViT-L/14-336 layers 6/12/18/24, with fixed-Q96 matching, `T=1`, threshold `0.5`, and every auxiliary feature flag disabled. The sealed `final_test` result is 99.29% (140/141): 100.00% AI-generated accuracy and 98.61% authentic accuracy.

The repository contains the pinned seed-42 RINE head and its clean prediction artifact, plus Task 8B evidence. Large datasets, CLIP weights, transformed image banks, and most run artifacts remain external in Colab/Drive by design. The only baseline verification gap is target-hardware resource profiling; SAFE, self-training, selective CLIP fine-tuning, ConvNeXt-Tiny, and external datasets are optional new experiments and may not trigger another read of the sealed `final_test`.

## 2. Dataset Manifest

Create one immutable manifest before training. Each row represents a clean source or one independently transformed variant.

Required fields:

| Field | Purpose |
|---|---|
| `sample_id` | Stable unique identifier |
| `source_id` | Groups an immutable original with all derived views and variants |
| `parent_id` | Identifies the direct parent of a derivative or transform |
| `source_path` | Path to the immutable original bytes |
| `image_path` | Path to the concrete image file |
| `clean_image_path` | Path to the canonical matched clean parent |
| `image_view` | `source_original`, `matched_clean`, or `robustness_variant`; runtime input is `received_view` |
| `sha256` | Exact-file duplicate detection |
| `perceptual_hash` | Near-duplicate screening |
| `label` | `authentic` or `ai_generated` |
| `split` | `seed_train`, `self_train_pool`, `selection_val`, or `final_test` |
| `dataset_name` | Source dataset |
| `license_status` | Verified license/provenance status |
| `transform` | `clean`, `jpeg`, `blur`, `resize`, `noise`, `color_jitter`, or `center_crop` |
| `transform_parameter` | Exact quality, sigma, scale, or crop value |
| `transform_seed` | Reproduces stochastic noise/jitter |
| `width`, `height`, `format` | Nuisance-distribution checks |
| `normalization_codec` | Codec used for the canonical matched derivative |
| `normalization_quality` | Label-independent matched quality value |
| `encoder_version` | Encoder library and version |
| `chroma_subsampling` | JPEG chroma-subsampling setting |
| `original_format` | Format of the immutable source |
| `estimated_original_quality` | Estimated pre-ingestion JPEG quality, when applicable |
| `quantization_table_hash` | Compression-table audit identifier |
| `resize_scale` | `0.5` or `0.25` for a resize round trip |
| `down_interpolation`, `up_interpolation` | Locked interpolation modes; both `bilinear` for the benchmark |
| `resize_library`, `resize_library_version` | Exact resize implementation |
| `antialias` | Locked antialias setting |
| `dimension_rounding` | Rule used for intermediate dimensions |
| `intermediate_width`, `intermediate_height` | Reproducible downsample dimensions |
| `original_width`, `original_height` | Dimensions restored by the upsample step |
| `output_storage_format` | Lossless format or in-memory array representation |
| `generator_paradigm` | GAN, diffusion, autoregressive, hybrid, or `unknown` |
| `generator_name` | Generator/model name when known |
| `generator_checkpoint` | Checkpoint/version when known |
| `decoder_family` | CNN, VAE, VQ, pixel-space, or `unknown` |
| `tokenizer_family` | Tokenizer metadata when applicable |
| `upsampling_factor` | Known decoder scale factor or `unknown` |
| `capture_source` | Camera/dataset stratum for authentic images when known |

Do not infer missing generator metadata from model predictions. Store it as `unknown`.

### Provenance screening

Before a sample enters the manifest:

1. verify its source and license;
2. confirm that its label satisfies the binary definition;
3. reject uncertain or mixed-generation histories;
4. calculate exact and perceptual hashes;
5. inspect cross-dataset duplicates before assigning splits.

## 3. Split Strategy and Leakage Prevention

Use the existing approximate 60/25/15 development split:

- **Seed training set (~60%)** — verified labels used for supervised training.
- **Self-training pool (~25%)** — labels retained for audit but hidden from the training loop until evaluation/review.
- **Locked evaluation block (~15%)** — divided once, before experiments, into:
  - `selection_val` for checkpoint selection, ablation decisions, and temperature fitting;
  - `final_test` kept sealed until the chosen pipeline is frozen.

The exact subdivision of the 15% block depends on dataset size, but it must be recorded before the first run and never changed to improve results.

Group splits by `source_id`, not file path. Also hold out complete generator checkpoints—and, where volume permits, complete generator or decoder families—to measure generalization. Authentic images should likewise include held-out camera/source-dataset strata so the model cannot memorize one real-image collection.

Never use `final_test` for:

- hyperparameter selection;
- auxiliary-feature retention;
- pseudo-label thresholds;
- calibration;
- early stopping;
- Stage 1 fast-track thresholds.

## 4. Training Augmentations and Independent Transform Generation

Keep every ingested original immutable. Before any re-encoding, run C2PA on the original bytes and preserve the metadata needed for provenance auditing. Native PRNU, frequency, color, and optics features may be extracted from this original view for an explicitly separated ablation.

Create a canonical matched clean derivative for primary-model training by decoding and re-encoding both labels with the same encoder, color conversion, chroma subsampling, metadata policy, and quality distribution. Every setting must be sampled independently of the label. Start by comparing fixed Q96 with a narrow high-quality distribution such as Q95-Q100; select the policy on the locked bias audit and held-out results rather than clean accuracy alone.

Generate each robustness variant directly from its canonical matched clean parent. These same controlled variants are selected randomly during training to simulate redistribution:

| Transform | Parameter cells | Real-world analogue |
|---|---|---|
| JPEG compression | quality = 90, 70, 50, 30 | Social-media re-encoding and messaging |
| Gaussian blur | sigma = 0.5, 1.0, 2.0 | Out-of-focus capture and screenshot smoothing |
| Resize and restore | 0.5x and 0.25x; bilinear down and bilinear up | Thumbnail generation and CDN resizing |
| Gaussian noise | sigma = 0.02, 0.05, 0.10 on normalized pixels | Low-light sensor noise |
| Color jitter | brightness/contrast/saturation within +/-20% | Filter apps and auto-enhancement |
| Center crop | crop 80% | Profile-picture cropping and reframing |

“Randomly” means sampling from this table, not constructing an arbitrary augmentation chain:

- a training draw is either clean or receives exactly one transform;
- a transformed draw selects one transform-and-parameter cell;
- no second benchmark transform is applied afterward;
- the exact transform, realized parameters, and random seed are logged;
- both labels use the same transform probabilities and parameter distributions.

Variants may be materialized ahead of time or generated on demand. Ahead-of-time variants are selected randomly by the sampler. On-demand variants must be deterministic from `source_id`, run seed, transform, and parameter, and the realized values must be written to the run manifest.

Color jitter is one named benchmark operation even when it changes brightness, contrast, and saturation together. It must not be followed by another benchmark transform.

For a resize row, load the canonical matched clean parent, calculate intermediate dimensions with the locked rounding rule, bilinearly downsample once, and bilinearly restore to the exact parent width and height. The 0.5x and 0.25x severities are separate cells and are never derived from one another. Retain the result in memory or save it losslessly; saving it as JPEG would add a second benchmark effect.

Matched re-encoding is offline dataset baseline construction, not a benchmark transformation, and is never rerun by the inference script. At runtime every branch uses the exact `received_view`. Normal model preparation—decoding, tensor conversion, and fixed CLIP conversion—is not a benchmark transform. Select local patches from the received image at its available resolution before global model-size conversion, retain the global view for context, and encode the selected crops separately.

Use the same normalization-quality distribution, transform-cell probabilities, parameter distributions, and seed policy for both labels. Never overwrite originals. Dataset-level matched re-encoding removes compression-history bias; JPEG Q90/Q70/Q50/Q30 augmentation separately teaches degradation robustness and does not replace the matched pass.

## 5. Batch Sampling

Naively expanding every clean image into all transform variants would make robustness examples dominate training. Instead, use a hierarchical sampler aligned with the 50/50 score:

1. choose **clean** or **transformed** with equal probability;
2. if transformed, choose uniformly across transform-and-parameter cells, treating resize-0.5x and resize-0.25x as separate cells;
3. choose `authentic` or `ai_generated` with equal probability;
4. choose a generator/decoder stratum for AI images or a capture/dataset stratum for authentic images;
5. sample a `source_id`, then load the matching clean or transformed row.

This makes clean and robustness learning equally visible without duplicating one source across a batch. If exact balance is impossible, log the realized distribution for every epoch.

Do not add undocumented random image augmentations. Horizontal flips, random crops, additional blur, or extra color changes would create training conditions outside the controlled protocol unless they are explicitly approved and represented in the manifest.

## 6. Precomputation and Train-Only Statistics

Deterministic auxiliary features may be cached to reduce training cost:

- frequency feature bank;
- reference-free PRNU-v2 vector and support masks;
- RGB/Lab correlation vector;
- chromatic-aberration and optional radial-distortion vector;
- validity, confidence, and coverage masks;
- texture-patch coordinates;
- frozen CLIP global and patch embeddings when the image pipeline is fixed.

Cache keys must include:

- image SHA-256;
- transform and parameter;
- extractor name/version;
- preprocessing version;
- image view and matched-encoding configuration;
- resize library/version, interpolation, antialias, rounding, and storage configuration;
- relevant configuration values.

Fit normalization means, standard deviations, PCA, or other feature scaling on `seed_train` only. Persist these statistics with the checkpoint and reuse them unchanged on validation, test, and inference data.

Check missingness and validity rates by label. If optical or PRNU extraction fails much more often for one label because of dataset construction, fix the dataset or mask/drop the feature rather than letting missingness become a shortcut.

Before training the main model, fit a nuisance-only audit on format, dimensions, file size, estimated original quality, quantization-table statistics, normalization settings, and feature validity. Compare its label predictiveness before and after matching; investigate or remove any residual class-correlated pipeline setting.

## 7. Model Inputs and Fusion

The retained baseline receives only the global controlled-RINE representation computed from the runtime `received_view`. The candidates below were evaluated as experimental additions from the same view, but none is enabled in the packaged model:

- global frozen CLIP representation;
- aggregated texture-patch representation;
- family-aware frequency vector;
- reference-free PRNU-v2 vector and support masks;
- RGB/Lab correlation vector;
- chromatic-aberration and eligible radial-distortion vector;
- validity/confidence masks for every auxiliary family.

Each deterministic feature family should pass through its own small normalization/projection block before concatenation. Missing or invalid values are zero-filled only after normalization and paired with an explicit validity mask. The final fusion MLP produces one binary logit.

No auxiliary feature may directly override the logit. C2PA remains the only provenance early exit, and the frequency fast-track is disabled in the retained configuration.

## 8. Staged Training Procedure

The project trained in stages so every added component had a measurable reason to remain. The outcomes below are frozen for the baseline.

### Stage A — Frozen-CLIP baseline

1. Load the CLIP ViT-L/14 vision tower.
2. Freeze all backbone parameters.
3. Use the global representation only.
4. Train a linear or small MLP binary head with `BCEWithLogitsLoss`.
5. Select the checkpoint by `selection_score`, with R.Acc./F.Acc. regression checks.

Outcome: Stage A reached a 94.71% three-seed mean locked score and remained the mandatory comparator.

### Stage B — RINE-style multi-layer features

1. Read the selected intermediate CLS/token representations.
2. Keep the CLIP backbone frozen.
3. Train the importance estimator and binary head.
4. Compare against Stage A on the same manifest and seeds.

Outcome: controlled RINE reached 100.00% clean, 99.62% robustness, and 99.81% mean locked accuracy and was retained. Seed 42, the strongest retained seed at 99.85% locked, became the packaged checkpoint.

### Stage C — Deterministic auxiliary families

Starting from the better CLIP baseline, add one family at a time:

1. frequency;
2. PRNU;
3. RGB correlation;
4. Lab correlation;
5. chromatic aberration;
6. radial lens distortion, only when coverage is adequate;
7. combinations that survived their individual tests.

Within the frequency ablation, compare phase-spectrum and magnitude-spectrum features separately on clean and Q90/Q70/Q50/Q30 rows. Do not promote phase features into the shipped vector unless they add held-out value beyond matched-view CLIP.

For each addition, train only the feature projection and fusion head while CLIP stays frozen. Run the same seeds and record:

- clean and robustness accuracy;
- R.Acc. and F.Acc.;
- per-transform performance;
- feature validity/coverage;
- per-generator/decoder results for frequency;
- correlation with already retained auxiliary features;
- latency and memory impact.

Outcome: frequency, Lab, and PRNU-v2 fusion were rejected; RGB/phase variants were dropped earlier; optics never became eligible. No Stage C feature is enabled.

### Stage D — Texture-aware local-detail head

1. Generate the multi-scale Laplacian/Sobel energy map.
2. select a fixed top-k set of non-overlapping patches;
3. retain the global CLIP view;
4. encode selected patches with the shared frozen backbone or the approved small shared patch head;
5. train soft attention/aggregation and the fusion head.

The clean pilot used four non-overlapping 112 px source patches plus one global view and passed its clean gate. Its frozen-checkpoint JPEG/blur/resize continuation then scored 93.13% mean robustness versus 99.80% for controlled RINE, with severe `resize_scale_0.25` failure. Texture was rejected and is disabled.

### Stage E — Selective backbone fine-tuning

Fine-tuning remains optional and was not used for the frozen baseline.

Try candidates in this order:

1. train the fusion/head layers only;
2. add parameter-efficient adapters to the last CLIP blocks, if available;
3. unfreeze only the final one or two vision blocks;
4. consider broader fine-tuning only if the dataset and compute budget clearly support it.

Use a much smaller learning rate for pretrained parameters than for new heads. Maintain a frozen-backbone checkpoint for rollback. Reject fine-tuning that improves training accuracy while reducing held-out generator or transformed accuracy.

## 9. Starting Hyperparameters

These are reproducible starting points, not fixed claims of optimality:

| Setting | Starting value |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW |
| Head-only learning rate | `1e-3` |
| Fusion/texture learning rate | `3e-4` |
| Unfrozen CLIP learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Effective batch size | Target 32 via accumulation; choose the physical microbatch only after a hardware probe |
| Warmup | 5% of optimizer steps |
| Gradient clipping | 1.0 |
| Head-training limit | 20 epochs |
| Selective fine-tuning limit | 5–10 epochs |
| Early-stopping patience | 3 validation checks |
| Random seeds | At least 3 for retained candidates |

Use mixed precision when supported. Do not apply label smoothing initially because probability calibration is a required output. If class-balanced sampling is used, report ordinary challenge accuracy—not sampler-weighted accuracy.

These values configure optimization; they do not establish compute feasibility. Frozen-backbone head training should use cached embeddings where the image views are fixed. Global plus `k` local CLIP views require approximately `1 + k` image encodes per sample even though they do not add another backbone, so texture experiments begin only after measuring this cost on a representative pilot. Any on-demand stochastic augmentation invalidates the corresponding embedding cache and must be included in the throughput estimate.

## 10. Validation, Checkpointing, and Retention Gates

Evaluate at a fixed interval using `selection_val`:

1. clean binary accuracy;
2. mean accuracy across independent transform cells;
3. the 50/50 `selection_score`;
4. R.Acc. and F.Acc. for clean and each transform;
5. per-generator/decoder results;
6. feature validity/coverage;
7. ECE for monitoring; final temperature fitting is attempted only after checkpoint selection;
8. latency and peak memory.

Save the artifacts supported by each trainer. Controlled RINE and auxiliary fusion preserve `latest.pt` and `best_50_50.pt`; Stage A additionally records best-clean and best-robustness checkpoints:

- latest checkpoint;
- best `selection_score` checkpoint;
- best robustness checkpoint;
- best clean-accuracy checkpoint;
- full configuration and manifest hash with each checkpoint.

An auxiliary feature or fine-tuning change is retained only when it:

- improves the pre-agreed primary selection criterion;
- does not exceed the agreed R.Acc./F.Acc. regression tolerance;
- generalizes beyond one generator/checkpoint;
- has acceptable validity coverage and inference cost;
- remains useful after accounting for correlation with existing signals.

Define numeric regression tolerances before inspecting candidate results.

## 11. Calibration and Decision Threshold

The planned procedure after selecting the final architecture and checkpoint was:

1. freeze all weights;
2. fit one temperature parameter on the **clean `selection_val` logits only**;
3. reuse that temperature unchanged for transformed validation, final test, and inference;
4. report clean and per-transform ECE;
5. use one fixed binary threshold—default `0.5` unless the challenge specifies otherwise.

The fit on 165 perfectly classified clean seed-42 rows converged to the lower search bound (`T=0.0500038`). Because this only sharpened already error-free logits and did not provide a meaningful interior calibration solution, it was rejected. Final evaluation and inference use unchanged raw logits with `T=1` and threshold `0.5`. No per-transform temperature or threshold is allowed.

## 12. Stage 1 Frequency Fast-Track Enablement

Train/evaluate the frequency-only scorer as an auxiliary baseline, but leave the early exit off.

Before it can be enabled, define a minimum precision and minimum coverage target, then verify them on:

- locked authentic images;
- held-out GAN/CNN decoders;
- held-out VAE/VQ/latent decoders;
- pixel-space or decoder-free generators when available;
- autoregressive/token-based generators when available;
- `unknown` generator metadata;
- every independent transform cell where the fast-track might run.

Use confidence intervals, not only point estimates. If any required stratum lacks sufficient evidence, keep the fast-track disabled. It may only exit toward `ai_generated`; no frequency result can early-exit toward `authentic`.

## 13. Binary Self-Training and Fine-Tuning Loop

Self-training was not used for the frozen baseline. If attempted later, it starts only after the supervised pipeline and locked evaluation protocol are stable and remains a separately versioned experiment.

For each iteration:

1. run the current frozen model on the `self_train_pool`;
2. apply separate conservative confidence thresholds for predicted `authentic` and `ai_generated`;
3. reject low-confidence samples and any sample with uncertain provenance;
4. audit hidden-label pseudo-label accuracy without exposing labels to training selection;
5. add accepted pseudo-labels with a loss weight of `0.5–0.7x` relative to verified labels;
6. generate training variants directly from their clean parent, one transform at a time;
7. fine-tune from the previous accepted checkpoint;
8. evaluate on `selection_val` only;
9. accept the iteration only if the agreed score and per-label gates pass;
10. otherwise restore the previous checkpoint and log the rollback reason.

The `final_test` set never participates in this loop. Stop when pseudo-label yield becomes too small, selection performance plateaus, or the time/compute budget is exhausted.

Human-reviewed examples may become verified training data only when their binary provenance is clear. Review must not force mixed or AI-edited content into either label.

## 14. Final Evaluation

The final architecture, weights, `T=1` decision, threshold, and feature flags were frozen before evaluation. `final_test` was read exactly once through `notebooks/11_final_test.ipynb` on 2026-08-31. It contained 141 direct fixed-Q96 matched-clean rows; it did not contain or authorize a second bank of transformed final-test variants.

The final report therefore contains overall and per-class accuracy, false-positive and false-negative rates, confusion matrix, and ECE for those 141 rows. The 14-cell robustness mean and locked 50/50 score remain development `selection_val` evidence from Notebook 07 and must not be presented as measurements on the sealed final-test population. Target-hardware latency, peak memory, and disk/cache profiling remain outstanding.

Recorded result: 141 samples, 99.29% overall accuracy (140/141), 100.00% AI-generated accuracy (69/69), 98.61% authentic accuracy (71/72), and ECE 0.0189. The checkpoint is controlled RINE seed 42 with threshold `0.5` and `T=1`. Never select a different checkpoint or rerun this `final_test` after viewing the result; later experiments require a new protected evaluation protocol.

## 15. Reproducibility Artifacts

Every run should save:

- resolved training configuration;
- Git commit hash;
- manifest hash and split version;
- random seed;
- package and model versions;
- feature-extractor versions;
- training/validation curves;
- per-transform and per-family metrics;
- checkpoint and calibration temperature;
- threshold and fast-track state;
- ablation/retention decision;
- failure or rollback reason.

Recommended artifact layout:

```text
artifacts/
  manifests/
  feature_cache/
  runs/<run_id>/
    config.yaml
    metrics.json
    curves.csv
    checkpoint.pt
    calibration.json
    ablations.csv
    environment.txt
```

Generated artifacts, downloaded datasets, and checkpoints should not be committed unless the repository explicitly defines a small, licensed fixture policy.

## 16. Completion record

- [x] Binary provenance, corruption, duplicate, and source-group leakage checks completed.
- [x] Split assignments and manifest hashes were frozen before model selection.
- [x] Matched encoding was label-independent and all 14 robustness cells derived directly from clean parents without chaining.
- [x] Resize settings and lossless output behavior were pinned and verified.
- [x] Nuisance predictiveness was audited before and after matching.
- [x] Controlled clean/transformed and label sampling was verified.
- [x] Frozen CLIP, clean-trained RINE, and controlled RINE were evaluated across seeds 42/43/44.
- [x] Frequency, color, PRNU-v2, and texture candidates received individual evidence-gated decisions; none was retained.
- [x] Texture global-only, local-only, and global-plus-local resize behavior and per-label errors were reported.
- [x] Physical-feature eligibility and validity were reported; optical candidates failed closed before binary fitting.
- [x] R.Acc./F.Acc. gates were applied to retained-model decisions.
- [x] The frequency fast-track remained disabled.
- [x] Temperature fitting was attempted once on clean `selection_val` and rejected; `T=1` was frozen.
- [x] The sealed `final_test` was evaluated exactly once after the pipeline was frozen.
- [x] The selected checkpoint, calibration decision, final report, and inference contract are reproducible from recorded artifacts.
- [ ] Run the implemented resource profiler on the target hardware and record model size, latency, peak memory, and disk/cache requirements.

SAFE, self-training, selective fine-tuning, ConvNeXt-Tiny, and external datasets are optional new experiments, not incomplete baseline requirements.
