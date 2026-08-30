# Training and Fine-Tuning Plan

This document is the operational runbook for training, fine-tuning, calibrating, and selecting the binary `authentic` versus `ai_generated` detector. It implements the product boundary in `PRD.md`, the architecture in `design.md`, and the components in `models.md` and `techStack.md`.

## 1. Training Objectives and Non-Negotiable Rules

The model produces one logit and one calibrated public value: `pred = P(ai_generated)`.

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
| Frozen CLIP baseline | Linear/MLP binary head | Binary authentic/AI labels | **Ready once data and pretrained weights are acquired** |
| RINE-style layer fusion | Layer-importance estimator and binary head | Binary authentic/AI labels | **Trainable** with CLIP frozen |
| Texture path | Patch aggregation/attention and fusion weights | Binary labels on clean and independently transformed views | **Trainable**, but patch selection itself is deterministic and patch count/size are validation hyperparameters |
| Frequency path | Feature projection, fusion weights, and an optional frequency-only classifier | Binary labels plus generator/decoder metadata for stratified validation | **Trainable as an auxiliary path**; FFT/DCT and residual statistics themselves have no learned weights |
| Reference-free PRNU-v2 path | Feature projection and fusion weights | Binary labels on clean and independent-transform rows | **Implemented as a locked classification ablation**; known-device fingerprints/PCE remain separate validation evidence and never become classifier inputs |
| RGB/Lab correlation path | Feature projection and fusion weights | Binary labels on matched jitter distributions | **Trainable as fusion**; channel-correlation extraction remains deterministic |
| Chromatic-aberration/radial-distortion path | Feature projection and fusion weights | Binary labels and extractor confidence, but no lens/camera calibration ground truth | **Trainable as fusion only**; the optical fit must remain deterministic or be validated on a separate calibrated dataset |
| Stage 1 early exit | Frequency-only scorer and a fixed threshold selected on validation | Binary labels, held-out generator strata, and independent transforms | **Possible but disabled by default** until the precision/coverage gate passes |
| Temperature scaling | One scalar temperature | Clean `selection_val` logits and labels | **Trainable after checkpoint selection** |
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
| GenImage | Binary training/evaluation and generator holdouts | Preserve generator subsets and source grouping; do not treat multiple derivatives as independent sources |
| WildFake | Generator/architecture/version stratification | Acquire and verify the exact released hierarchy/licenses, then retain only fully synthetic and authentic branches |
| CIFAKE | Low-resolution stress test or quick backbone smoke test | Its 32x32 sources are ineligible for PRNU, chromatic aberration, radial distortion, CFA periodicity, and primary texture training; upscaling does not recreate those signals |
| COCO plus DALL-E benchmark subsets | External clean/robustness comparison | Validation-only as already declared; never use for head fitting, calibration, or feature selection |
| Chameleon | Final stress-test comparison | Do not use for training or retention decisions |

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

The completed local Task 8B pilot meets that matched-view gate at 0.50 balanced
accuracy using 256 px crop-only, uncompressed RGB TIFF views. This does not by
itself authorize fusion training. The separate seed-train-only device validation
reaches AUC 0.538 for multi-image fingerprints and 0.543 for the single-image
proxy, both below the frozen 0.60 minimum. CA is also ineligible because calibrated
lens/focal/edge-rich coverage is absent. Consequently no PRNU/CA projection or
fusion weights are trained, RINE remains unchanged, and no physical feature is
retained.

### 1.3 Current readiness status

This repository currently contains architecture documents only: no downloaded dataset, frozen manifest, pretrained checkpoint, feature cache, training environment, or measured hardware budget is present. The design is therefore **architecturally trainable but not yet execution-ready**.

The minimum evidence required before claiming training readiness is:

1. acquire a licensed binary subset and confirm class counts after tampered/mixed/ambiguous rows are removed;
2. audit native dimensions, formats, source groups, generator metadata, and exact/near duplicates;
3. materialize the label-independent matched-clean views and independent transform rows;
4. measure feature eligibility/coverage by label, dataset, and transform before fitting fusion weights;
5. run a small CLIP embedding-extraction pilot to measure throughput, peak memory, and cache size on the actual machine;
6. freeze the manifest/splits and only then train Stage A.

The minimum viable training path is Stage A, followed by Stage B and one-at-a-time auxiliary fusion. PRNU, optical features, texture crops, the frequency fast-track, self-training, and CLIP fine-tuning are retention-gated experiments—not dependencies for producing a valid detector.

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

The mandatory Stage 2 baseline receives global and patch CLIP representations from the same training sample view that will appear as `received_view` at runtime. Experimental fusion additionally receives the following auxiliary features computed from that same view:

- global frozen CLIP representation;
- aggregated texture-patch representation;
- family-aware frequency vector;
- reference-free PRNU-v2 vector and support masks;
- RGB/Lab correlation vector;
- chromatic-aberration and eligible radial-distortion vector;
- validity/confidence masks for every auxiliary family.

Each deterministic feature family should pass through its own small normalization/projection block before concatenation. Missing or invalid values are zero-filled only after normalization and paired with an explicit validity mask. The final fusion MLP produces one binary logit.

No auxiliary feature may directly override the logit. C2PA remains the only provenance early exit, and the frequency fast-track remains disabled during initial training.

## 8. Staged Training Procedure

Train in stages so every added component has a measurable reason to remain.

### Stage A — Frozen-CLIP baseline

1. Load the CLIP ViT-L/14 vision tower.
2. Freeze all backbone parameters.
3. Use the global representation only.
4. Train a linear or small MLP binary head with `BCEWithLogitsLoss`.
5. Select the checkpoint by `selection_score`, with R.Acc./F.Acc. regression checks.

This is the minimum baseline every later stage must beat.

### Stage B — RINE-style multi-layer features

1. Read the selected intermediate CLS/token representations.
2. Keep the CLIP backbone frozen.
3. Train the importance estimator and binary head.
4. Compare against Stage A on the same manifest and seeds.

Retain the multi-layer extractor only if it adds locked selection value without unacceptable class-specific regression.

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

Remove features that are redundant, unstable, or useful only on one known generator.

### Stage D — Texture-aware local-detail head

1. Generate the multi-scale Laplacian/Sobel energy map.
2. select a fixed top-k set of non-overlapping patches;
3. retain the global CLIP view;
4. encode selected patches with the shared frozen backbone or the approved small shared patch head;
5. train soft attention/aggregation and the fusion head.

Tune patch size, patch count, energy-map scale, and aggregation method only on `selection_val`. Keep a fixed inference budget. Texture energy chooses where to look; it is never an authenticity threshold.

### Stage E — Selective backbone fine-tuning

Fine-tuning is optional and begins only after the frozen pipeline is stable.

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
7. ECE for monitoring only until final calibration;
8. latency and peak memory.

Save:

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

After selecting the final architecture and checkpoint:

1. freeze all weights;
2. fit one temperature parameter on the **clean `selection_val` logits only**;
3. reuse that temperature unchanged for transformed validation, final test, and inference;
4. report clean and per-transform ECE;
5. use one fixed binary threshold—default `0.5` unless the challenge specifies otherwise.

Do not fit a separate temperature or threshold per transform. That would use knowledge unavailable at inference and hide distribution shift.

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

Self-training starts only after the supervised pipeline and locked evaluation protocol are stable.

For each iteration:

1. run the current calibrated model on the `self_train_pool`;
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

Once architecture, weights, calibration, and threshold are frozen:

1. run `final_test` exactly once for the final internal report;
2. evaluate the clean set;
3. evaluate every transform-and-parameter set independently from its clean parent;
4. compute:

`final_score = 0.50 × clean_accuracy + 0.50 × mean_independent_transform_accuracy`

Report:

- clean accuracy, R.Acc., F.Acc., confusion matrix, and ECE;
- one row for every independent transform parameter;
- robustness mean and final 50/50 score;
- generator/decoder-family breakdown;
- auxiliary ablations;
- feature validity/coverage;
- false positives and false negatives;
- inference latency, peak memory, and model size;
- whether the Stage 1 fast-track remained disabled or passed its gate.

Never select a different checkpoint after viewing final-test results. Any later change starts a new documented experiment with a newly protected final evaluation protocol.

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

## 16. Training Completion Checklist

- [ ] Binary provenance screening is complete.
- [ ] Duplicate and source-group leakage checks pass.
- [ ] Split and manifest hashes are frozen.
- [ ] Original bytes and metadata are immutable and C2PA is checked before derivation.
- [ ] Matched-encoding quality, codec, encoder, and subsampling distributions are label-independent.
- [ ] Every transformed sample has one canonical matched clean parent and one transform cell.
- [ ] Resize-0.5x and resize-0.25x use pinned bilinear, antialias, rounding, color, dtype, and library settings.
- [ ] Every resize output matches its parent dimensions and is cached losslessly or kept as an array.
- [ ] Compression-nuisance predictiveness is audited before and after matching.
- [ ] Clean/transformed and label sampling are balanced as designed.
- [ ] Original/unmatched, matched-view, and same-received-view fusion ablations are reproducible.
- [ ] Global-only, local-only, and global-plus-local resize ablations are reproducible.
- [ ] Authentic false-positive and synthetic false-negative changes are reported for both resize rows.
- [ ] Frozen-CLIP baseline is reproducible.
- [ ] Every auxiliary family has an individual ablation.
- [ ] Generator/decoder-family holdouts are reported.
- [ ] Texture and forensic feature validity/coverage are reported.
- [ ] Selective fine-tuning beats or justifiably replaces the frozen baseline.
- [ ] Temperature is fitted once on clean selection validation.
- [ ] R.Acc./F.Acc. regression gates pass.
- [ ] Stage 1 fast-track is either explicitly disabled or has passed its enablement gate.
- [ ] Stage 1 fast-track is disabled unless its precision gate passes on resized authentic images.
- [ ] Final-test evaluation occurs only after the pipeline is frozen.
- [ ] The final 50/50 score and error analysis are reproducible from saved artifacts.
