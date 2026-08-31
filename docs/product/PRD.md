# PRD — Robust Detection of AI-Generated Images Under Real-World Transformations

**Document purpose:** Forward-looking product and experimentation plan for the
submission. This PRD defines what the team intends to build, evaluate, and
present. Experimental features remain candidates until measured evidence
supports retaining them.

## 1. Background & Problem Statement

Generative AI makes it trivial to produce highly realistic synthetic images at scale, creating risk for online platforms: misinformation, impersonation, fraud, and eroded trust in digital content. Detection is hardest in exactly the conditions that matter most — after images are compressed, cropped, reposted, or lightly edited. Lab-only accuracy on clean images is not a useful proxy for real-world performance.

**Goal:** build a prototype that distinguishes AI-generated images from authentic ones with strong robustness under realistic post-processing and redistribution, not just on clean data.

## 2. Scope

**In scope:**
- Image-level AIGC (AI-generated content) detection
- Binary classification of fully AI-generated images versus authentic images
- Robustness to common image transformations (JPEG compression, blur, resize, noise, color jitter, cropping)
- Experimental reference-free PRNU-v2 residual features as auxiliary evidence and a documented robustness ablation
- Texture-aware local-detail features trained and evaluated under the same independent-transform protocol
- Deterministic inter-channel color and optical-aberration features fused into Stage 2 with confidence/coverage indicators
- Generator-family-stratified spectral analysis covering decoder/upsampling fingerprints rather than assuming one universal frequency signature
- Feature engineering, model design, evaluation design, error analysis, explainability

**Out of scope:**
- Full production deployment / platform-wide moderation systems
- Non-image modalities (video, audio)
- Mixed-origin images, composited images, AI-edited authentic images, face swaps, and partial-AI content
- Chained or overlaid transformations in the robustness benchmark
- Adversarial-perturbation robustness and pixel-mask segmentation

**Constraints:**
- Hackathon-scale prototype, limited compute, no access to internal production systems
- Models must be **<2B parameters**
- Must run **locally** — no dependency on external APIs in the inference path
- Public or properly licensed datasets only; assumptions about deployment context must be stated explicitly

## 3. Users & Use Case

A platform trust & safety team (or similar) needs a tool that ingests a directory of images and outputs, per image, a calibrated probability that it is AI-generated — one that stays meaningful even after the image has been re-encoded, resized, or lightly edited somewhere along its distribution path.

## 4. Product Requirements

### 4.1 Core Classification
- Primary output: **`pred`** — a calibrated `P(ai_generated)` score, strictly for fully-synthetic images (no real source image at all).
- Ground-truth labels are limited to **`authentic`** and **`ai_generated`**.
- `authentic` means a genuinely captured source image. `ai_generated` means a fully synthetic image with no authentic source image.
- Mixed, composited, face-swapped, AI-enhanced, or otherwise AI-edited images are excluded rather than mapped into either class.

### 4.2 Robustness Requirement
Must maintain reasonable accuracy after realistic post-processing:

| Transform | Parameters | Real-World Analog |
|---|---|---|
| JPEG Compression | quality 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian Blur | σ = 0.5, 1.0, 2.0 | Out-of-focus |
| Resize round trip | 0.5× or 0.25× bilinear downsample, then bilinear restore to original dimensions | Thumbnail generation, CDN resize |
| Gaussian Noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color Jitter | brightness/contrast/sat ±20% | Filter apps, auto-enhance |
| Center Crop | crop 80% | Profile-picture cropping |

Each transformed test sample is created directly from its clean source with **exactly one transform and one parameter setting**. Transformations must not be chained, mixed, composited, or overlaid. For example, the JPEG-50 set contains JPEG-50 only, not JPEG-50 followed by resize or blur.

The brief does not specify resize interpolation, so the project declares bilinear interpolation for both steps. The 0.5× and 0.25× conditions are separate rows; each complete down-and-up round trip counts as one benchmark transform, and the intermediate image is not evaluated separately. Resize outputs retain the parent dimensions and are cached losslessly or kept as arrays so the resize row does not add JPEG compression.

For primary-model experiments, `clean` means a canonical matched derivative created with a label-independent JPEG encoder and quality policy. The untouched downloaded or generated file remains an immutable `original` for C2PA and explicitly separated native-forensics ablations. Dataset normalization is baseline construction; each robustness row still applies exactly one benchmark transform to its matched clean parent.

The product must not silently fail by defaulting to "real" when a transform destroys its detection signal — this failure mode must be measured and reported, not hidden.

### 4.3 Deliverable JSON Contract
```
{"image_path": "img001.jpg", "pred": 0.12}
```
`image_path` and `pred` are the public contract. Internal-only diagnostic fields (e.g. `verdict_source` and degradation-bias metrics) must **not** appear in the public JSON.

### 4.4 Evaluation & Reporting Requirements
- **Robustness Evaluation Summary** — clean vs. transformed performance, split by real-accuracy (R.Acc.) and fake-accuracy (F.Acc.) per transform, so any "predicts real under degradation" bias is directly visible rather than hidden in an aggregate number.
- **Independent transform rows** — every transform and parameter setting is evaluated from the clean source; no row may contain multiple transformations.
- **Generator generalization table** — held-out generator family, evaluated on a JPEG/resolution-matched dataset so the eval measures genuine generalization, not compression/resolution bias.
- **Binary confusion matrix** for `authentic` versus `ai_generated`.
- **Calibration table** — Expected Calibration Error (ECE) for both binary labels, on clean and independently transformed validation data.
- **PRNU ablation** — compare CLIP-only, PRNU-only, and fused predictions on clean data and every independent transformation row.
- **Texture ablation** — compare global CLIP, texture-only, CLIP + texture, CLIP + PRNU, and full fusion on clean data and every independent transformation row.
- **Color/optics ablation** — compare RGB versus Lab correlation features, chromatic-aberration and radial-distortion coverage, feature-only discrimination, and incremental value in the full fused model for every transform row.
- **Frequency-family matrix** — report frequency-only performance and fast-track coverage by known generator/decoder family and independent transform. Include a held-out family test and an `unknown` metadata bucket.
- **Compression-bias audit** — measure how well format, dimensions, file size, estimated JPEG quality, quantization-table statistics, and feature validity predict the label before and after matched preprocessing.
- **Matched-preprocessing ablation** — compare otherwise identical models trained on original/unmatched views and canonical matched derivatives, using the same split and seed.
- **JPEG degradation table** — report clean and Q90/Q70/Q50/Q30 R.Acc., F.Acc., aggregate accuracy, ECE, and clean-to-degraded accuracy loss.
- **Signal-family comparison** — report representation-level and low-level frequency-feature degradation separately under JPEG.
- **Resize degradation table** — report clean, resize-0.5×, and resize-0.25× R.Acc., F.Acc., aggregate accuracy, ECE, authentic false-positive rate, synthetic false-negative rate, and clean-to-resized loss.
- **Resize architecture ablation** — compare global-only CLIP, local-crop-only representation, and global-plus-local CLIP on both resize severities.
- **Resize fast-track audit** — report Stage 1 precision and coverage on resized authentic images before enabling any frequency-based early exit.
- **Error Analysis Note** — representative false positives/negatives and stated trade-offs.

### 4.5 Scoring

The final evaluation score is split evenly:

| Component | Weight | Definition |
|---|---:|---|
| Accuracy | 50% | Binary accuracy on the clean test set |
| Robustness | 50% | Mean binary accuracy across all independent transform-and-parameter test sets |

`Final score = 0.50 × clean accuracy + 0.50 × mean independent-transform accuracy`.

R.Acc. and F.Acc. must still be reported beside each aggregate so class-specific collapse cannot be hidden by the final score.

### 4.6 Non-Functional Requirements
- Local, memory- and space-efficient: minimize simultaneously-loaded models at inference time.
- Deliberate, explainable architectural decisions over maximal complexity.
- Iterative improvement mechanism (self-training loop) demonstrated over multiple cycles with measurable results, not just a one-shot model.

## 5. Deliverables (per challenge brief)
1. **Written project description** (via Devpost) — approach, tools, models/APIs, libraries, datasets used.
2. **Public code repository** — well-structured, commented code; an inference script that takes an image directory and outputs a JSON file with `image_path` and `pred` per image; a README with setup, reproduction steps, limitations/future work, and contributions.
3. **Demo video** — end-to-end demo, public on YouTube, linked in Devpost, no third-party trademarked/copyrighted content.
4. **Robustness Evaluation Summary** (table/visual, clean vs. transformed).
5. **Error Analysis Note** (false positives/negatives, trade-offs).

## 6. Evaluation Weights

Model evaluation is weighted **50% accuracy and 50% robustness**, using the formula in Section 4.5. Other qualitative judging considerations may shape the presentation, but they do not replace this model-scoring split.

## 7. Planned Execution

The project will proceed in evidence-gated stages. Each stage produces a
reproducible artifact and a decision before the next feature family is added.
The sealed final test remains unavailable for model selection.

### Phase 1 — Reproducible project foundation

- Freeze configuration, dependency versions, random seeds, data paths, and
  artifact schemas.
- Establish a Colab-compatible workflow that can resume from persisted
  checkpoints without changing the experiment definition.
- Add smoke tests for configuration, deterministic behavior, and output
  contracts before expensive training begins.

**Exit gate:** the same configuration can be restored and produces the same
split membership and transform definitions.

### Phase 2 — Data contract and leakage controls

- Build a source manifest containing provenance, label, generator or camera
  family where known, file format, dimensions, and eligibility decisions.
- Exclude mixed-origin, AI-edited, composited, ambiguous, or improperly
  licensed samples.
- Split by source identity and generator family before creating derivatives so
  related images cannot cross train, validation, and test boundaries.
- Create label-independent matched clean derivatives while preserving immutable
  originals for provenance and native-forensics experiments.
- Audit compression, resolution, format, and other nuisance variables for
  accidental label shortcuts.

**Exit gate:** the eligible binary dataset is provenance-audited, leakage-free,
and has a documented matched-preprocessing policy.

### Phase 3 — Independent robustness benchmark

- Materialize or deterministically generate every transform cell directly from
  its clean parent.
- Keep JPEG, blur, resize, noise, color jitter, and crop conditions independent;
  do not chain transforms.
- Verify parent identifiers, transform parameters, image counts, and
  reproducibility hashes.

**Exit gate:** every benchmark row can be traced to one clean parent and exactly
one declared transform setting.

### Phase 4 — Representation baselines

- Train a frozen-CLIP baseline to establish a simple global semantic reference.
- Train RINE using selected intermediate CLIP layers to test whether generation
  traces are better represented below the final semantic layer.
- Evaluate at least three fixed seeds using clean accuracy, mean robustness,
  class-specific accuracy, calibration, and the locked 50/50 score.

**Exit gate:** select one reproducible parent model using validation evidence
only. Record variance and class-specific regressions rather than choosing the
best isolated run.

### Phase 5 — Candidate forensic features

Candidate signals will be added one family at a time so their incremental value
can be measured.

1. **Frequency:** spectral magnitude, residual, phase, periodicity, and
   generator-family behavior.
2. **Color:** RGB and Lab inter-channel relationships with explicit extractor
   confidence and validity.
3. **Native physical signals:** PRNU, chromatic aberration, and radial distortion
   only where the source data is eligible for physical-signal analysis.
4. **Texture:** global and local-patch detail intended to complement the global
   CLIP representation.

For each family, compare feature-only and fused variants against the frozen
parent on identical rows and seeds. A candidate is retained only if it improves
the mean locked score without breaching class-regression guardrails or showing
unacceptable cross-seed instability. Missing or low-confidence physical
features remain neutral and never become rule-based authenticity evidence.

**Exit gate:** every candidate has an explicit `retain`, `diagnostic-only`, or
`reject` decision backed by clean, per-transform, per-class, and cross-seed
evidence.

### Phase 6 — Robustness-aware retraining

- Retrain the selected parent with a controlled sampler that presents either a
  clean image or one independently transformed version of that image.
- Preserve class balance and one-transform-per-parent semantics.
- Compare controlled retraining with the clean-trained baseline before adding
  any auxiliary fusion.
- Stop or revise training if robustness gains are produced by unacceptable
  authentic or AI-generated class collapse.

**Exit gate:** freeze the model configuration, checkpoint-selection rule, and
decision threshold before final testing.

### Phase 7 — Calibration, packaging, and sealed evaluation

- Fit calibration on validation logits only and retain it only if held-out ECE
  improves without changing the classification decision contract.
- Package a root inference script that recursively scores an image directory
  and emits only `image_path` and `pred` in the public JSON.
- Add deterministic discovery, invalid-image reporting, atomic output writes,
  and explicit exit codes.
- Optionally use verified C2PA AI-generation claims as an AI-only fast path;
  missing metadata must always fall through to the model.
- Run the sealed final test exactly once after all selection decisions and
  packaging behavior are frozen.

**Exit gate:** the judge-facing path runs from a fresh checkout, the public JSON
matches the required schema, and the final report can be regenerated from saved
predictions without reopening model selection.

## 8. Findings and Submission Presentation Plan

The presentation will show the experimental question, evidence, and decision
for each stage. It will not present every engineered feature as part of the
final model, and it will not hide negative results.

### 8.1 Core narrative

1. **Problem:** clean-image detection is insufficient after real-world
   redistribution.
2. **Evaluation contract:** fully authentic versus fully AI-generated only;
   independent transforms; equal weighting of clean accuracy and robustness.
3. **Approach:** establish a frozen-CLIP baseline, test intermediate-layer RINE,
   then admit forensic feature families only through measured retention gates.
4. **Evidence:** compare models on the same rows, seeds, class breakdowns, and
   transform cells.
5. **Product:** demonstrate directory inference and the exact JSON output
   contract.
6. **Limitations:** state dataset, calibration, generator-generalization,
   provenance, and physical-signal boundaries explicitly.

### 8.2 Required findings tables and visuals

- A headline table with clean accuracy, mean robustness accuracy, and the
  locked 50/50 score for every serious model candidate.
- A robustness matrix with one row per independent transform setting and
  columns for authentic accuracy, AI-generated accuracy, aggregate accuracy,
  and ECE.
- A clean confusion matrix and class-specific false-positive/false-negative
  rates.
- A held-out generator-family generalization table using matched preprocessing.
- Feature-family ablations showing the parent, feature-only candidate, and
  fused candidate under the same evaluation contract.
- A compression-bias audit before and after matched preprocessing.
- Calibration reliability plots or tables for clean and transformed validation
  data.
- An error-analysis panel with representative false positives and false
  negatives, accompanied by likely failure modes rather than unsupported causal
  claims.
- A compact decision log stating which candidates were retained, kept only for
  diagnostics, or rejected and why.

### 8.3 Demo flow

1. Show the binary scope and unsupported mixed/edited cases.
2. Score a directory containing clean, transformed, and invalid images.
3. Open `predictions.json` to demonstrate the exact public contract.
4. Open `report.json` to show that invalid inputs are surfaced rather than
   defaulted to authentic.
5. Present the robustness matrix and one representative ablation decision.
6. Close with the sealed-test summary, known limitations, and highest-value
   next experiments.

### 8.4 Reporting principles

- Separate validation/model-selection findings from sealed final-test results.
- Report means and seed variability, not only the best seed.
- Show authentic and AI-generated accuracy beside aggregate metrics.
- Label implemented behavior, planned behavior, and future work distinctly.
- Treat rejected features as useful findings when they reveal instability,
  redundancy, or transformation fragility.
- Do not claim universal AI detection, production readiness, calibrated
  confidence, or coverage of mixed/AI-edited images without supporting evidence.

## 9. Open Product Questions
- Whether to expose `verdict_source` (c2pa / frequency / model) as a visible Table 1 breakdown for judges, even though it's not part of the required JSON.
- How to demonstrate the C2PA early-exit mechanism given that none of the reference training datasets are known to carry intact C2PA manifests (requires a small self-synthesized demo set).
- How much of the self-training machinery to build live versus describe as a designed-but-time-boxed mechanism, given hackathon time constraints.

## 10. Known Gaps / Risks Called Out for Follow-up
- **Error analysis mechanics**: need a concrete script to pull top-10 false positives/negatives; decide whether this is reflected in the UI/demo and how human verification confirms "true" detection rate.
- **Error handling**: no defined baseline yet for minimum image resolution or corruption detection (garbage-in handling).
- **Dataset provenance**: each sample must be screened so mixed, AI-edited, or ambiguous-origin images do not enter either binary class.
- **Compression-history shortcut**: authentic and generated sources may have systematically different save histories. Primary-model experiments must use label-independent matched derivatives, while immutable originals are retained for provenance and native-forensics ablations.
- **PRNU limitation**: without multiple images from a known camera, the system cannot estimate or verify a device fingerprint. Its single-image coherence score is experimental evidence, not proof that an image is authentic.
- **Transformation fragility**: JPEG, blur, resize, and added noise can severely weaken PRNU. Missing PRNU must never independently produce an `ai_generated` verdict, and PRNU must never act as an early-exit gate.
- **Interpolation-artifact confusion**: resize restoration can introduce periodic or grid-aligned structure that frequency-sensitive features may mistake for generation evidence. Resampling evidence is auxiliary, and authentic false-positive changes must be reported for both resize rows.
- **DSNU**: considered but deprioritized because reliable isolation normally needs dark-frame/reference calibration, while correction pipelines can suppress the signal. An exported single image provides neither the required reference nor a dependable residual.
- **Texture shortcut risk**: absolute smoothness, edge density, OCR confidence, or any fixed threshold can confuse degraded authentic images with synthetic images. Texture evidence must be learned under independently transformed training copies and can never override the binary model by rule.
- **Signal redundancy**: texture, frequency, and PRNU paths may capture overlapping low-level artifacts. Ablations must show that each retained path adds value to the locked 50/50 score.
- **Color-space limitation**: per-channel normalization can reduce sensitivity to affine brightness/contrast changes, but Lab conversion does not make the cue invariant to saturation or general color jitter. RGB and Lab must both be tested.
- **Optics limitation**: chromatic aberration and radial lens distortion may be weak, corrected by the camera pipeline, simulated by a generator, or unobservable in scenes without enough edge/line support. Missing or low-confidence estimates are neutral, never evidence of either class.
- **Color-jitter alignment**: authentic and AI-generated training samples must use the same jitter parameter distribution and sampling policy so jitter itself cannot become a label shortcut.
- **Spectral-family limitation**: checkerboards, spectral replicas, periodic autocorrelation, and high-frequency decay are architecture/decoder dependent and can be mitigated or erased. A frequency detector tuned to one generator family must not be presented as universal.
- **Autoregressive uncertainty**: token-based generators may inherit artifacts from their tokenizer or VAE/VQ decoder, but no single natural autoregressive spectral signature is assumed. They require their own held-out evaluation.
