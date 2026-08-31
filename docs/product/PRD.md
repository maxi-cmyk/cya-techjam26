# PRD — Robust Detection of AI-Generated Images Under Real-World Transformations

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

## 7. Open Product Questions
- Whether to expose `verdict_source` (c2pa / frequency / model) as a visible Table 1 breakdown for judges, even though it's not part of the required JSON.
- How to demonstrate the C2PA early-exit mechanism given that none of the reference training datasets are known to carry intact C2PA manifests (requires a small self-synthesized demo set).
- How much of the self-training machinery to build live versus describe as a designed-but-time-boxed mechanism, given hackathon time constraints.

Resolved: PRNU-v2 validates a device-specific signal independently, but its
binary fusion is rejected for the current handoff after severe cross-seed
instability.

## 8. Known Gaps / Risks Called Out for Follow-up
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
