# Models — Binary Detection Plan

This project detects only two source classes: **fully authentic images** and **fully AI-generated images**. It does not attempt to detect AI editing, face swaps, mixed-origin images, compositing, or partial-AI content.

## Primary Model

The shipped detector is a frozen CLIP ViT-L/14 vision backbone with RINE-style multi-layer feature extraction, a texture-aware local-detail head, deterministic PRNU/color/optics feature vectors, and a lightweight binary classification head.

- Input: one image.
- Output: calibrated `P(ai_generated)`.
- Labels: `authentic` and `ai_generated` only.
- Parameter budget: approximately 304M for the vision backbone, below the 2B limit.
- Inference: one live backbone; no ensemble and no face- or edit-specific specialists.
- Local input: a fixed budget of texture-rich patches selected by multi-scale edge/energy maps.
- Auxiliary input: single-image PRNU-coherence features derived from the noise residual.
- Auxiliary input: RGB/Lab inter-channel correlations plus confidence-aware chromatic-aberration and optional radial-distortion estimates.

Patch-level probabilities may be combined using soft averaging or attention pooling. Temperature scaling is fit on the clean validation set and then reused unchanged for every robustness set so distribution-shift effects remain measurable.

## Family-Stratified Frequency Feature Bank

Stage 1 extracts a compact deterministic vector containing:

- 2-D log FFT/DCT magnitude summaries
- radial and angular power distributions
- periodic peak locations/prominence
- residual autocorrelation peak statistics
- local neighboring-pixel dependency statistics
- input-quality and feature-validity indicators

These features target decoder and upsampling behavior rather than treating all synthetic images as one spectral family. Convolutional decoders may show checkerboards, replicated spectra, or anomalous high-frequency decay regardless of whether the upstream generator is a GAN, diffusion model, transformer, or autoregressive model. Latent and token systems may inherit periodic traces from a shared VAE/VQ decoder. Pixel-space or otherwise decoder-free systems form a separate empirical stratum; no universal signature is assumed.

Every AI training/evaluation sample records generator family, checkpoint/version, decoder/tokenizer type, and known scale factors where available. Unknown provenance stays `unknown`. These fields support stratified evaluation only and do not expand the binary public prediction contract.

Stage 1's synthetic early exit is disabled by default. Enabling it requires a locked high-precision result across authentic images, held-out generator families/checkpoints, and independent transforms. Missing or weak spectral evidence is neutral and always falls through to Stage 2.

## Texture-Aware Local-Detail Head

A multi-scale Laplacian/Sobel map ranks non-overlapping patches by local detail energy. The model retains both a global image view and a small fixed number of selected patches so selection does not discard semantic context. Selected patches share the live CLIP backbone or use a small shared head, then soft attention aggregates their representations.

Patch energy only chooses candidate regions. It is not an authenticity score. The learned head must distinguish structural texture evidence from ordinary loss of detail using authentic and AI-generated training examples that receive the same **single** transformations. No fixed “smooth,” “regular,” low-variance, LBP, GLCM, or OCR threshold may directly change the verdict.

LBP and GLCM may be logged as interpretable diagnostics. OCR-derived structure is a stretch feature evaluated only on detected text regions; missing text and low OCR confidence are neutral. Bespoke hair, foliage, and reflection detectors are not part of the initial model.

The detector never creates extra blurred, compressed, or resized variants at inference. Such a degradation-curve method would chain transformations on already transformed inputs and fall outside the agreed evaluation protocol.

## PRNU-Coherence Feature Extractor

PRNU is a low-amplitude, multiplicative pattern associated with a physical image sensor. The extractor denoises one input image, computes its residual, and summarizes block-wise residual statistics, irradiance coupling, spatial self-consistency, and candidate CFA periodicity. The normalized vector is fused with CLIP features before the binary head.

This is an experimental single-image proxy, not classical device attribution. There is no known-camera reference fingerprint and no multi-image sensor estimate, so the feature cannot verify which camera captured an image—or prove that a camera captured it at all. PRNU is never a gate: weak PRNU cannot independently mean `ai_generated`, and strong sensor-like noise cannot independently mean `authentic`.

DSNU is excluded because a reliable estimate normally needs dark-frame/reference calibration, while correction can suppress the residual before export. That evidence is unavailable from one unknown image.

## Color and Optical Feature Extractors

### Inter-channel correlation

Compute standardized pairwise correlation statistics in both RGB and Lab, globally and over local windows. Include masks/coverage for nearly constant channels or windows. The compact vector is concatenated after CLIP; correlation maps are not inserted as extra channels into the frozen three-channel backbone.

Lab is a candidate, not a guaranteed winner. Per-channel standardization removes sensitivity to affine shifts within each channel, but saturation, clipping, nonlinear color conversion, and other jitter effects can still change the correlation structure. RGB-only, Lab-only, and combined variants are selected on locked clean and independent color-jitter results.

### Optical aberrations

Estimate lateral chromatic aberration with a radial inter-channel scale/center model and emit its parameters, residual, spatial consistency, edge support, and confidence. A simple channel-edge cross-correlation may initialize the fit but cannot replace the radial model.

Optionally estimate barrel/pincushion distortion from long line/arc support with a one-parameter radial model. This is scene-dependent: when geometric support is insufficient, the feature is masked and its confidence/coverage is recorded.

Neither feature authenticates an image. Camera correction may remove optical effects, generators may imitate them, and blur/resize/crop/color processing may corrupt their estimates. Missing or low-confidence values remain neutral. Both extractors run inline and add no model, stage, queue, or async service.

## Offline Comparison Model

ConvNeXt-Tiny may be trained once as an offline baseline. It is not loaded alongside CLIP during inference. If it wins the locked binary evaluation, it can replace CLIP as the single shipped backbone.

## Dataset Contract

Only samples with clear source provenance are eligible:

| Label | Included | Excluded |
|---|---|---|
| `authentic` | Genuinely captured source images | AI-enhanced, face-swapped, composited, or ambiguous images |
| `ai_generated` | Fully synthetic images with no authentic source | Image-to-image edits, inpainting of authentic images, partial-AI or mixed content |

A clean source and every test variant derived from it must remain in the same split to prevent leakage.

## Independent Transformation Protocol

Each robustness sample is produced directly from its clean source with exactly one transformation and one parameter setting:

- JPEG compression
- Gaussian blur
- Resize and restore
- Gaussian noise
- Color jitter
- Center crop

No transformed output is passed into another transformation. Results are recorded separately for every transform and parameter; there are no mixed, sequential, or overlaid transformation cases.

## Evaluation and Score

The final score is split equally:

`Final score = 0.50 × clean accuracy + 0.50 × robustness score`

The robustness score is the mean binary accuracy across all independent transform-and-parameter test sets. For the clean set and every robustness row, also report:

- authentic accuracy (R.Acc.)
- AI-generated accuracy (F.Acc.)
- overall binary accuracy
- binary confusion matrix
- Expected Calibration Error (ECE)
- PRNU-only, CLIP-only, and fused accuracy
- texture-only, CLIP + texture, CLIP + PRNU, and full-fusion accuracy
- distribution of `prnu_coherence` by label
- RGB-only, Lab-only, chromatic-aberration-only, eligible radial-distortion-only, and incremental full-fusion results
- color/optics feature validity, coverage, and confidence by label and transform
- frequency-only accuracy and candidate fast-track precision/coverage by decoder family, checkpoint holdout, and transform

This reporting exposes class collapse even when the aggregate 50/50 score looks acceptable.

PRNU is expected to be most recoverable after cropping and fragile under JPEG compression, blur, resize, and added noise. Those are hypotheses to test independently, not assumptions used to alter labels or scoring.

Retain the texture head only if it improves the locked 50/50 score or a pre-agreed robustness diagnostic without causing unacceptable class-specific regression. Also measure correlation between texture, frequency, and PRNU outputs to identify redundant low-level signals.

Apply the same retention rule to color and optical features. Color-jitter parameters must be sampled from the same distribution for both labels, and feature degradation under color jitter must be reported rather than normalized away.

Apply the retention rule to every spectral sub-feature as well. Remove family-specific cues that add no held-out value, and measure correlation with texture and PRNU so redundant high-frequency evidence is not counted as independent support.

## Offline Improvement Loop

Self-training remains binary. High-confidence pseudo-labels are accepted using separate thresholds for `authentic` and `ai_generated`, then down-weighted during retraining. Low-confidence or uncertain-provenance images are reviewed or excluded. A candidate checkpoint is rolled back if either label's locked-validation accuracy regresses beyond the agreed tolerance.
