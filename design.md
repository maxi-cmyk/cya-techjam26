# Design — System Architecture & Evaluation

This document describes the shipped inference pipeline and the offline evaluation loop. See `techStack.md` for implementation choices and `PRD.md` for product requirements.

## 1. Design Principles

1. **Binary-only classification.** The only labels are `authentic` and `ai_generated`. Mixed-origin, composited, face-swapped, AI-enhanced, and otherwise AI-edited images are outside the dataset and product scope.
2. **One transform per test image.** Every robustness variant is generated directly from its clean source with exactly one transformation and one parameter setting. A transformed output is never used as input to another transformation.
3. **One-directional shortcuts.** A fast path may shortcut toward `ai_generated`, never toward `authentic`. Missing forensic evidence does not prove authenticity, especially after transformations erase artifacts.
4. **Single live backbone.** Models may be compared offline, but only one backbone is loaded while scoring a test directory.
5. **PRNU is auxiliary evidence.** A single-image PRNU-coherence proxy may support the backbone, but it is not a verified camera fingerprint and can never make a standalone verdict.
6. **Texture is learned, not thresholded.** Fine-detail regions inform a lightweight learned head; rules such as “smooth equals AI” are prohibited because transformations can make authentic images equally smooth.
7. **Color and optics require confidence.** Deterministic color-correlation and optical-fit features run inline, but absent or weak physical-camera cues are neutral rather than proof of synthesis.
8. **Frequency signatures are family-dependent.** Stage 1 measures multiple spectral and neighborhood-pixel cues and is evaluated by generator/decoder family; it does not assume one GAN/diffusion/autoregressive fingerprint.

## 2. Inference Architecture

```text
Input image
    |
    v
Stage 0 — C2PA provenance check
    |-- valid signed AI-generation claim --> high-confidence ai_generated
    `-- missing, invalid, stripped, or other claim
                    |
                    v
Stage 1 — family-aware frequency feature bank
    |-- validated clean, high-confidence signature --> candidate ai_generated fast-track
    `-- uncertain, transformed, or authentic-leaning
                    |
                    v
Stage 2 — single frozen CLIP-ViT backbone
          + binary classification head
          + soft patch aggregation
          + texture-aware local-detail head
          + PRNU-coherence auxiliary features
          + color-correlation and optical-aberration features
          + temperature scaling
                    |
                    v
       image_path + pred = P(ai_generated)
```

### Stage 0 — Provenance

C2PA is metadata evidence, not proof by absence. Only a cryptographically valid manifest that explicitly identifies AI generation can trigger an early synthetic verdict. Missing, stripped, invalid, or authenticity-claiming metadata falls through to the model.

### Stage 1 — Frequency feature bank and conditional fast-track

A deterministic feature bank measures 2-D log-spectrum structure, radial/angular power, periodic peaks, autocorrelation periodicity, and local neighboring-pixel relationships. These cues cover more than generic “high-frequency energy” and are forwarded to Stage 2 with validity/confidence values.

The expected signatures are organized by the image-producing decoder, not only the marketing label of the generator:

| Family | Candidate evidence | Boundary |
|---|---|---|
| CNN/GAN decoders | checkerboard/periodic replicas, anomalous high-frequency decay, upsampling-linked pixel dependencies | architecture changes or spectral suppression can remove them |
| Latent generators with VAE/VQ decoders | decoder-scale periodic peaks and autocorrelation patterns; mid/high-frequency residual differences | can span diffusion, transformer, or autoregressive paradigms and often reflects the shared decoder/upsampler |
| Pixel-space or otherwise decoder-free generators | empirical radial/angular or residual-spectrum differences | no latent-decoder periodicity assumption; must be learned and validated separately |
| Autoregressive/token-based | possible tokenizer, VAE/VQ decoder, patch-grid, or multi-scale traces | no universal natural spectral signature is assumed; use held-out evidence only |
| Unknown | generic multi-scale features only | never assigned to a family merely from a spectral guess |

Generator family, decoder type, upsampling factor, checkpoint, and version are dataset metadata when known; they are not extra public output classes. Evaluation holds out complete generator families/checkpoints so a detector cannot pass by memorizing one model fingerprint.

The early exit is **disabled by default**. It may be enabled only if a pre-agreed high-precision threshold holds on locked authentic data, held-out generator families, and every applicable independent-transform set. Even then, Stage 1 may only return `ai_generated`, never `authentic`; all other cases fall through and retain the feature vector for Stage 2.

JPEG compression, blur, resize, and deliberate spectral correction can hide or alter these cues. A missing checkerboard or expected peak is therefore neutral. The feature bank is an auxiliary forensic signal, not proof of origin.

### Stage 2 — Main model

Most inputs pass through a frozen CLIP-ViT backbone with a lightweight binary head. A global image view is combined with selected local-detail patches using soft averaging or attention pooling. Texture, PRNU, color-correlation, and optical-aberration vectors are concatenated with the global representation before the final binary head. Temperature scaling converts the output into a calibrated `P(ai_generated)`.

### Texture-aware local-detail path

The texture path preserves local evidence that can disappear when an entire high-resolution image is resized to the backbone input. It does not attempt to hard-code hair, skin, foliage, text, or reflection defects.

1. Compute a cheap multi-scale Laplacian/Sobel energy map on the current input.
2. Select a small, fixed number of non-overlapping texture-rich patches, while retaining the global image view for context.
3. Encode patches with the same live backbone or a small shared patch head; aggregate them with soft attention.
4. Fuse the resulting texture vector with global CLIP, frequency, and PRNU features in the binary head.

Energy selects where to look, not what verdict to produce. The head learns structural differences from labeled data. Raw sharpness, smoothness, edge density, LBP, GLCM, or OCR confidence must not be used as fixed authenticity thresholds.

Each clean training source may produce separate single-transform copies. The texture head sees authentic and AI-generated examples under the same JPEG, blur, resize, noise, color-jitter, and crop conditions, but no training or test copy chains transformations. This is intended to teach the head the difference between generator-linked structure and ordinary degradation without violating the benchmark boundary.

LBP/GLCM descriptors may be logged for explainability and ablation, but are not core decision rules. Category-specific hair, foliage, or reflection detectors are excluded from the initial build because they add models and brittle assumptions without covering every image.

Text/OCR is a stretch ablation only. If a text region is detected, structural OCR features may be added to the learned fusion vector; no text, low confidence, stylized fonts, unsupported languages, blur, or compression must remain neutral rather than automatically implying AI generation.

Inference-time “degradation curves” are out of scope. Applying new blur or JPEG operations to an input that may already be transformed would create chained transformations, violate the independent-test protocol, and multiply inference cost.

### PRNU auxiliary path

Photo-response non-uniformity (PRNU) is a weak, multiplicative sensor pattern introduced during physical capture. A fully synthetic image has no true camera-sensor PRNU, although a generator or later processing can synthesize noise that resembles it.

For each input image, the auxiliary path:

1. denoises the image and computes the residual `W = image - denoised_image`;
2. divides the residual and luminance image into aligned blocks;
3. measures residual energy, the relationship between residual strength and local irradiance, cross-patch self-consistency, and candidate CFA/sensor periodicity;
4. produces a small normalized feature vector and a scalar `prnu_coherence` diagnostic for fusion with CLIP features.

This is deliberately called **PRNU coherence**, not camera identification. Classical device-level PRNU verification needs a reference fingerprint estimated from multiple images from the same sensor. With one unknown image, the system can only test whether its residual looks physically sensor-like. Low coherence can result from either synthesis or ordinary redistribution, so it cannot independently imply `ai_generated`; high coherence can also be simulated and cannot independently prove `authentic`.

DSNU is not implemented. Reliable dark-signal isolation normally needs dark-frame/reference calibration, and fixed-pattern-noise correction can suppress the signal. A single exported image therefore does not provide a dependable DSNU estimate.

### Color-correlation auxiliary path

The CHROMA-inspired path measures pairwise dependence between image channels. It is deterministic and runs on the current input before fusion:

1. decode sRGB consistently and derive both RGB and Lab representations;
2. normalize each channel per image, with an epsilon and low-variance mask to avoid unstable correlations;
3. compute compact global and local-window pairwise correlation statistics for R/G, R/B, G/B and L/a, L/b, a/b;
4. concatenate the statistics and validity/coverage values after CLIP rather than changing the frozen three-channel CLIP input layer.

RGB and Lab are both candidates. Lab separates luminance from chrominance, and normalized correlation is invariant to simple per-channel affine changes, but neither choice guarantees invariance to saturation shifts, nonlinear conversion, clipping, or general color jitter. The locked ablation chooses RGB, Lab, both, or neither.

### Optical-aberration auxiliary path

The optical vector combines two related but distinct estimates:

- **Lateral chromatic aberration:** fit a low-parameter radial expansion/contraction between color channels around an estimated optical center, using edge support and channel alignment/mutual information. Log the fitted scale/center, residual, spatial consistency, support, and confidence.
- **Radial lens distortion (stretch):** when enough long line or arc support exists, fit a simple barrel/pincushion coefficient with a plumb-line-style objective. Log the coefficient, fit residual, support, and confidence.

Claude's proposed per-channel edge cross-correlation is acceptable as a cheap initialization, but a single global offset is not the physical lateral-CA model because aberration varies radially across the image. The final estimator must fit and validate the spatial model.

These cues were developed primarily for camera calibration or inconsistency/tampering analysis, not as proof that an unknown image passed through a camera. Modern lens correction can remove them, crop/resize can change their geometry, weak-edge scenes can make them unidentifiable, and generators can simulate them. Therefore, invalid/low-confidence fits are masked to neutral and no optical statistic can issue a verdict by itself.

All color and optics features run synchronously in the existing per-image batch loop. They add no service, queue, asynchronous worker, external API, new pipeline stage, or separately loaded neural model.

The public JSON contains only `image_path` and `pred`. Diagnostics such as `verdict_source`, transform name, and degradation-bias measurements remain internal.

## 3. Data Assembly and Leakage Controls

- **Strict provenance:** authentic samples originate from genuine capture; AI samples are fully synthetic with no authentic source.
- **Exclusions:** mixed-origin, image-to-image edits of authentic sources, inpainting of authentic images, face swaps, AI enhancement, compositing, and ambiguous provenance.
- **Split integrity:** a clean source and every variant derived from it belong to one split only.
- **Format matching:** balance resolution, aspect ratio, and file-format artifacts across both labels before drawing generator-family holdouts.
- **Class balance:** keep the clean test set and every robustness set balanced between `authentic` and `ai_generated`, or report balanced accuracy if exact balance is impossible.

## 4. Independent Transformation Protocol

For every clean test image, construct separate variants for each parameter setting:

| Transform | Parameter settings |
|---|---|
| JPEG compression | quality 90, 70, 50, 30 |
| Gaussian blur | sigma 0.5, 1.0, 2.0 |
| Resize and restore | 0.5x, 0.25x |
| Gaussian noise | sigma 0.02, 0.05, 0.10 |
| Color jitter | brightness/contrast/saturation +/-20% |
| Center crop | retain 80% |

Every cell above is its own evaluation set, created from the clean source. There are no cases such as JPEG plus resize, blur plus noise, repeated transformations, alpha overlays, or blends between consecutive outputs.

For color jitter, the authentic and AI-generated training copies use the same parameter ranges, probability, and random-sampling policy. Per-channel normalization is applied before correlation. This mitigates simple brightness/contrast shifts but does not neutralize saturation changes, so color-correlation and optics results are reported separately for the color-jitter row.

## 5. Evaluation and Scoring

The evaluation is split evenly:

`Final score = 0.50 × clean accuracy + 0.50 × robustness score`

`Robustness score = mean binary accuracy across all independent transform-and-parameter sets`

Report the following for the clean set and every independent transformation row:

- overall binary accuracy
- authentic accuracy (R.Acc.)
- AI-generated accuracy (F.Acc.)
- binary confusion matrix
- Expected Calibration Error (ECE)

The temperature parameter is fitted once on clean validation data and is not refitted on transformed sets. This makes any calibration loss under distribution shift visible.

Generator-generalization results use held-out generator families and the same format-matching rules. A frozen CLIP baseline and the offline ConvNeXt-Tiny comparison may be included as additional rows.

The auxiliary paths are evaluated with **CLIP-only**, single-family feature baselines, incremental additions, and the full fused model. At minimum this covers frequency, texture, PRNU, RGB/Lab correlation, chromatic aberration, and eligible radial-distortion features. Frequency results are additionally split by generator/decoder family and held-out checkpoint. The same ablations run on clean images and every independently generated transformation set. Report both accuracy and feature coverage/confidence: a method that appears accurate only because it abstains on difficult images is not considered robust. Measurements, not expectations, decide which paths remain in the shipped model.

## 6. Offline Improvement Loop

Self-training stays binary and separate from the shipped inference path:

```text
Seed labeled set (~60%) --> train seed model
                               |
Held-out pool (~25%) ----------+
                               v
                    predict binary confidence
                         |              |
                  high confidence   low confidence
                         |              |
                down-weighted       review or exclude
                pseudo-label        ambiguous provenance
                         `------.-------'
                                v
                    retrain candidate checkpoint
                                |
                                v
                    locked validation set (~15%)
                         |              |
                  no regression     either label regresses
                         |              |
                     accept          roll back and log
```

Use separate confidence thresholds for `authentic` and `ai_generated`. Pseudo-labels are down-weighted relative to verified labels. The locked validation set is never pseudo-labeled, and a candidate is rejected if either label regresses beyond the agreed tolerance.

## 7. Known Gaps

- No minimum-resolution or corruption-detection baseline is defined yet.
- Error-analysis tooling for top false positives and false negatives is specified but not built.
- The team must define a reproducible provenance-screening process for excluding mixed and AI-edited samples.
- Whether to expose an aggregated `verdict_source` breakdown in the report remains open; it is not part of the public JSON.
- The single-image PRNU feature definition, block size, denoiser, and acceptance criteria require validation on the locked dataset.
- Texture patch size/count, energy-map scale, aggregation method, and inference budget require locked-data ablation.
- OCR is not approved as a core dependency unless it adds value across languages and transformed text-rich images without increasing authentic-image false positives.
- RGB versus Lab correlation representation, window size, and low-variance handling require locked-data selection.
- Chromatic-aberration and radial-distortion estimators need explicit minimum-support and fit-confidence criteria; their usable-image coverage is unknown.
- The Stage 1 fast-track threshold is not approved until family-stratified held-out results satisfy the agreed precision and robustness gate.
- Autoregressive/token-based coverage depends on obtaining clearly documented generators and decoder metadata; otherwise results remain in the `unknown` family bucket.
