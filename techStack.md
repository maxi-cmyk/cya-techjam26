# Tech Stack

Breakdown of what is used, why it was chosen, and how it fits into the pipeline. All models are confirmed **<2B parameters**; only one backbone is loaded at inference time (see design.md for the architecture rationale).

## 1. Primary Backbone (Inference-Time)

### Frozen CLIP ViT-L/14 — linear-probe detector
- **What:** OpenAI CLIP's vision tower, frozen, with a lightweight linear/MLP head trained on top.
- **Why chosen over alternatives:** most degradation-robust known detector family (UnivFD/CLIPDetection-style results hold up far better under JPEG-50 than frequency/patch-artifact methods); CLIP's web-pretraining gives it inherent robustness to noisy, degraded images; generalizes best cross-generator (+15 mAP on unseen generators vs. from-scratch CNNs/ViTs).
- **JPEG rationale:** compression preferentially removes high-frequency information where many generator-specific fingerprints reside. CLIP therefore supplies the principal representation-level signal; frequency, texture, PRNU, color, and optics remain auxiliary features that must demonstrate incremental value after JPEG matching and degradation.
- **Size:** ~304M params (ViT-L/14@336px vision tower only — the text encoder is unused and excluded from the inference budget). A ViT-B/32 variant (~86–87.5M) is a smaller fallback if resource-constrained further.
- **Reference implementations to build from:**
  - `WisconsinAIVision/UniversalFakeDetect` (UnivFD, CVPR 2023) — canonical CLIP linear-probe.
  - `mever-team/rine` (RINE, ECCV 2024) — intermediate CLIP block CLS tokens + Trainable Importance Estimator; reports ~1-epoch (~8 min) training and strong avg-improvement across 20 test datasets. Candidate to swap in for the plain linear probe.
  - `grip-unina/ClipBased-SyntheticImageDetection` — CLIP-feature detector, documented robust across DALL·E-3/MJv5/Firefly and post-processing, ships Docker + CSV batch tooling.
- **Fallback candidate:** AIDE (ICLR 2025) — hybrid CLIP-semantic + frequency-patch experts, stronger on clean benchmarks but heavier; switch to this if Chameleon-benchmark fake-recall drops below ~50% after augmentation.

### ConvNeXt-Tiny — offline comparison only
- **What:** trained from scratch/fine-tuned once, entirely offline.
- **Why it exists:** produces the Table 1 "why we chose CLIP" ablation — a lightweight CNN baseline to justify the backbone decision empirically rather than by assertion.
- **Not shipped:** never loaded at inference time. If it empirically outperforms CLIP in the team's own robustness tests, it becomes the swapped-in single backbone instead (architecture stays backbone-agnostic).
- **Rationale for even trying it:** LAID (2025) shows lightweight CNNs can match transformers on GenImage at far lower cost — worth a cheap offline check before committing.

## 2. Stage 0 — Provenance (Pre-Model, Near-Zero-Cost)

### C2PA reader/verifier
- **Library:** `c2pa-python` (official CAI bindings, wraps `c2pa-rs`), MIT OR Apache-2.0, Python 3.10+.
- **What it does:** parses and cryptographically verifies embedded C2PA manifests. This is metadata parsing, not model inference — negligible compute/memory cost.
- **How used:** one-directional early exit. Only a *valid, signature-verified* manifest that explicitly claims AI-generation can shortcut to `ai_generated`. Absence, invalid signature, or a manifest claiming "real" all fall through to the model pipeline — never trusted as evidence of authenticity, since C2PA metadata is routinely stripped by re-encoding/compression/resize.
- **Operation mode:** fully offline for manifest reading, validation, and signature verification. Optional network calls (trust-list checks, OCSP/CRL revocation, remote-manifest fetch) are off by default and should stay off for the hackathon deliverable.
- **Known limitation, by design:** none of the reference datasets (SID_Set, CIFAKE, WildFake) reliably carry C2PA manifests, so this stage will trigger on ~0% of standard eval data — a small self-synthesized C2PA-tagged test set is needed to demonstrate the mechanism at all.

## 3. Stage 1 — Frequency Feature Bank and Conditional Fast-Track

- **What:** a deterministic family-stratified feature bank using FFT/DCT magnitude, radial/angular power, periodic-peak prominence, residual autocorrelation, and NPR-style local pixel dependencies.
- **Role:** always supplies an auxiliary vector to Stage 2. A one-directional synthetic fast-track exists in the architecture but is disabled until locked validation approves it. It can never independently return `authentic`.
- **GAN/CNN decoder hypothesis:** transposed-convolution or related upsampling can create checkerboards, spectrum replicas, high-frequency decay discrepancies, and local pixel dependencies.
- **Latent VAE/VQ hypothesis:** periodic peaks may follow the decoder's upsampling factor across diffusion, transformer, GAN, or autoregressive generators; do not misattribute them automatically to denoising or token prediction.
- **Pixel-space/decoder-free hypothesis:** evaluate radial/angular and residual-spectrum features separately without assuming latent-decoder periodicity.
- **Autoregressive/token hypothesis:** stratify by tokenizer and decoder when known. Do not claim a universal natural autoregressive fingerprint; shared VAE/VQ decoders can blur family boundaries.
- **Metadata:** record generator paradigm, model/checkpoint version, decoder/tokenizer family, and known upsampling factor. Unknown values remain `unknown`.
- **Fast-track gate:** require pre-agreed high precision on locked authentic images, held-out generator families/checkpoints, and all applicable independent transforms before enabling. Otherwise every image falls through to Stage 2.
- **Why conservative:** JPEG, blur, resize, architectural changes, spectral regularization, and explicit artifact suppression can erase or alter the signal. Absence of an expected peak is neutral.
- **JPEG-focused ablation:** compare phase-spectrum and magnitude-spectrum features separately; phase is a candidate for greater compression stability, not an assumed universal detector.

## 4. Stage 2 — Main Classification Heads (all sit on the frozen CLIP backbone)

| Head | Purpose | Notes |
|---|---|---|
| Binary AIGC head | Core `P(ai_generated)` | Supports both the clean-accuracy and robustness halves of the final score |
| Patch aggregation layer | Combines per-patch scores | Soft-probability averaging or attention pooling — replaces hard voting, which underperforms and suffers "Few-Patch Bias" (over-relying on a minority of patches) |
| Texture-aware detail head | Preserves and aggregates local structural evidence | Uses a fixed patch budget and learned fusion; never treats low detail as an AI rule |
| PRNU feature fusion | Adds physical-capture residual statistics | Small auxiliary vector only; never a standalone gate or verdict |
| Color/optics feature fusion | Adds RGB/Lab dependence and confidence-aware optical estimates | Deterministic inline extraction; missing cues are masked to neutral |
| Calibration layer | Temperature scaling | Binary ECE, equal-mass binning; explicitly re-tested (not re-fit) on each independently transformed set |

The binary head is the only classification head. Face-, edit-, and mixed-content specialists are intentionally excluded from the project scope.

### Texture-aware detail head

- **Region proposal:** OpenCV/scikit-image multi-scale Laplacian or Sobel energy map, followed by top-k non-overlapping patch selection.
- **Representation:** reuse the frozen CLIP vision backbone for global and selected local views, or use a small shared patch head if latency is too high.
- **Aggregation:** soft attention or probability averaging under a fixed patch and latency budget.
- **Training rule:** transformed copies are generated independently from clean sources; each copy receives one transform and one parameter setting only.
- **No hand-coded verdicts:** absolute sharpness, smoothness, edge density, LBP, GLCM, and OCR confidence are not authenticity thresholds.
- **Optional diagnostics:** scikit-image LBP/GLCM summaries for error analysis, not required public output.
- **OCR stretch ablation:** a lightweight text detector/OCR may contribute learned structural features only when text is present. Missing/low-confidence text is neutral, and multilingual/transformed evaluation is required before retention.
- **Rejected test-time method:** do not apply extra blur/JPEG/resize passes to estimate a degradation curve; this chains transformations and increases inference cost.
- **Retention gate:** keep the head only if locked ablations improve the 50/50 score or an agreed robustness diagnostic without unacceptable R.Acc./F.Acc. regression.

### PRNU-coherence extractor

- **Signal:** photo-response non-uniformity, modeled as a weak multiplicative sensor pattern in the denoising residual.
- **Candidate implementation:** wavelet or established PRNU-style denoising, residual extraction, block-wise NumPy/scikit-image analysis, and normalized feature fusion in PyTorch.
- **Features:** residual variance/energy, local luminance-to-residual coupling, cross-patch self-consistency, and CFA/sensor-periodicity statistics.
- **Boundary:** this is a single-image coherence proxy. Classical normalized correlation against a camera fingerprint is unavailable because the source device and reference image set are unknown.
- **Decision rule:** no direct decision rule. The feature vector is learned jointly with CLIP features, and missing/weak PRNU never independently triggers a synthetic verdict.
- **Robustness expectation:** crop should preserve the signal best; JPEG, blur, resize, and Gaussian noise are expected to weaken it substantially. Each claim must be checked in the independent transformation table.
- **Ablation gate:** retain PRNU fusion only if it improves the locked 50/50 score or a pre-agreed class-specific robustness measure without causing unacceptable regression.
- **DSNU:** considered but not implemented because reliable isolation normally needs dark-frame/reference calibration, while correction can suppress the residual before export. One unknown image supplies neither a reference nor a dependable estimate.

### Inter-channel color correlation

- **Basis:** CHROMA reports discriminative inter-channel structure in both RGB and Lab; neither space is assumed universally superior.
- **Implementation:** OpenCV color conversion plus NumPy global/local-window standardized correlations for R/G, R/B, G/B, L/a, L/b, and a/b.
- **Fusion:** concatenate a compact feature/coverage vector after CLIP. Do not add correlation maps as channels to the frozen CLIP input.
- **Numerical guardrails:** epsilon-stabilized variance, clipping checks, and masks for low-variance channels/windows.
- **Color-jitter handling:** normalize per channel and train with label-matched jitter distributions. This helps with affine brightness/contrast changes but does not guarantee saturation-jitter invariance.
- **Retention gate:** compare RGB-only, Lab-only, combined, and no-color variants across clean and every independent transform set.

### Optical-aberration vector

- **Chromatic aberration:** initialize from channel-edge alignment if useful, then fit the Johnson–Farid-style radial expansion/contraction model around an optical center. Emit fit parameters, residual, regional consistency, edge support, and confidence.
- **Radial lens distortion (stretch):** on images with sufficient line/arc support, fit a simple plumb-line barrel/pincushion coefficient and emit coefficient/residual/support/confidence.
- **Neutral masking:** insufficient support or poor fit produces a masked neutral feature, not `authentic` or `ai_generated` evidence.
- **Known confounds:** in-camera/software correction, crop, resize, blur, weak or curved scene geometry, channel clipping, and generator-simulated optics.
- **Runtime:** deterministic OpenCV/NumPy/SciPy-style optimization inline in the current batch loop; no new neural model, pipeline stage, queue, or asynchronous worker.
- **Retention gate:** require incremental improvement on the locked 50/50 score or a pre-agreed robustness metric and report usable-image coverage.

## 5. Datasets

| Dataset | Role |
|---|---|
| `saberzl/SID_Set` (Hugging Face) | Synthetic + real source pool |
| CIFAKE (Kaggle, `birdy654/cifake-real-and-ai-generated-synthetic-images`) | Synthetic + real source pool (small, CIFAR-scale — no container metadata) |
| WildFake (ModelScope) | Synthetic + real source pool (translate via platform's translation button before use) |
| COCO val2017 (subset) | Non-AIGC portion of the **validation-only** benchmark (not for training) |
| DALL·E Advanced (subset) | AIGC portion of the **validation-only** benchmark (not for training) |
| GenImage / AIGCDetectBenchmark | Canonical external training/eval reference sets |
| Chameleon (`AIDE` repo companion, ~26K images) | Stress-test benchmark — human-Turing-test-passing AI fakes; expect ~70% accuracy even for good detectors |

Mixed-origin, AI-edited, face-swapped, composited, and ambiguous-provenance samples must be filtered out of both labels.

## 6. Dataset Normalization and Transformation Pipeline

### Dataset normalization

Preserve immutable source bytes for provenance and offline native-forensics ablations. Create the canonical primary-model view by decoding and re-encoding both labels with an identical, label-independent policy. Version and record the JPEG library, RGB conversion behavior, quality, chroma subsampling, progressive/baseline mode, and metadata handling. Compare fixed Q96 with a narrow high-quality distribution such as Q95-Q100 before freezing the policy. This matched pass is never rerun by the inference script.

### Training robustness augmentation

Generate one independently sampled robustness variant from the canonical matched clean parent when the sampler selects a transformed example. Use the same cell probabilities, parameter distributions, and seeds for both labels. Matched normalization removes compression-history bias; augmentation teaches degradation tolerance.

### Independent evaluation transformations

Applied uniformly to both binary labels. For evaluation, every variant starts from the canonical matched clean derivative and receives exactly one benchmark transformation with one parameter setting:
- JPEG re-encode: quality 90/70/50/30
- Gaussian blur: σ 0.5/1.0/2.0
- Resize: 0.5× or 0.25× bilinear downsample, then bilinear restore to exact parent dimensions
- Gaussian noise: σ 0.02/0.05/0.10
- Color jitter: brightness/contrast/saturation ±20%
- Center crop: 80%

No test image may combine, chain, or overlay benchmark transformations. Dataset normalization is baseline construction and does not change the rule that each robustness row contains exactly one benchmark transformation.

### Resize implementation contract

Use one pinned library and version with bilinear interpolation in both directions. Lock antialiasing, intermediate-dimension rounding, RGB/channel handling, dtype/range conversion, and exact restoration dimensions. Keep transformed pixels as arrays or save them losslessly; do not JPEG-encode a resize result. Area-to-bicubic, nearest-neighbor, and Lanczos are optional sensitivity ablations, not primary benchmark settings.

The shipped detector never generates resize variants or repeats matched JPEG normalization. It processes the exact received view. The global branch performs the required CLIP input conversion, while the local branch selects multiple detail-rich crops at available resolution before that conversion and pads them when needed.

For color jitter, both labels use the same brightness/contrast/saturation parameter distributions and sampling probability. RGB/Lab correlation, chromatic-aberration, and radial-distortion outputs are reported separately on that row because color jitter directly perturbs channel statistics and may reduce edge-fit quality.

## 7. Scoring and Metrics

- **Accuracy component (50%):** binary accuracy on clean images.
- **Robustness component (50%):** mean binary accuracy across every independent transform-and-parameter set.
- **Required diagnostics:** R.Acc., F.Acc., binary confusion matrix, and ECE for clean data and each transformation row.
- **PRNU diagnostics:** PRNU-only, CLIP-only, and fused accuracy plus `prnu_coherence` distributions for clean and independent transform rows.
- **Texture diagnostics:** texture-only, CLIP + texture, CLIP + PRNU, and full-fusion accuracy; report R.Acc./F.Acc. and overlap with frequency/PRNU signals.
- **Color/optics diagnostics:** RGB-only, Lab-only, chromatic-aberration-only, eligible radial-distortion-only, incremental fusion results, and fit validity/coverage for every transform row.
- **Frequency diagnostics:** frequency-only R.Acc./F.Acc., per-family/checkpoint recall, candidate fast-track precision/coverage, and degradation for every independent transform row.
- **Compression-bias diagnostics:** nuisance-only label predictiveness before and after matching, plus original/unmatched CLIP, matched-view CLIP, and same-received-view fusion ablations.
- **Resize diagnostics:** global-only, local-only, and global-plus-local accuracy; authentic false-positive rate; synthetic false-negative rate; and Stage 1 fast-track precision/coverage for resize-0.5× and resize-0.25×.

## 8. External / Commercial APIs (Offline-Comparison Baselines Only — Never in the Inference Path)

| API | Use | Free tier |
|---|---|---|
| AI or Not | Binary AI-vs-real baseline comparison | 20 image checks + 1M words text/month |
| Sightengine | `genai` model baseline, scored only on eligible fully synthetic/authentic samples | 2,000 ops/month; AI checks cost 5 ops each (~400 checks/mo) |
| Hive Moderation | AI-generated classifier baseline, restricted to the binary scope | $50 free credits |
| Winston AI | Full forensics incl. EXIF/IPTC/C2PA parsing | 2,500 free credits (~8 free images) |
| Google SynthID | Watermark-only detector for Google/partner content | Waitlist; **not usable as a general detector** |

These exist purely to produce a comparison row in the demo/report — the product must remain fully local and offline for the actual deliverable.

## 9. Development Tooling (to be finalized by team)
- Standard ML stack: PyTorch, Hugging Face Transformers (for CLIP loading), scikit-learn (calibration/metrics), pandas (logging/tables).
- Environment: VSCode / Colab / Jupyter (per Devpost write-up requirement — must be disclosed in submission).
- Versioned checkpoint logging for the iterative/self-training loop.
- A synchronous directory batch loop is sufficient; no queue, async worker, or service infrastructure is planned.

## 10. Explicitly Rejected / Deprioritized
- **Running CLIP + ConvNeXt-Tiny simultaneously at inference** (v7 design) — dropped in v8 for a single-backbone design; halves inference-time model count and directly serves the local/memory-constrained requirement.
- **ViT-Small trained from scratch** — deprioritized; LAID (2025) findings and hackathon-scale data volume favor a frozen, pretrained CLIP backbone.
- **Hard-vote patch aggregation** — replaced by soft-probability averaging/attention pooling.
- **DSNU extraction** — deprioritized because the available single-image input lacks dark-frame/reference calibration and may already have passed through fixed-pattern-noise correction.
- **DIRE-style reconstruction error** — deprioritized because it adds diffusion inversion/reconstruction and a separately loaded diffusion/VAE model family, conflicting with the current single-backbone, local-compute design. Faster latent variants may be reconsidered only if the lightweight auxiliary paths fail.

## 11. PRNU Technical References

- Martin, *Significance of image brightness levels for PRNU camera identification* — documents residual extraction, multi-image reference-fingerprint estimation, correlation scoring, and brightness-dependent error rates: https://onlinelibrary.wiley.com/doi/10.1111/1556-4029.15673
- Martin-Rodriguez, *Testing Robustness of Camera Fingerprint (PRNU) Detectors* — motivates explicit robustness testing under post-processing: https://arxiv.org/abs/2102.09444
- Chen et al., *Uniformity Correction of CMOS Image Sensor Modules for Machine Vision Cameras* — describes calibration-based DSNU/PRNU correction and measured fixed-pattern-noise suppression: https://pmc.ncbi.nlm.nih.gov/articles/PMC9783237/

## 12. Texture and Detail Technical References

- Konstantinidou et al., *TextureCrop: Enhancing Synthetic Image Detection through Texture-based Cropping* — evaluates texture-guided cropping for synthetic-image detection: https://openaccess.thecvf.com/content/WACV2025W/SynRDinBAS/html/Konstantinidou_TextureCrop_Enhancing_Synthetic_Image_Detection_through_Texture-based_Cropping_WACVW_2025_paper.html
- Mu et al., *No Pixel Left Behind: A Detail-Preserving Architecture for Robust High-Resolution AI-Generated Image Detection* — combines local native-detail tiles with a global view and explicitly studies compression: https://openreview.net/pdf?id=d3bd6393bdc32d6878e6d07aaceb1bbf5a6451cf
- Zhang et al., *TextFake: Benchmarking AI-Generated Image Detection on Text-Rich Images* — shows that text-rich detection and perturbation robustness remain difficult, supporting OCR as an ablation rather than a trusted shortcut: https://arxiv.org/abs/2606.01050
- Zhu et al., *GenImage* — defines cross-generator and degraded-image detection tasks covering low resolution, blur, and compression: https://openreview.net/forum?id=GF84C0z45H

## 13. Color and Optical Technical References

- Sotelo et al., *CHROMA: Detecting AI-Generated Images Through Inter-channel Color-Space Correlations* — motivates RGB/Lab pairwise correlation features: https://arxiv.org/abs/2606.08864
- Johnson and Farid, *Exposing Digital Forgeries Through Chromatic Aberration* — models lateral chromatic aberration as radial inter-channel expansion/contraction and estimates spatial consistency: https://farid.berkeley.edu/downloads/publications/acm06c.pdf
- Farid and Popescu, *Blind Removal of Lens Distortion* — describes single-image radial-distortion estimation from image statistics: https://farid.berkeley.edu/downloads/publications/josa01.pdf
- *Line-Based Correction of Radial Lens Distortion* — supports the optional plumb-line estimator when sufficient straight-edge geometry is available: https://doi.org/10.1006/gmip.1996.0407
- Wang et al., *DIRE for Diffusion-Generated Image Detection* — uses diffusion inversion/reconstruction error and motivates the documented compute/dependency exclusion: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.pdf

## 14. Frequency-Fingerprint Technical References

- Durall et al., *Watch Your Up-Convolution* — links CNN upsampling to spectral-distribution errors: https://openaccess.thecvf.com/content_CVPR_2020/html/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.html
- Corvi et al., *Intriguing Properties of Synthetic Images: From GANs to Diffusion Models* — compares GAN, diffusion, and VQ-GAN spectra/autocorrelation and connects periodic peaks to decoder upsampling factors: https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_to_CVPRW_2023_paper.html
- Tan et al., *Rethinking the Up-Sampling Operations in CNN-based Generative Network* — motivates local neighboring-pixel relationships across GAN and diffusion decoders: https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html
- Dong et al., *Think Twice Before Detecting GAN-Generated Fake Images From Their Spectral Domain Imprints* — demonstrates that spectral artifacts can be mitigated and frequency-only detectors can fail: https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Think_Twice_Before_Detecting_GAN-Generated_Fake_Images_From_Their_Spectral_CVPR_2022_paper.html
- Karageorgiou et al., *Any-Resolution AI-Generated Image Detection by Spectral Learning* — documents cross-generator spectral variation and a real-spectrum modeling alternative: https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html
