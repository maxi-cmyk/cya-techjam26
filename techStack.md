# Tech Stack

Breakdown of what is used, why it was chosen, and how it fits into the pipeline. All models are confirmed **<2B parameters**; only one backbone is loaded at inference time (see design.md for the architecture rationale).

## 1. Primary Backbone (Inference-Time)

### Frozen CLIP ViT-L/14 — linear-probe detector
- **What:** OpenAI CLIP's vision tower, frozen, with a lightweight linear/MLP head trained on top.
- **Why chosen over alternatives:** most degradation-robust known detector family (UnivFD/CLIPDetection-style results hold up far better under JPEG-50 than frequency/patch-artifact methods); CLIP's web-pretraining gives it inherent robustness to noisy, degraded images; generalizes best cross-generator (+15 mAP on unseen generators vs. from-scratch CNNs/ViTs).
- **Size:** ~304M params (ViT-L/14@336px vision tower only — the text encoder is unused and excluded from the inference budget). A ViT-B/32 variant (~86–87.5M) is a smaller fallback if resource-constrained further.
- **Reference implementations to build from:**
  - `WisconsinAIVision/UniversalFakeDetect` (UnivFD, CVPR 2023) — canonical CLIP linear-probe.
  - `mever-team/rine` (RINE, ECCV 2024) — intermediate CLIP block CLS tokens + Trainable Importance Estimator; reports ~1-epoch (~8 min) training and strong avg-improvement across 20 test datasets. Candidate to swap in for the plain linear probe.
  - `grip-unina/ClipBased-SyntheticImageDetection` — CLIP-feature detector, documented robust across DALL·E-3/MJv5/Firefly and post-processing, ships Docker + CSV batch tooling.
  - Effort (ICML 2025 Spotlight) — CLIP ViT-L/14 + SVD orthogonal-subspace fine-tuning, integrated in DeepfakeBench.
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

## 3. Stage 1 — Frequency-Domain Fast-Track

- **What:** lightweight frequency-artifact detector (DCT/FFT statistics; reference architectures: FreqNet, DFFreq, NPR).
- **Role:** narrow, one-directional fast-track for *clean, high-confidence synthetic* images only. Cannot independently return "real." Its score is also passed to Stage 2 as an auxiliary feature.
- **Why narrowed vs. a naive gate:** frequency detectors collapse toward 0–1.4% fake-accuracy under JPEG-50 while real-accuracy stays artificially high — i.e., degraded fakes systematically look "clean" to this signal. Allowing it to shortcut toward "real" would silently defeat the robustness goal.

## 4. Stage 2 — Main Classification Heads (all sit on the frozen CLIP backbone)

| Head | Purpose | Notes |
|---|---|---|
| Binary AIGC head | Core `P(ai_generated)` | Highest judging-weight signal; kept stable via frozen/lightly-finetuned shared features |
| 5-way classification head | `real_clean` / `real_edited` / `real_ai_enhanced` / `real_face_swapped` / `ai_generated` | Lightweight probe, not full joint backbone fine-tuning, to avoid multi-task interference |
| Coherence-ablation head | Semantic/lighting/perspective coherence signal, esp. for face-swap | Frozen or separately-trained auxiliary probe |
| Patch aggregation layer | Combines per-patch scores | Soft-probability averaging or attention pooling — replaces hard voting, which underperforms and suffers "Few-Patch Bias" (over-relying on a minority of patches) |
| Calibration layer | Temperature scaling | Per-class ECE, equal-mass binning; explicitly re-tested (not re-fit) on transformed data to catch distribution-shift calibration gaps |

## 5. Face Routing & Face-Specific Detection

### Face detector (lightweight, pretrained, routing only)
Options, ranked by footprint/accuracy trade-off:
| Model | Params | Notes |
|---|---|---|
| EResFD | ~92K | Most compact well-performing option (WIDER FACE overall mAP ~85.8%) |
| YuNet | ~76K | OpenCV Zoo, very compact, weaker on small faces |
| MTCNN | ~0.12M | ~4ms latency, 5-point landmarks, weaker on hard/small faces |
| RetinaFace (MobileNet-0.25) | ~0.42M | Standard lightweight choice, mAP ~85.6%, landmarks included |
| BlazeFace | ~1–2MB | Mobile-optimized (Google MediaPipe) |

Cost is zero for the majority of non-face images since the detector is a cheap routing gate, not run through the full cascade.

### Face-swap / deepfake specialist
- **Model zoo:** DeepfakeBench (`SCLBD/DeepfakeBench`) — 36 detectors with published weights, unified eval.
- **Preferred checkpoint:** **Effort** (CLIP-based, ICML 2025) over Xception, for better cross-domain generalization — Xception-style checkpoints are known to overfit dataset-specific artifacts (e.g., drop to ~60% on unseen FakeAVCeleb despite >90% on FF++).
- **Role:** used as an independent second-opinion specialist during the *self-training/pseudo-labeling loop* (see design.md) for face-containing images — not part of the shipped `<2B` inference path.

### AI-enhancement / retouch specialist
- **Model:** RetouchingFFHQ-trained head (Ying et al., ACM MM 2023).
- **Coverage:** faces only — smoothing, whitening, eye enlarging, face lifting, at controllable intensity levels. Dataset access-gated via request form (Megvii/Alibaba/Tencent-sourced retouching APIs).
- **Gap it does NOT cover:** non-face AI-upscaling/denoise/auto-enhance — no packaged detector exists in the field for this. Team must self-synthesize training data (Real-ESRGAN + an AI-denoiser run over a clean real-photo corpus).

## 6. Datasets

| Dataset | Role |
|---|---|
| `saberzl/SID_Set` (Hugging Face) | Synthetic + real source pool |
| CIFAKE (Kaggle, `birdy654/cifake-real-and-ai-generated-synthetic-images`) | Synthetic + real source pool (small, CIFAR-scale — no container metadata) |
| WildFake (ModelScope) | Synthetic + real source pool (translate via platform's translation button before use) |
| COCO val2017 (subset) | Non-AIGC portion of the **validation-only** benchmark (not for training) |
| DALL·E Advanced (subset) | AIGC portion of the **validation-only** benchmark (not for training) |
| FaceForensics++ / Celeb-DF | `real_face_swapped` sourcing — registration-gated, licensing must be confirmed early |
| RetouchingFFHQ | AI-retouch fine-tuning — access-gated via request form |
| GenImage / AIGCDetectBenchmark | Canonical external training/eval reference sets |
| Chameleon (`AIDE` repo companion, ~26K images) | Stress-test benchmark — human-Turing-test-passing AI fakes; expect ~70% accuracy even for good detectors |

## 7. Augmentation Pipeline (Training-Time)

Applied uniformly across all five classes, kept distinct from each class's *defining* transform so augmentation noise isn't confused with class signal:
- JPEG re-encode: quality 90/70/50/30
- Gaussian blur: σ 0.5/1.0/2.0
- Resize: 0.5×/0.25× then upscale
- Gaussian noise: σ 0.02/0.05/0.10
- Color jitter: brightness/contrast/saturation ±20%
- Center crop: 80%

Empirically the single highest-ROI item in the whole plan (reference literature reports ~91.4% avg precision cross-architecture from augmentation alone; ~98.5% on StyleGAN with blur+JPEG combined).

## 8. External / Commercial APIs (Offline-Comparison Baselines Only — Never in the Inference Path)

| API | Use | Free tier |
|---|---|---|
| AI or Not | Binary AI-vs-real baseline comparison | 20 image checks + 1M words text/month |
| Sightengine | `genai` model baseline (fully-synthetic + AI-edited) | 2,000 ops/month; AI/deepfake checks cost 5 ops each (~400 checks/mo) |
| Hive Moderation | AI-generated + deepfake classifier baseline | $50 free credits |
| Winston AI | Full forensics incl. EXIF/IPTC/C2PA parsing | 2,500 free credits (~8 free images) |
| Google SynthID | Watermark-only detector for Google/partner content | Waitlist; **not usable as a general detector** |

These exist purely to produce a comparison row in the demo/report — the product must remain fully local and offline for the actual deliverable.

## 9. Development Tooling (to be finalized by team)
- Standard ML stack: PyTorch, Hugging Face Transformers (for CLIP loading), scikit-learn (calibration/metrics), pandas (logging/tables).
- Environment: VSCode / Colab / Jupyter (per Devpost write-up requirement — must be disclosed in submission).
- Versioned checkpoint logging for the iterative/self-training loop.

## 10. Explicitly Rejected / Deprioritized
- **Running CLIP + ConvNeXt-Tiny simultaneously at inference** (v7 design) — dropped in v8 for a single-backbone design; halves inference-time model count and directly serves the local/memory-constrained requirement.
- **ViT-Small trained from scratch** — deprioritized; LAID (2025) findings and hackathon-scale data volume favor a frozen, pretrained CLIP backbone.
- **Hard-vote patch aggregation** — replaced by soft-probability averaging/attention pooling.