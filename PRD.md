# PRD — Robust Detection of AI-Generated Images Under Real-World Transformations

## 1. Background & Problem Statement

Generative AI makes it trivial to produce highly realistic synthetic images at scale, creating risk for online platforms: misinformation, impersonation, fraud, and eroded trust in digital content. Detection is hardest in exactly the conditions that matter most — after images are compressed, cropped, reposted, or lightly edited. Lab-only accuracy on clean images is not a useful proxy for real-world performance.

**Goal:** build a prototype that distinguishes AI-generated images from authentic ones with strong robustness under realistic post-processing and redistribution, not just on clean data.

## 2. Scope

**In scope:**
- Image-level AIGC (AI-generated content) detection
- Robustness to common image transformations (JPEG compression, blur, resize, noise, color jitter, cropping)
- Feature engineering, model design, evaluation design, error analysis, explainability

**Out of scope:**
- Full production deployment / platform-wide moderation systems
- Non-image modalities (video, audio)
- Adversarial-perturbation robustness, pixel-mask segmentation beyond face-region routing (time-permitting only)

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
- Secondary output (bonus, not required by spec): a **5-way category** classification, richer than binary real/fake:

| Category | Meaning |
|---|---|
| `real` (clean) | Untouched camera photo |
| `real_edited` | Real photo, traditionally edited (HDR, Photoshop, filters) |
| `real_ai_enhanced` | Real photo, AI-touched (upscale, beauty filter, auto-enhance) |
| `real_face_swapped` | Real photo, face swapped/deepfaked |
| `ai_generated` | Fully AI-generated, no real source image |

- `real_face_swapped` does **not** count toward `pred` — it is surfaced only via `category` / `category_confidence`, keeping `pred` semantically clean.

### 4.2 Robustness Requirement
Must maintain reasonable accuracy after realistic post-processing:

| Transform | Parameters | Real-World Analog |
|---|---|---|
| JPEG Compression | quality 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian Blur | σ = 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | 0.5×/0.25× then upscale | Thumbnail generation |
| Gaussian Noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color Jitter | brightness/contrast/sat ±20% | Filter apps, auto-enhance |
| Center Crop | crop 80% | Profile-picture cropping |

The product must not silently fail by defaulting to "real" when a transform destroys its detection signal — this failure mode must be measured and reported, not hidden.

### 4.3 Deliverable JSON Contract
```
{"image_path": "img001.jpg", "pred": 0.12, "category": "real_face_swapped", "category_confidence": 0.77}
```
Only `image_path` and `pred` are strictly required by the challenge spec; `category` / `category_confidence` are bonus fields. Internal-only diagnostic fields (e.g. `verdict_source`, degradation-bias metrics) must **not** appear in the public JSON.

### 4.4 Evaluation & Reporting Requirements
- **Robustness Evaluation Summary** — clean vs. transformed performance, split by real-accuracy (R.Acc.) and fake-accuracy (F.Acc.) per transform, so any "predicts real under degradation" bias is directly visible rather than hidden in an aggregate number.
- **Generator generalization table** — held-out generator family, evaluated on a JPEG/resolution-matched dataset so the eval measures genuine generalization, not compression/resolution bias.
- **5-way confusion matrix** for real-subclass discrimination.
- **Calibration table** — Expected Calibration Error (ECE) per class, on both clean and transformed validation data.
- **Error Analysis Note** — representative false positives/negatives and stated trade-offs.

### 4.5 Non-Functional Requirements
- Local, memory- and space-efficient: minimize simultaneously-loaded models at inference time.
- Deliberate, explainable architectural decisions over maximal complexity.
- Iterative improvement mechanism (self-training loop) demonstrated over multiple cycles with measurable results, not just a one-shot model.

## 5. Deliverables (per challenge brief)
1. **Written project description** (via Devpost) — approach, tools, models/APIs, libraries, datasets used.
2. **Public code repository** — well-structured, commented code; an inference script that takes an image directory and outputs a JSON file with `image_path` and `pred` per image; a README with setup, reproduction steps, limitations/future work, and contributions.
3. **Demo video** — end-to-end demo, public on YouTube, linked in Devpost, no third-party trademarked/copyrighted content.
4. **Robustness Evaluation Summary** (table/visual, clean vs. transformed).
5. **Error Analysis Note** (false positives/negatives, trade-offs).

## 6. Judging Criteria & Weights

| Criteria | Weight | What it rewards |
|---|---|---|
| Technical Execution | 35% | Well-structured code, thoughtful architecture, reliable demo, deliberate complexity |
| Innovation & Problem Insight | 20% | Original framing, sharp understanding of *why* the problem is hard |
| Impact & Relevance | 20% | Real-world value beyond the hackathon prompt |
| Feasibility & Practicality | 15% | Realistic, buildable, resource-proportionate |
| Presentation & Communication | 10% | Clear storytelling, ability to field questions |

## 7. Open Product Questions
- Whether to expose `verdict_source` (c2pa / frequency / model) as a visible Table 1 breakdown for judges, even though it's not part of the required JSON.
- How to demonstrate the C2PA early-exit mechanism given that none of the reference training datasets are known to carry intact C2PA manifests (requires a small self-synthesized demo set).
- How much of the self-training / specialist-disagreement machinery to actually build out live vs. describe as a designed-but-time-boxed mechanism, given hackathon time constraints.

## 8. Known Gaps / Risks Called Out for Follow-up
- **Error analysis mechanics**: need a concrete script to pull top-10 false positives/negatives; decide whether this is reflected in the UI/demo and how human verification confirms "true" detection rate.
- **Error handling**: no defined baseline yet for minimum image resolution or corruption detection (garbage-in handling).
- **`real_ai_enhanced` (AI-enhancement) class**: confirmed as a genuine open gap in the field — no general packaged detector exists, especially for non-face upscaling/denoise. This is the project's highest-novelty and highest-risk deliverable.
- **Face-swap dataset licensing** (FaceForensics++, Celeb-DF) requires registration — flagged as the most likely Day-1 external bottleneck.