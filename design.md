# Design — System Architecture & How It Works

This document describes how the system works end-to-end at a high level: the inference-time detection cascade, and the separate offline self-training loop that improves the model over time. For *what* each component is built from, see `techStack.md`. For *why* it must exist and what it must deliver, see `PRD.md`.

---

## 1. Design Philosophy

Two principles run through every architectural decision:

1. **One-directional shortcuts only.** Any fast-path or early-exit stage may shortcut *toward* `ai_generated`, never toward "real." This is the fix for the plan's biggest historical bug (v6): early stages relying on signals that transforms (JPEG, blur, resize) systematically destroy — meaning a degraded fake would look "clean" to a weak stage and exit misclassified as real. By making every shortcut one-directional, a degraded or ambiguous image always falls through to the strongest available signal instead of silently passing as authentic.
2. **Single live backbone at inference.** Multiple backbones may be *compared offline*, but only one is ever loaded when actually scoring a judge's test directory — this keeps the deliverable local, memory-light, and fast to load, per the resource constraints in the PRD.

---

## 2. Inference-Time Architecture (the shipped pipeline)

```
Input Image
     │
     ▼
Stage 0 — C2PA Provenance Check
(parse + verify manifest, metadata only, no model)
     │
     ├── Valid manifest, explicitly claims AI-gen ──► Return ai_generated (audited on a 10% sample)
     │
     └── No manifest / stripped / invalid signature / claims "real" (untrusted)
                     │
                     ▼
          Stage 1 — Frequency-Domain Fast-Track
          (flags CLEAN, high-confidence synthetic only)
                     │
     ┌───────────────┴────────────────────┐
     │                                     │
Confident SYNTHETIC,               Everything else: uncertain,
clean image only                   any suspected degradation,
     │                              or "real"-leaning
     ▼                                     ▼
Return ai_generated              Stage 2 — Frozen CLIP-ViT Backbone
(audited)                        + soft-probability patch aggregation
                                  + face-region detector head (routing only)
                                  + coherence-ablation head (frozen)
                                  + 5-way classification head
                                  + calibration (temperature scaling)
                                             │
                                             ▼
                              pred + category + category_confidence
                              (+ internal-only: verdict_source,
                                 degradation-bias diagnostic)
```

### How each stage works

**Stage 0 — Provenance (C2PA).** A library call, not a model. If an image carries a cryptographically valid manifest that explicitly names an AI generation tool, the system trusts that signed claim and exits immediately with a high-confidence `ai_generated` verdict. Every other outcome — no manifest, a manifest that fails signature verification, or a manifest that claims the image is real — is treated as *uninformative*, not as evidence of authenticity, and falls through to Stage 1. This mirrors how C2PA works in the real world: presence is strong evidence, absence proves nothing, because manifests are routinely stripped by re-encoding and platform uploads.

**Stage 1 — Frequency fast-track.** A cheap frequency-domain check (DCT/FFT-based) that recognizes the unmistakable spectral fingerprints left by many generators on *clean* images. It is deliberately narrow: it may only fast-track a confident synthetic verdict on images with no sign of degradation. It is never allowed to issue a "real" verdict on its own, because the same transforms this project must survive (JPEG, blur, resize) are exactly what erase frequency artifacts — a degraded fake would otherwise look clean here and slip through as real. Its raw score is still forwarded into Stage 2 as one input feature, so no information is wasted even when it doesn't get to make the final call.

**Stage 2 — Main model.** Everything that reaches this point (the large majority of real-world images, especially anything transformed) goes through the single frozen CLIP-ViT backbone. On top of the shared frozen features sit several lightweight heads:
- A **binary head** producing the core `P(ai_generated)` score.
- A **5-way head** that further distinguishes untouched, traditionally-edited, AI-enhanced, face-swapped, and fully-synthetic images — trained as a lightweight probe rather than fine-tuning the whole backbone, so it can't destabilize the primary binary signal.
- A **face-region detector**, which only activates when a face is present in the image; it routes patch attention toward face-boundary regions (blending seams, resolution mismatches, lighting inconsistencies) that matter specifically for face-swap detection. Cost is effectively zero for the majority of non-face images since the check is skipped entirely.
- A **coherence-ablation head**, a frozen auxiliary probe that checks global scene coherence (does the lighting/perspective on a face match the rest of the scene) — a plausible extra signal specifically for face-swap detection.
- **Patch aggregation** by soft-probability averaging (or attention pooling) rather than hard voting, so the verdict reflects a distribution across the image instead of over-weighting a handful of patches.
- **Calibration** via temperature scaling, so the final confidence score is meaningful, not just the argmax.

**Output.** The system emits `pred` (`P(ai_generated)`, fully-synthetic-only), plus bonus fields `category` and `category_confidence`. Internal diagnostics (which stage actually produced the verdict, degradation-bias flags) are logged for evaluation but never leak into the public deliverable JSON.

### Why C2PA sits before the frequency screen, not instead of it
They answer different questions and stack rather than compete: C2PA is *provenance* evidence (what a tool claims about how the image was made, if the tag survives), while the frequency screen is *forensic* evidence (artifacts left behind by generation itself, independent of any claim). An image can have no C2PA tag at all — most training-dataset images will — and still be an obvious synthetic image the frequency screen catches. Neither stage ever gets to shortcut toward "real."

### Why only one backbone is loaded live
Running two full models (a cascade backbone plus a "safety net" backbone) at inference for every image was identified as the single biggest resource risk given the local, `<2B`-parameter, limited-compute constraint. The insight that resolved this: knowing *which* backbone is more robust only requires a one-time **offline** comparison (produces a static table), not a live ensemble at inference. ConvNeXt-Tiny is still trained once, entirely offline, purely to produce that comparison table for the write-up — it is never loaded when scoring a judge's test directory. If it had won the offline comparison, it would have become the single shipped backbone instead of CLIP; the architecture is designed to be backbone-agnostic in this way.

---

## 3. Data Assembly & Leakage Controls

Five classes are assembled with two guardrails layered on top of standard dataset construction:

- **Disjointness:** the same source real image can never appear in more than one class (e.g., can't be both `real_clean` and `real_edited`).
- **JPEG/resolution matching:** before any generator-family holdout split is drawn, a common JPEG re-encode pass and common resize/crop pipeline is applied across *all* classes. Without this, an evaluation claiming "cross-generator generalization" can actually just be measuring an artifact of mismatched compression/resolution between the real and synthetic pools — a documented and easy-to-miss inflation of reported accuracy.
- **Construction-pipeline diversification:** the three constructed classes (`real_edited`, `real_ai_enhanced`, `real_face_swapped`) are built with varied tools/parameters rather than one fixed pipeline, and at least one variant is held out from training. This stops the model from learning to recognize one specific tool's fingerprint (e.g., one Real-ESRGAN setting) instead of the underlying concept of "AI-enhanced."

## 4. Evaluation Design

Four tables make the robustness story explicit and auditable:

1. **Transform robustness** — clean vs. each of the six transforms, split into **R.Acc.** (real-labeled accuracy) and **F.Acc.** (fake accuracy) per transform, plus a frozen-CLIP baseline row for comparison. This split exists specifically so a known field-wide failure mode — detectors collapsing toward predicting "real" under degradation rather than failing randomly — is directly visible instead of hidden inside one aggregate number.
2. **Generator generalization** — held-out generator family, run only after the JPEG/resolution-matching step, so the result reflects genuine generalization rather than a matching artifact.
3. **5-way confusion matrix** — where real-subclass errors concentrate (e.g., is `real_ai_enhanced` being confused with `ai_generated`).
4. **Calibration (ECE)** — reported on both clean and transformed validation data *without re-fitting* the temperature parameter between the two, so any distribution-shift calibration gap becomes a visible, reportable finding rather than an unnoticed flaw.

## 5. Offline Improvement Loop (Self-Training)

Separate from the shipped inference pipeline, an iterative pseudo-labeling loop improves the model between the seed dataset and however much time remains in the hackathon. This entire loop runs offline / at training time — it never touches the inference path.

```
Seed labeled set (~60%) ──► Train seed model (iteration 0)
                                    │
                                    ▼
Held-out "unlabeled" pool (~25%, labels hidden)
                                    │
                        Run current model → get pred + category + confidence
                                    │
                    ┌───────────────┼────────────────────────┐
                    ▼               ▼                        ▼
            High confidence   Low confidence          Face detected?
            (per-class          │                            │
             threshold)         ▼                    ┌───────┴───────┐
                    │      Human review queue         ▼               ▼
                    │      (CSV log + minimal      Run specialist   No specialist
                    │       viewer)                 (DeepfakeBench /  → normal
                    │           │                    RetouchingFFHQ)   routing
                    ▼           ▼                        │
          Accept as pseudo-  Reviewed examples      Agree → stronger
          label (down-       → seed set              pseudo-label
          weighted 0.5–0.7x)  (full trust)           Disagree → forced
                    │                                  human review
                    └───────────────┬────────────────────────┘
                                    ▼
                    Retrain / fine-tune from previous checkpoint
                    (re-run full augmentation on new examples too)
                                    │
                                    ▼
                    Evaluate ONLY on locked validation set (~15%,
                    never pseudo-labeled, never touched otherwise)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
          Improved / held steady            Regressed on any class
          → becomes new current model         by more than tolerance
          → return to pseudo-labeling         → ROLL BACK to previous
                                                 checkpoint, log why
```

### Why three pools, not two
The held-out "unlabeled" pool secretly retains its ground-truth labels (hidden from the training loop only) specifically so pseudo-label accuracy can be measured at every iteration — this turns the loop from a black box into something with a reportable, per-iteration accuracy metric. The locked validation set is the one pool self-training is never allowed to touch in any way, which is what makes every other reported number trustworthy rather than circular.

### Why confidence gating is per-class, not global
Classes are imbalanced and some (`real_ai_enhanced`, `real_face_swapped`) are inherently harder and more prone to overconfidence. A single global threshold would let the model accept low-quality pseudo-labels on its weakest classes while being needlessly conservative on its strongest ones. Per-class thresholds, started conservative (e.g., 90th percentile of that class's own confidence distribution), address this directly.

### Why a specialist-disagreement check exists for face-containing images
Confidence alone can't catch a model that is *confidently wrong*. Where an independent specialist model exists — DeepfakeBench (Effort checkpoint) for face-swaps, a RetouchingFFHQ-trained head for facial AI-retouching — that specialist's verdict is compared against the cascade's. Agreement strengthens the pseudo-label; disagreement forces human review regardless of either model's individual confidence, since disagreement between two independent signals is a stronger "this is a hard example" signal than low confidence alone. This check only runs during the offline self-training loop, never in the shipped `<2B` inference path, and only for face-containing images — non-face classes fall back to plain per-class confidence gating, which is stated explicitly as a deliberate asymmetry rather than an oversight.

### Why pseudo-labels are down-weighted and rollback is automatic
A wrong pseudo-label can drag a model further off course than it corrects. Down-weighting pseudo-labeled loss contributions (0.5–0.7×) relative to true-labeled ones limits that risk, and the rollback rule — discard the iteration and revert to the previous checkpoint if any class regresses beyond a small tolerance on the locked validation set — is the backstop that prevents confirmation-bias drift from silently compounding across iterations.

### Retraining trigger
Two independent proxy signals, since either alone is easy to game unintentionally:
1. **Confidence drift** on the team's own uploaded test images (a rising rate of low-confidence predictions suggests the model is seeing something outside its training distribution).
2. **Scheduled locked-validation re-checks** every N pseudo-labeling batches, which catch slow, confidently-wrong degradation that confidence drift alone might miss.

Either signal firing triggers another loop cycle; every trigger event and its outcome is logged as evidence of a genuinely iterative system.

## 6. Known Design Gaps (carried forward for follow-up)
- No defined minimum-resolution or corruption-detection baseline yet for malformed/garbage input images.
- Error-analysis tooling (a script to pull the top-10 false positives/negatives per class) is specified as a requirement but not yet built.
- Whether `verdict_source` should be surfaced as a visible breakdown in the demo/report (recommended, low-effort, not yet decided as final).