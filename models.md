# Step 3a Overview — Specialist-Model Disagreement Check

## Base model (unchanged, not a comparison target)

**Frozen CLIP ViT-L/14 (~304M params) + RINE-style multi-layer feature extraction.**
This is your primary model — reads CLS tokens from multiple CLIP transformer blocks (not just the final layer), combined via a small trainable importance estimator, feeding all four downstream heads (binary, 5-way, coherence, patch-aggregation). This is what Step 3a compares *against*, never a thing Step 3a replaces.

## Trigger condition

Specialist comparison only runs on **face-containing images**, determined by the RetinaFace (MobileNet-0.25) detector that already runs on every image as part of your v7/v8 face-region routing. No face → skip Step 3a entirely, fall through to Step 3's plain per-class confidence gating.

## Specialist models, per category

| Category | Specialist model(s) | Why this one | Params | When it runs |
|---|---|---|---|---|
| `real_face_swapped` | **DeepfakeBench — Xception** (default, every iteration) | Cheap, fast, standard baseline across the self-training loop | ~22M | Every face-containing image, every iteration |
| `real_face_swapped` (tie-breaker only) | **DeepfakeBench — Effort** (CLIP-based) | Better cross-dataset generalization than Xception; reserved for disputed cases only, since it's a full CLIP ViT-L/14 forward pass and expensive to run on the whole pool every iteration | ~304M | Only on images where Xception already disagreed with the cascade |
| `real_ai_enhanced` | **RetouchingFFHQ-trained head** | Only existing open precedent for AI-enhancement detection — but face-retouch-specific (beauty/smoothing/whitening/eye-enlarging/face-lifting), not general upscaling/denoise | Small (probe/MAM module on a CNN backbone, tens of M at most) | Only face-containing images where the cascade predicts `real_ai_enhanced` |
| `ai_generated` | *(none — see below)* | Comparing your own CLIP+RINE model to itself isn't an independent signal | — | Not applicable |
| `real_edited` | *(none available)* | No open specialist model found in research | — | Falls back to Step 3 confidence gating only |
| `real_clean` | *(none — default class)* | Nothing to specialize against | — | Falls back to Step 3 confidence gating only |

## Model loading strategy

- CLIP+RINE base model: loaded once, resident for the entire session (runs on every image regardless).
- RetinaFace face detector: loaded once, negligible cost, runs on every image (already budgeted in v7/v8).
- DeepfakeBench-Xception: loaded once at the start of each self-training iteration's pseudo-labeling pass (Step 3); run conditionally per-image based on face-presence.
- DeepfakeBench-Effort: loaded once per iteration, but only *invoked* on the smaller disagreement subset — not run on every face-containing image.
- RetouchingFFHQ head: loaded once per iteration, run conditionally on face + `real_ai_enhanced` prediction.
- All specialists can be released from memory between self-training iterations if memory-constrained, or kept resident for the run — loading cost is paid once regardless, forward-pass cost is what's conditional.

## Routing logic

```
Stage 2 output (pred, category, category_confidence)
              |
        face detected? (RetinaFace, already run on every image)
        |
        |-- NO  -> No-specialist path: Step 3 confidence gating only
        |          [covers: real_edited, real_clean,
        |           and real_ai_enhanced predictions with no face]
        |
        |-- YES -> run DeepfakeBench-Xception
                        |
                agree with cascade's category?
                |-- YES -> stronger pseudo-label (raises effective
                |           confidence for Step 3's threshold)
                |-- NO  -> run DeepfakeBench-Effort as tie-breaker
                                |
                        still disagree? -> disagreement flag = true
                                            -> human review, regardless
                                               of individual confidence

           [in parallel, if category == real_ai_enhanced]
           -> run RetouchingFFHQ head
                   |
           agree with cascade? -> same accept/flag logic as above
```

## Coverage summary (the honest asymmetry)

- **Full specialist coverage:** `real_face_swapped`
- **Partial coverage (faces only):** `real_ai_enhanced`
- **No coverage:** `real_edited`, `real_clean`, non-face `real_ai_enhanced`, `ai_generated` (self-comparison excluded as non-independent)

State this explicitly in the writeup — it reflects what specialist models actually exist in the field (per research), not a gap in your own design effort.

## Logging (extends Step 7's review-queue CSV)

Add two columns: `specialist_verdict` (empty if no specialist ran) and `specialist_disagreement` (boolean, empty if no specialist ran). Enables reporting specialist-agreement rate as an Error Analysis / Table 1 metric.

## Scope boundary

Everything in Step 3a runs during the **self-training loop only** — never part of the shipped inference script that processes a judge's test directory. Does not affect the <2B parameter constraint on the deliverable's inference path. State this distinction explicitly to avoid it reading as scope creep against the hackathon's compute/local-only requirements.




## Evaluation Design

Four tables make the robustness story explicit and auditable:

1. **Transform robustness** — clean vs. each of the six transforms, split into **R.Acc.** (real-labeled accuracy) and **F.Acc.** (fake accuracy) per transform, plus a frozen-CLIP baseline row for comparison. This split exists specifically so a known field-wide failure mode — detectors collapsing toward predicting "real" under degradation rather than failing randomly — is directly visible instead of hidden inside one aggregate number.
2. **Generator generalization** — held-out generator family, run only after the JPEG/resolution-matching step, so the result reflects genuine generalization rather than a matching artifact.
3. **5-way confusion matrix** — where real-subclass errors concentrate (e.g., is `real_ai_enhanced` being confused with `ai_generated`).
4. **Calibration (ECE)** — reported on both clean and transformed validation data *without re-fitting* the temperature parameter between the two, so any distribution-shift calibration gap becomes a visible, reportable finding rather than an unnoticed flaw.

## Offline Improvement Loop (Self-Training)

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
