# Robust AI-Generated Image Detection

This project classifies images as either **fully authentic** or **fully AI-generated** and measures how well that decision survives common image transformations.

## Evaluation Boundaries

- Only fully authentic and fully AI-generated source images are included.
- Mixed-origin, AI-edited, face-swapped, composited, and partial-AI images are excluded.
- Each robustness case applies exactly one transformation to the clean source.
- Transformations are never chained, mixed, or overlaid.
- The final score is weighted **50% clean accuracy and 50% robustness**.
- PRNU coherence is tested as an auxiliary physical-capture feature, never as a standalone authenticity gate; DSNU is currently deprioritized.
- A learned texture-aware head preserves selected local details, but smoothness, edge density, and OCR confidence are never fixed AI rules.
- Deterministic RGB/Lab correlation and optical-aberration features run inline with confidence masking; absent lens/camera artifacts are neutral.
- Frequency features are evaluated by generator/decoder family; Stage 1's synthetic fast-track stays disabled until held-out precision and robustness justify it.
- Immutable originals are retained for C2PA and native-forensics experiments; the primary model uses label-independent matched JPEG derivatives so compression history cannot become a label shortcut.
- Dataset-level matched re-encoding and training-time JPEG augmentation are separate controls: the former removes encoding-history bias, while the latter teaches degradation robustness.

## JPEG Robustness Strategy

JPEG can erase high-frequency generation artifacts and can also create a dataset shortcut when authentic and synthetic images have different encoding histories. The project addresses these as separate problems:

1. **Matched dataset preparation:** retain immutable originals, then create the primary clean view by re-encoding both labels with the same JPEG-quality distribution, encoder, and settings.
2. **JPEG-aware training:** create independent quality 90/70/50/30 variants from the matched clean parent to teach robustness to platform-style re-encoding.
3. **Representation-level backbone:** use frozen CLIP-ViT as the principal signal and treat frequency, texture, PRNU, color, and optics as auxiliary evidence subject to JPEG ablation.
4. **Bias auditing:** test whether format, resolution, file size, estimated JPEG quality, quantization tables, or feature validity predict the label before and after matching.

C2PA runs on immutable source bytes during dataset construction and on the exact received bytes at inference. Native-image forensic features remain experimental offline ablations; the shipped visual pipeline processes only the received view, and no compression artifact is proof of authenticity or synthesis. Matched JPEG normalization is never rerun at inference.

## Resize Robustness Strategy

The resize benchmark is one compound downsample-and-restore operation, evaluated independently at 0.5x and 0.25x severity. Both steps use bilinear interpolation with pinned library, antialiasing, rounding, color, and dtype settings; the restored output retains the parent dimensions and is cached losslessly so JPEG is not added accidentally.

At inference, the detector scores the received image once and does not generate extra resized, compressed, or blurred variants. Stage 2 combines a global CLIP view with multiple detail-rich crops selected from the received view before global model-size conversion. Resize-aware training uses identical settings for both labels, while evaluation explicitly checks whether interpolation artifacts increase authentic false positives.

See [PRD.md](PRD.md) for requirements, [design.md](design.md) for the pipeline, [models.md](models.md) for the model/evaluation plan, [training.md](training.md) for training and fine-tuning, and [techStack.md](techStack.md) for implementation choices.
