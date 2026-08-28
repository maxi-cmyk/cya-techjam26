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

See [PRD.md](PRD.md) for requirements, [design.md](design.md) for the pipeline, [models.md](models.md) for the model/evaluation plan, [training.md](training.md) for training and fine-tuning, and [techStack.md](techStack.md) for implementation choices.
