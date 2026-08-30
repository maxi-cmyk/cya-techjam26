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

See [PRD.md](docs/product/PRD.md) for requirements, [design.md](docs/architecture/design.md) for the pipeline, [models.md](docs/architecture/models.md) for the model/evaluation plan, [training.md](docs/training/training.md) for training and fine-tuning, and [techStack.md](docs/architecture/techStack.md) for implementation choices.

## Colab execution

Run `notebooks/00_colab_setup.ipynb`, then `01_task2_data_contract.ipynb`, and finally `02_stage_a_clip.ipynb`. The Stage A notebook uses the resolved CLIP commit in every embedding-cache key, trains only the binary head, compares both Task 2 matching policies over seeds 42/43/44, and keeps `final_test` locked. Clean reports are available immediately; the locked 50/50 score and robustness checkpoints remain unavailable until Task 3 supplies the independent transform cells.

Task 8B uses a separate licensed native-camera/synthetic manifest under the same
`hackathon_data` and artifact roots. It does not replace SID_Set or retrain the
existing backbone. See [Task 8B dataset](docs/data/task8b_dataset.md) for the
verified sources, non-commercial GenImage assumption, storage layout, inventory
schema, grouped-split rules, and manual Drive staging boundary.

The completed local Task 8B pilot normalizes 1,164 eligible rows to identical
256 px uncompressed TIFF views and passes the nuisance gate at 0.50 balanced
accuracy. PRNU fails its independent device-signal gate (AUC 0.538; minimum
0.60), CA lacks calibration coverage, and the recorded outcome is no physical
feature retained and no RINE fusion training.
