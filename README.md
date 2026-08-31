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
feature retained and no RINE fusion training. A bounded PRNU v2 estimator is
available as `make task8b-v2-prnu-validate`; its label-free device test reaches
AUC 0.859 and top-1 accuracy 0.657 versus 0.10 random at the binary-compatible
256 px crop. The reference-free PRNU-only diagnostic scores 78.09% locked, but
RINE+PRNU is rejected after severe seed instability (33.43% mean versus 99.81%
for controlled RINE). The experiment preserves the original evidence, writes
only under `artifacts/task8b_v2`, and keeps the competition `final_test` sealed.

## Task 9 result

The texture-aware local-detail path (Task 9) passed its clean gate (100%
`global_local` clean accuracy versus 99.394% for the internal `global_only`
ablation) but was **rejected** at the frozen-checkpoint Stage-1 robustness
screen: `global_local` tied controlled RINE on clean accuracy but scored
93.13% mean robustness accuracy versus 99.80% for controlled RINE, with both
locked-score deltas negative and multiple worst-cell failures — most severely
`resize_scale_0.25`, where AI-generated accuracy collapsed to 50.2%
(`global_local`) and 18.4% (`local_only`, near-random). The local patch
signal is fragile under aggressive downsampling/blur/compression. **Controlled
RINE (Task 6) is the sole retained model** going into Task 10; no texture,
frequency, Lab, or PRNU fusion candidate is retained.

## Task 10 — Inference CLI, calibration, and packaging

Task 10 is split into two independently scoped pieces so the CLI skeleton
does not have to wait on final model selection, and calibration/`final_test`
does not have to wait on CLI plumbing.

### Task 10A — Inference skeleton (in progress)

Everything needed to run a directory of images through *some*
`predict_probability` function and publish a spec-compliant result, with no
opinion about which model that function calls:

- Recursive, deterministic image discovery (`.jpg`/`.jpeg`/`.png`/`.webp`/
  `.tif`/`.tiff`, case-insensitive, symlinks never followed, NFC-normalized
  relative-path ordering, empty discovery and path collisions are fatal).
- Decode/validate/normalize each image to RGB, with an explicit
  decompression-bomb guard and five stable, path-sanitized error codes
  (`file_unreadable`, `decode_failed`, `unsupported_image`,
  `invalid_dimensions`, `decompression_bomb`) rather than leaking exception
  text.
- Stage 0 C2PA check (`has_verified_ai_generation_claim`): true only for a
  verified, active `c2pa.created`/`trainedAlgorithmicMedia` claim; every other
  outcome (missing dependency, no manifest, invalid signature, malformed
  claim, authenticity-only claim, parser failure) returns `False` and falls
  through to the predictor. No network access, no registry, ever.
  This is the one place that deliberately catches everything internally —
  everywhere else, an uncatalogued exception is a fatal run failure.
- `predict_probability(image) -> float` is injected (dependency injection,
  not a registry) — Task 10A ships a trivial stub (fixed constant) as the
  default so `scripts/run_inference.py <image_dir> --output-dir <dir>` is
  runnable end-to-end today; Task 10B swaps in the real controlled-RINE
  adapter as one function.
- Publishes `predictions.json` (public `{"image_path", "pred"}` contract) and
  `report.json` (`schema_version`, `summary`, `errors[]`) to `--output-dir`
  only on non-fatal completion — both are written to temp siblings and
  renamed into place (`report.json` first, `predictions.json` last, so its
  presence is the strongest "this run is real and complete" signal), and a
  fatal run never touches a prior successful run's output. Exit codes: `0`
  full success, `1` fatal, `2` argparse usage error, `3` partial success.
  One stdout line per processed image plus a final summary line.

Design lives at
`docs/superpowers/specs/2026-08-31-task-10a-inference-skeleton-design.md`
(in progress) on the `task10a-inference-skeleton` branch/worktree.

Implementation is split for parallel work:

- **Shared, first:** `src/cya_detector/inference/contracts.py` — pure types
  (`ValidationError`, `PredictionRecord`, `RunSummary`, the `Predictor`
  protocol, exit-code constants). Both streams below depend on it immediately.
- **Stream A — input & trust boundary:** `src/cya_detector/inference/inputs.py`
  (recursive discovery, decode/validate/normalize, decompression-bomb guard)
  and `src/cya_detector/inference/c2pa.py` (Stage 0 verified-claim check), with
  `tests/test_c2pa_inference.py`.
- **Stream B — orchestration & output:** `src/cya_detector/inference/runner.py`,
  `src/cya_detector/inference/output.py`, `src/cya_detector/inference/cli.py`,
  and `scripts/run_inference.py`, with `tests/test_inference_cli.py`. Stream B
  can proceed immediately against a fake predictor and fake C2PA function via
  dependency injection — it does not need Stream A's real implementations to
  make progress.
- **Integration, last:** `tests/test_directory_inference.py` exercises the
  full pipeline end-to-end; write it once both streams land, or against the
  other stream's real code once merged.

### Task 10B — Model selection, calibration, and final_test (not started)

Everything Task 10A deliberately defers, blocked on Task 10A's `Predictor`
protocol landing:

- Select the final retained checkpoint — controlled RINE is the only
  surviving candidate after Task 9's rejection; no further architecture
  search is expected.
- Fit one temperature on clean `selection_val` logits and reuse it unchanged
  everywhere; keep the binary threshold fixed unless the challenge specifies
  otherwise.
- Write the real `predict_probability` adapter around the selected,
  calibrated checkpoint and wire it into `scripts/run_inference.py` in place
  of Task 10A's stub.
- Measure final model size, latency, peak memory, and disk/cache
  requirements on the target machine.
- Run the sealed `final_test` exactly once, after weights, calibration,
  threshold, and feature flags are frozen; generate robustness, ablation,
  generalization, and error-analysis artifacts.
- Any optional post-baseline experiment (fine-tuning, self-training,
  external data) is separately versioned and must clear the same score/
  per-label/held-out-source/resource gates or be rolled back.
