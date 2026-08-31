# Robust AI-Generated Image Detection

## Project overview

This project classifies images as either **fully authentic** or **fully
AI-generated** and measures how well that decision survives common image
transformations (JPEG re-compression, blur, and resize). The scored decision
is weighted 50% clean accuracy and 50% robustness across independent
transform cells.

The retained model is **controlled RINE**: a frozen-CLIP intermediate-layer
representation with a lightweight trained binary head, retrained under a
balanced clean-or-one-transform sampler. It reached 100.00% clean accuracy,
99.62% mean robustness accuracy, and a 99.81% locked 50/50 score across
seeds 42/43/44 (seed 42 individually: 99.85%, the highest of the three — see
[Steps to reproduce your results](#steps-to-reproduce-your-results)).
Frequency, color/Lab, PRNU, and a texture-aware local-patch path were all
built and evaluated as candidate additions and were all **rejected** after
failing the locked-score or robustness gate — see
[Experimental history and rejected candidates](#experimental-history-and-rejected-candidates)
below for why each one failed and what that implies.

**Submission entry point:** [`run_inference.py`](run_inference.py) at the
repository root:

```
python run_inference.py <image_dir> --output-dir <output_dir> --checkpoint artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt
```

It takes a directory of images and writes a `predictions.json` file of
`{"image_path", "pred"}` confidence scores (0 = authentic, 1 = AI-generated),
plus a `report.json` validation summary, using the real controlled-RINE
seed-42 checkpoint. See
[Steps to reproduce your results](#steps-to-reproduce-your-results) for full
commands, including how to restore the checkpoint. Everything else under
`scripts/` is internal pipeline tooling (dataset construction, training,
evaluation, robustness testing across Tasks 1–10) — not something a grader
needs to run.

**Current status:** the submission entry point is fully implemented, tested,
and wired to the real, selected model (controlled RINE, seed 42) — see
[Limitations and what we'd improve with more time](#limitations-and-what-wed-improve-with-more-time)
for what's still outstanding (resource profiling on target hardware, the
sealed `final_test` run).

## Setup and installation instructions

Requires Python >= 3.10.

```bash
git clone <this repository>
cd cya-techjam26
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

(`make install` runs the same two commands. Use `make install-dev` instead
if you also want `pytest`/`ruff` for running the test suite, or
`make install-colab` on a Google Colab runtime, which pins a
Colab-compatible `torch` build.)

Verify the install:

```bash
make smoke          # checks config, imports, accelerator; requires all deps
make smoke-bootstrap # same, but tolerates missing optional deps (e.g. c2pa-python)
make test            # full test suite, ~300+ tests, no GPU or dataset required
```

## Steps to reproduce your results

### Running the submission script

```bash
python run_inference.py <path-to-image-directory> --output-dir <output-directory>
```

This recursively discovers `.jpg`/`.jpeg`/`.png`/`.webp`/`.tif`/`.tiff` files
under `<path-to-image-directory>` (case-insensitive, symlinks never
followed), runs each one through a C2PA verified-AI-generation-claim check
followed by the predictor, and writes:

- `<output-directory>/predictions.json` — `[{"image_path": "...", "pred": 0.0-1.0}, ...]`
- `<output-directory>/report.json` — `{"schema_version", "summary": {"discovered", "predicted", "invalid"}, "errors": [...]}`

Exit codes: `0` full success, `1` fatal (nothing published — e.g. an empty
input directory), `2` bad arguments, `3` partial success (some images were
invalid; `predictions.json` still contains every image that scored
successfully, and `report.json` explains why the rest didn't).

A minimal check that it works:

```bash
mkdir -p /tmp/demo_images /tmp/demo_output
python -c "from PIL import Image; Image.new('RGB', (64, 64)).save('/tmp/demo_images/sample.png')"
python run_inference.py /tmp/demo_images --output-dir /tmp/demo_output
cat /tmp/demo_output/predictions.json
```

### Reproducing the retained model's clean/robustness numbers

The 99.81% locked-score result for controlled RINE and the per-seed scores
(42: 99.85%, 43: 99.81%, 44: 99.78%) come from the Task 6 / post-Task-3
robustness notebook run recorded in
[`docs/planning/nextSteps.md`](docs/planning/nextSteps.md). Reproducing it
from scratch requires the full dataset pipeline (Tasks 1–3) and is run on
Google Colab:

1. `notebooks/00_colab_setup.ipynb` — environment setup, mounts the shared
   dataset.
2. `notebooks/01_task2_data_contract.ipynb` — builds the matched-clean
   fixed-Q96 manifest.
3. `notebooks/02_stage_a_clip.ipynb` — frozen-CLIP linear-probe baseline.
4. `notebooks/03_rine_stage_b.ipynb` — the initial RINE ablation.
5. `notebooks/07_robustness_rerun.ipynb` — materializes all 14 independent
   Task 3 transform cells and retrains **controlled RINE**, the retained
   model, across seeds 42/43/44.

Each notebook is a thin launcher over `src/cya_detector/` and `scripts/`; no
model or training logic is inlined in any notebook. `final_test` stays
sealed throughout — none of this reproduction path touches it.

### Reproducing the Task 9 rejection and Task 10A/10B work

- `notebooks/09_texture_stage_d.ipynb` reproduces the Task 9 clean-gate pass.
- `notebooks/10_texture_robustness_stage1.ipynb` reproduces the Stage-1
  robustness rejection (`reject_texture_robustness_stage1`) documented below.
- `python scripts/fit_temperature_calibration.py --predictions <path> --seed 42 --output <report>`
  reproduces the temperature-calibration fit (Task 10B), once the real
  controlled-RINE seed-42 clean `selection_val` predictions CSV is restored
  locally (it lives on the shared Drive artifact root, not in this
  checkout — see the notebook's own restore cell for the path).

## Limitations and what we'd improve with more time

- **Reported probabilities are uncalibrated (T=1) by deliberate decision, not
  oversight.** Fitting one temperature on seed 42's clean `selection_val`
  logits (as originally planned) produced a degenerate result — that set has
  zero errors, so NLL minimization has nothing to penalize overconfidence
  and the fit ran to the search bound (T=0.05) instead of converging. Using
  it would have crushed informative probabilities toward 0/1 without
  changing any classification, since the binary threshold stays fixed
  regardless of temperature. We chose to report raw sigmoid probabilities
  instead. With more time and a calibration set that actually contains
  errors (e.g. one of the robustness cells, or held-out data with induced
  noise), a non-degenerate fit would likely improve the usefulness of the
  reported confidence scores without changing any pass/fail decision.
- **`final_test` has never been read.** Every result quoted here is on
  `selection_val`; the sealed held-out set is reserved for exactly one
  evaluation after the model, calibration, and threshold are fully frozen,
  and that run hasn't happened yet.
- **Robustness coverage stops at JPEG/blur/resize.** The locked evaluation
  only exercises the 14 independent Task 3 cells (four JPEG qualities, three
  blur levels, two resize scales, plus color/noise/crop families that were
  deferred). Real-world degradation (screenshots, social-media re-encodes,
  watermarking, cropping) is not directly covered, and the Task 9 result
  shows sensitivity to at least one un-covered regime (aggressive
  downsampling) even for the fusion candidate that failed — so it's
  plausible controlled RINE itself has blind spots that this evaluation
  wouldn't surface.
- **Every fusion/auxiliary-feature attempt was rejected**, including a
  local-patch texture path that passed its clean gate but collapsed toward
  random on aggressive resize (see below) — meaning the shipped model is
  frozen CLIP + RINE alone, with no PRNU, frequency, color, or texture
  signal. That's a validated, evidence-based decision, not a shortcut, but
  it does mean there's a real ceiling on what additional passive-signal
  fusion could offer without a fundamentally different approach (e.g. an
  architecture that doesn't lose the texture signal under downsampling,
  rather than trying to fuse a fragile one).
- **The C2PA Stage 0 check's manifest-schema parsing is unverified against a
  real signed asset.** It was implemented against the real installed
  `c2pa-python` 0.37.8 API surface and the public C2PA/IPTC digital-source-type
  vocabulary, and is unit-tested against synthetic manifest dictionaries, but
  we don't have a genuine C2PA-signed "trained algorithmic media" test image
  to validate the exact JSON shape end-to-end. The design fails closed (any
  parsing surprise returns `False` and falls through to the real predictor),
  so getting this wrong degrades gracefully rather than breaking anything —
  but it should be validated against a real signed sample before being
  relied on for the fast-path early exit.
- **No latency/memory numbers exist yet on target hardware.** The profiling
  utility (`src/cya_detector/evaluation/resource_profile.py`) is built and
  tested against fixtures, and the real predictor has been verified to run
  correctly end-to-end on local CPU, but it hasn't been profiled on the
  target Colab GPU.
- **Given more time**, the priority order would be: (1) fit a non-degenerate
  calibration on data that actually contains errors, (2) validate the C2PA
  schema against a real signed sample, (3) run the remaining Task 3
  noise/color-jitter/crop cells now that a full robustness screen exists as
  infrastructure, (4) run resource profiling on target hardware, (5) only
  then consider the optional post-baseline experiments already scoped in
  `docs/planning/nextSteps.md` (adapters/fine-tuning, self-training), each
  as its own separately gated experiment.

## Team member contributions

<!-- TODO: fill in participant names and their contribution areas before
     submitting, e.g.:
- Name A — Tasks 1-5 (data pipeline, CLIP baseline, evaluation harness)
- Name B — Tasks 6-8 (RINE, frequency/color/PRNU features, robustness fusion)
- Name C — Task 9 (texture-aware local-detail path)
- Name D — Task 10 (inference CLI skeleton, calibration, packaging)
-->

## Experimental history and rejected candidates

### Evaluation boundaries

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

### JPEG robustness strategy

JPEG can erase high-frequency generation artifacts and can also create a dataset shortcut when authentic and synthetic images have different encoding histories. The project addresses these as separate problems:

1. **Matched dataset preparation:** retain immutable originals, then create the primary clean view by re-encoding both labels with the same JPEG-quality distribution, encoder, and settings.
2. **JPEG-aware training:** create independent quality 90/70/50/30 variants from the matched clean parent to teach robustness to platform-style re-encoding.
3. **Representation-level backbone:** use frozen CLIP-ViT as the principal signal and treat frequency, texture, PRNU, color, and optics as auxiliary evidence subject to JPEG ablation.
4. **Bias auditing:** test whether format, resolution, file size, estimated JPEG quality, quantization tables, or feature validity predict the label before and after matching.

C2PA runs on immutable source bytes during dataset construction and on the exact received bytes at inference. Native-image forensic features remain experimental offline ablations; the shipped visual pipeline processes only the received view, and no compression artifact is proof of authenticity or synthesis. Matched JPEG normalization is never rerun at inference.

### Resize robustness strategy

The resize benchmark is one compound downsample-and-restore operation, evaluated independently at 0.5x and 0.25x severity. Both steps use bilinear interpolation with pinned library, antialiasing, rounding, color, and dtype settings; the restored output retains the parent dimensions and is cached losslessly so JPEG is not added accidentally.

At inference, the detector scores the received image once and does not generate extra resized, compressed, or blurred variants. Resize-aware training uses identical settings for both labels, while evaluation explicitly checks whether interpolation artifacts increase authentic false positives.

See [PRD.md](docs/product/PRD.md) for requirements, [design.md](docs/architecture/design.md) for the pipeline, [models.md](docs/architecture/models.md) for the model/evaluation plan, [training.md](docs/training/training.md) for training and fine-tuning, and [techStack.md](docs/architecture/techStack.md) for implementation choices.

### Rejected candidates

| Candidate | Result | Verdict |
|---|---|---|
| Frozen CLIP Stage A (linear probe) | 94.71% locked | Superseded by controlled RINE |
| RINE + frequency fusion | 52.15% locked (parent: 99.81%) | Rejected |
| RINE + Lab/color fusion | 98.95% locked, AI-accuracy regressed 1.82 pts | Rejected |
| PRNU-only (Task 8B-v2) | 78.09% locked | Rejected as standalone |
| RINE + PRNU-v2 fusion | 33.43% locked, collapsed in 2/3 seeds | Rejected |
| Task 9 texture (`global_local`) | Clean gate: 100% (passed). Stage-1 robustness: 93.13% mean vs. 99.80% for controlled RINE, worst-cell failures down to 50.2% AI-accuracy at `resize_scale_0.25` | Rejected at robustness |

**Controlled RINE (99.81% locked, 100.00% clean, 99.62% robustness) is the
only candidate that passed every gate**, and is the sole retained model. See
[`docs/planning/nextSteps.md`](docs/planning/nextSteps.md) for the full
per-task history, dates, and reasoning behind each decision.

### Task 8B — native physical pilot

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

## Task 10 architecture — inference CLI, calibration, and packaging

Task 10 is split into two independently scoped pieces so the CLI skeleton
did not have to wait on final model selection, and calibration/`final_test`
does not have to wait on CLI plumbing.

### Task 10A — Inference skeleton (implemented)

A model-agnostic, synchronous directory-inference CLI, layered as explicit
dependency injection (no plugin registry, no stateful monolith), under
`src/cya_detector/inference/`:

- `contracts.py` — pure types (`ValidationError`, `PredictionRecord`,
  `RunSummary`, `RunResult`, the `Predictor` protocol, exit-code constants).
  No I/O.
- `inputs.py` — deterministic recursive discovery (never opens file content;
  NFC-normalized relative-POSIX ordering; empty discovery and path
  collisions are fatal; symlinks never followed) and the decode/validate/
  normalize boundary (explicit decompression-bomb guard checked against the
  image header before full decode; five stable, path-sanitized error codes —
  `file_unreadable`, `decode_failed`, `unsupported_image`,
  `invalid_dimensions`, `decompression_bomb`; every image is normalized to
  owned RGB).
- `c2pa.py` — Stage 0 verified-claim check
  (`has_verified_ai_generation_claim`): `True` only for a verified, active
  claim containing a `c2pa.created` action with `digitalSourceType`
  `trainedAlgorithmicMedia`; every other outcome (missing dependency, no
  manifest, untrusted signature, malformed claim, authenticity-only claim,
  parser failure) returns `False` and falls through to the real predictor.
  No network access (remote manifest fetch and OCSP fetch are both
  explicitly disabled), no registry. This is the one module that
  deliberately catches every exception internally — a C2PA parsing failure
  is a documented *safe* `False`, unlike an uncatalogued failure anywhere
  else in the pipeline, which is fatal rather than silently reclassified.
- `runner.py` — orchestrates discovery → per-image validate → C2PA check →
  predict, with both the predictor and the C2PA check injected as plain
  callables. A predictor exception or an out-of-range/non-finite/boolean
  prediction is a fatal run failure, not a per-image error.
- `output.py` — publishes `predictions.json` (public `{"image_path", "pred"}`
  contract) and `report.json` (`schema_version`, `summary`, `errors[]`) only
  on non-fatal completion, both via write-temp-then-rename
  (`report.json` first, `predictions.json` last, so its presence is the
  strongest "this run is real and complete" signal). A fatal run never
  touches a prior successful run's output in the same `--output-dir`.
- `cli.py` — argument parsing, the exit-code boundary (`0` full success, `1`
  fatal, `2` argparse usage error, `3` partial success), per-image plus
  final-summary stdout progress, and the default stub predictor (fixed
  `0.5`) that makes [`run_inference.py`](run_inference.py) runnable today.

Tests: `tests/test_c2pa_inference.py`, `tests/test_directory_inference.py`
(discovery, decode/validate, and the full run_inference pipeline), and
`tests/test_inference_cli.py`.

### Task 10B — Model selection, calibration, and resource profiling

- **Model selection: done.** Controlled RINE seed 42 (99.85% locked score,
  the highest of the three retained seeds) is the packaged checkpoint. The
  real checkpoint and its clean `selection_val` predictions were restored
  from Drive to `artifacts/robustness/train-controlled-rine/seed_42/` and
  verified against the recorded 99.85%.
- **Calibration: run against real data — skipped, using T=1.**
  `src/cya_detector/evaluation/calibration.py`'s `fit_temperature()` fits one
  scalar temperature minimizing NLL on clean `selection_val` logits, and is
  unit-tested to correctly recover known temperatures on synthetic data. Run
  for real against seed 42's 165 clean rows, it returned a degenerate result:
  the fit hit the search bound (T=0.05) instead of an interior minimum,
  because that set has zero errors — with nothing to penalize
  overconfidence, NLL minimization only wants to sharpen further, and using
  T=0.05 would crush already-informative probabilities toward 0/1 without
  changing any classification (the threshold stays fixed at 0.5 regardless).
  **Decision: skip calibration, report raw sigmoid probabilities (T=1).**
  See `docs/planning/nextSteps.md` for the full writeup.
- **Real predictor: wired in.** `src/cya_detector/inference/rine_predictor.py`'s
  `RinePredictor` loads the frozen CLIP-ViT-L/14-336 backbone (pinned
  resolved revision), extracts layers 6/12/18/24 CLS tokens, and applies the
  loaded controlled-RINE seed-42 head — validating checkpoint identity and
  the resolved CLIP revision before serving any prediction, and refusing
  non-finite output. Wired into the CLI as an optional `--checkpoint` flag
  (`run_inference.py <dir> --output-dir <dir> --checkpoint
  artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt`);
  omitting it keeps the Task 10A stub for testing. 8 tests pass against a
  deterministic fixture CLIP, plus a real end-to-end run against the actual
  downloaded CLIP weights and checkpoint (`exit=0`, a real non-trivial
  probability, not the stub's constant `0.5`).
- **Resource profiling: implemented, not yet run on target hardware.**
  `src/cya_detector/evaluation/resource_profile.py`'s `profile_predictor()`
  measures per-call latency (mean/median/p95/max) and peak GPU memory;
  `checkpoint_disk_footprint()` reports on-disk checkpoint size. Both are
  model-agnostic and tested against fixtures; not yet run against the real
  checkpoint on the target Colab GPU.
- **Not started:** running the sealed `final_test` once everything above is
  frozen (needs separate explicit approval regardless of readiness), and any
  optional post-baseline experiment.
