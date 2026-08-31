# Robust AI-Generated Image Detection

This branch is the judge-facing submission package: the inference CLI, a
thin FastAPI backend, and the React UI that calls it. The full development
history (data pipeline, training, rejected candidates, robustness
experiments across Tasks 1–10) lives on `main`; this branch keeps only what's
needed to install and run the final result.

## Project overview

This project classifies images as either **fully authentic** or **fully
AI-generated**, weighted 50% clean accuracy and 50% robustness to common
image transforms (JPEG re-compression, blur, resize).

The retained model is **controlled RINE**: a frozen-CLIP intermediate-layer
representation (CLIP-ViT-L/14-336, layers 6/12/18/24) with a lightweight
trained linear head, retrained under a balanced clean-or-one-transform
sampler. It reached 100.00% clean accuracy, 99.62% mean robustness accuracy,
and a 99.81% locked 50/50 score across seeds 42/43/44. Frequency, color/Lab,
PRNU, and a texture-aware local-patch path were all built and evaluated as
candidate additions and were all **rejected** after failing the locked-score
or robustness gate (see `docs/planning/nextSteps.md` on `main` for the full
experimental history).

**Sealed `final_test` result (run exactly once, 2026-08-31, 141 held-out
samples): 99.29% overall accuracy** — 100.00% AI-generated accuracy (69/69,
0.00% false negative rate), 98.61% authentic accuracy (71/72, **1.39% false
positive rate**), controlled RINE seed 42, T=1 (uncalibrated, see
[Limitations](#limitations-and-what-wed-improve-with-more-time)).

**Submission entry point:** [`run_inference.py`](run_inference.py) at the
repository root — takes a directory of images, writes `predictions.json` and
`report.json`, and uses the real controlled-RINE seed-42 checkpoint by
default. The checkpoint is committed directly in this repository
(`artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt`, 17KB;
the frozen CLIP backbone downloads automatically from Hugging Face on first
use). **No Colab, Drive access, or manual staging needed** — `git clone` +
`pip install` + run is the complete judge-facing path.

The `backend/` FastAPI service and `frontend/` React app wrap the same CLI
logic behind a drag-and-drop UI, for a visual demo of the same model.

## Setup and installation instructions

Requires Python >= 3.10.

```bash
git clone <this repository>
cd cya-techjam26
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run the test suite (no GPU or dataset required):

```bash
python -m pytest tests/
```

### Backend + frontend (optional UI demo)

```bash
# Terminal 1 — backend API (from the repo root)
PYTHONPATH=src python -m uvicorn backend.app:app --port 8000

# Terminal 2 — frontend dev server
cd frontend
npm install
npm run dev
```

Open the printed frontend URL (default `http://localhost:5173`), drag in an
image, and it's scored by the same controlled-RINE model as the CLI via
`POST http://localhost:8000/predict`.

## Steps to reproduce your results

### Running the submission script

```bash
python run_inference.py <path-to-image-directory> --output-dir <output-directory>
```

Recursively discovers `.jpg`/`.jpeg`/`.png`/`.webp`/`.tif`/`.tiff` files
under `<path-to-image-directory>` (case-insensitive, symlinks never
followed), runs each through a C2PA verified-AI-generation-claim check
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

### Reproducing the training/evaluation results

The 99.81% locked-score result, the Task 9 texture-path rejection, and the
sealed 99.29% `final_test` result all come from Colab notebook runs recorded
in `docs/planning/nextSteps.md` on the `main` branch, which also carries the
full data pipeline, training scripts, and notebooks required to reproduce
them from scratch. `final_test` itself is not reproducible by design — it
has already been run exactly once and the code that reads it refuses to run
a second time against existing output.

This `submission` branch intentionally omits that pipeline so the
judge-facing footprint stays small; check out `main` for it.

## Limitations and what we'd improve with more time

- **Reported probabilities are uncalibrated (T=1) by deliberate decision, not
  oversight.** Fitting one temperature on seed 42's clean `selection_val`
  logits produced a degenerate result — that set has zero classification
  errors, so NLL minimization had nothing to penalize and just drove the
  temperature to the search bound. We chose to ship raw sigmoid outputs
  rather than a fit we couldn't validate.
- **C2PA verified-claim detection is implemented against the spec but not
  validated against a real signed sample** — it should be checked against
  one before being relied on for the fast-path early exit.
- **No latency/memory numbers exist yet on target hardware.** The predictor
  has been verified to run correctly end-to-end on local CPU, but hasn't
  been profiled on the target Colab GPU.
- **Given more time**, priority order:
  (1) fit a non-degenerate calibration on data that actually contains errors
  (2) validate the C2PA schema against a real signed sample
  (3) run resource profiling on target hardware
  (4) revisit the rejected texture-aware path with a more robust fusion strategy under aggressive downsampling/blur, since it matched controlled RINE on clean accuracy but failed only on robustness.

## Team member contributions

Max (maxi-cmyk) — Built the repository infrastructure, reproducible configuration, Colab workflows, and image-data pipeline; implemented the CLIP/RINE detector baselines, evaluation and robustness tooling, physical-feature experiments, inference backend and CLI, calibration, resource profiling, final-test workflow, and submission integration.

Rae (raetan2023) — Built the deterministic image-transformation and augmentation framework for JPEG, blur, resize, noise, color, and crop robustness; designed and implemented the texture-aware local-detail detector; and strengthened testing, cross-platform behavior, architecture, planning, and submission documentation.

Shan He (cshsean) — Implemented the initial deterministic frequency, color, optics, PRNU, and texture feature extractors with tests; authored SAFE augmentation and training-data guidance; and built the React/Vite image-prediction and evaluation-dashboard frontend.
