# Implemented Tech Stack

This stack supports the frozen controlled-RINE detector, its local directory CLI,
and the offline experiment pipeline. [`design.md`](design.md) defines component
boundaries; [`models.md`](models.md) records the retained model and rejected
candidates.

## Runtime Stack

| Technology | Implemented role |
|---|---|
| Python 3.10+ | CLI, orchestration, validation, and evaluation code |
| PyTorch 2.3–2.x | Frozen CLIP inference and RINE head execution |
| Hugging Face Transformers 4.44–5.x | Pinned CLIP model and image processor |
| Pillow | Image decoding, pixel-limit validation, and owned RGB conversion |
| `c2pa-python` 0.37.x | Offline C2PA manifest parsing and signature state |
| NumPy | Numeric support and offline metric/data operations |
| `pathlib` and JSON | Local paths and public prediction/report artifacts |

The model identifier is `openai/clip-vit-large-patch14-336`, pinned to resolved
revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. The repository checkpoint
stores the small controlled-RINE head; the much larger frozen CLIP weights are
obtained from Hugging Face and cached on first use.

The runtime accepts JPEG, PNG, WebP, and TIFF extensions (`.jpg`, `.jpeg`,
`.png`, `.webp`, `.tif`, `.tiff`). It runs locally and synchronously. It does not
require a database, queue, web server, external inference API, or network access
after model weights are cached. C2PA remote-manifest and OCSP fetching are
explicitly disabled.

## Runtime Components

| Component | Source | Responsibility |
|---|---|---|
| Command entry point | [`../../run_inference.py`](../../run_inference.py) | Starts directory inference |
| CLI wiring | [`../../src/cya_detector/inference/cli.py`](../../src/cya_detector/inference/cli.py) | Arguments, checkpoint/device selection, progress, exit codes |
| Discovery and decoding | [`../../src/cya_detector/inference/inputs.py`](../../src/cya_detector/inference/inputs.py) | Deterministic traversal, validation, RGB conversion |
| Provenance shortcut | [`../../src/cya_detector/inference/c2pa.py`](../../src/cya_detector/inference/c2pa.py) | Verified AI-generation claim check |
| Visual predictor | [`../../src/cya_detector/inference/rine_predictor.py`](../../src/cya_detector/inference/rine_predictor.py) | Pinned CLIP plus retained RINE head |
| Run orchestration | [`../../src/cya_detector/inference/runner.py`](../../src/cya_detector/inference/runner.py) | Per-image fall-through and fatal/partial boundaries |
| Artifact publication | [`../../src/cya_detector/inference/output.py`](../../src/cya_detector/inference/output.py) | Atomic `report.json` and `predictions.json` writes |

Example invocation:

```bash
python run_inference.py INPUT_DIRECTORY --output-dir OUTPUT_DIRECTORY
```

The real checkpoint defaults to
`artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt`. If it is not
available, the CLI warns and uses a constant-`0.5` test stub. A release or demo
must therefore confirm that the checkpoint was loaded; stub results are
meaningless.

## Offline Training and Evaluation Stack

| Technology | Role |
|---|---|
| torchvision | Tensor/image operations used by model and experiment code |
| pandas | Manifest and prediction-table handling |
| SciPy and scikit-learn | Metrics, calibration, and statistical utilities |
| scikit-image, PyWavelets, OpenCV | Rejected auxiliary-feature experiments and image analysis |
| Accelerate and safetensors | Hugging Face model/runtime support |
| tqdm | Long-running preprocessing and evaluation progress |
| Jupyter/Google Colab | GPU training and controlled notebook execution |
| Google Drive | External storage for source data and large run artifacts |

`requirements.txt` defines the complete local development/runtime environment.
`requirements-colab.txt` intentionally does not replace Colab's matched
PyTorch/torchvision/CUDA stack.

## Data Actually Used

| Source | Role in this project |
|---|---|
| SID_Set | Primary binary corpus: 20,000 raw rows, 19,882 eligible after validation; fixed-Q96 derivatives used for controlled training and evaluation |
| PREMIER N1/N2 | CC BY-SA 4.0 native-camera/device-signal rows for Task 8B diagnostics |
| Tiny-GenImage/GenImage AI subset | CC BY-NC-SA 4.0 AI comparison rows for Task 8B diagnostics, used under the recorded non-commercial research assumption |

Dataset provenance, licenses, eligibility, hashes, and split membership belong in
the data manifests and training documentation. Other datasets discussed during
planning were not silently added to fitting or final selection.

## Deliberately Excluded from Runtime

- frequency vectors and the proposed Stage-1 early exit;
- texture-rich local crops and patch aggregation;
- reference-free PRNU-v2 features;
- RGB/Lab inter-channel correlation features;
- chromatic-aberration and radial-distortion estimates;
- temperature scaling beyond `T=1`;
- ConvNeXt or any second-model ensemble;
- test-time JPEG matching or robustness transformations.

These paths remain useful experiment history, but none improved the locked
controlled-RINE parent under the retention gate or had sufficient calibration
support.

## Current Operational Limits

- First use needs access to the pinned CLIP weights unless they are already
  cached.
- Target-hardware latency, peak memory, and disk/cache consumption have not yet
  been profiled.
- The C2PA integration is unit-tested but still needs an end-to-end check with a
  real signed AI-generated asset.
- The validated model contract covers fully authentic versus fully AI-generated
  images only.
