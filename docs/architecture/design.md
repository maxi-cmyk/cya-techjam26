# Shipped System Architecture

This document describes the implementation that was frozen for packaging and the
one-time final evaluation. Detailed model decisions are in
[`models.md`](models.md), dependencies are in [`techStack.md`](techStack.md), and
the product boundary is in [`../product/PRD.md`](../product/PRD.md).

The public task is binary: distinguish a fully authentic image from a fully
AI-generated image. Mixed-origin, AI-edited, composited, inpainted, and
face-swapped images are outside the validated contract.

## Shipped Inference Flow

```text
input directory
    |
    v
recursive deterministic discovery
(.jpg/.jpeg/.png/.webp/.tif/.tiff; regular files only; no symlinks)
    |
    v
decode, validate, and copy to owned RGB
(maximum 64,000,000 pixels)
    |
    v
C2PA check against the original file path
    |-- verified active c2pa.created + trainedAlgorithmicMedia --> pred = 1.0
    `-- absent, invalid, untrusted, malformed, or parser failure ----------+
                                                                         |
                                                                         v
                                                       pinned CLIP processor
                                                                         |
                                                                         v
                                                  frozen CLIP ViT-L/14-336
                                                 layers 6, 12, 18, and 24
                                                                         |
                                                                         v
                                             learned softmax layer weighting
                                                                         |
                                                                         v
                                                    one linear binary head
                                                                         |
                                                                         v
                                                raw sigmoid score (T = 1)
                                                                         |
                                                                         v
                                    report.json, then predictions.json atomically
```

The model threshold used for evaluation is `0.5`. The public JSON retains the
continuous `pred` value rather than converting it to a label. `pred` is a raw
sigmoid score, not a separately calibrated probability guarantee.

Frequency, texture, PRNU, RGB/Lab, chromatic-aberration, and radial-distortion
features are not part of this runtime path. The detector also does not create a
matched JPEG, resize, crop, or other test-time variant from an input image.

## Component Boundaries

### Input discovery and validation

[`../../src/cya_detector/inference/inputs.py`](../../src/cya_detector/inference/inputs.py)
recursively discovers supported extensions, skips symlinked files and directories,
normalizes relative paths to NFC POSIX form, rejects normalized path collisions,
and orders work by UTF-8 path bytes. Each image is decoded once and converted to
an owned RGB image. Recognized per-image failures are recorded; an empty input,
path collision, or uncatalogued failure is fatal.

Validation occurs before the C2PA check. The decoded RGB image is passed to the
visual predictor, while C2PA reads the original file path so it can inspect the
original bytes and embedded manifest.

### C2PA provenance shortcut

[`../../src/cya_detector/inference/c2pa.py`](../../src/cya_detector/inference/c2pa.py)
returns `true` only when the active manifest is trusted and contains a
`c2pa.created` action whose `digitalSourceType` includes
`trainedAlgorithmicMedia`. This is a one-way AI-generation shortcut; no manifest
can produce an authenticity verdict. Remote-manifest and OCSP fetching are
disabled. Missing dependencies, missing manifests, invalid signatures, malformed
claims, and parser failures safely fall through to the visual model.

### Controlled-RINE visual model

[`../../src/cya_detector/inference/rine_predictor.py`](../../src/cya_detector/inference/rine_predictor.py)
loads `openai/clip-vit-large-patch14-336` at resolved revision
`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. The CLIP backbone is frozen. CLS
representations from layers 6, 12, 18, and 24 are fused by four learned layer
logits after softmax, then passed through one linear classifier.

The packaged checkpoint must identify itself as controlled-RINE robustness
training, seed 42, fixed-Q96 matching, and the same four layers. The adapter also
checks the resolved CLIP revision and rejects malformed checkpoints or non-finite
outputs.

### Output and failure behavior

[`../../src/cya_detector/inference/runner.py`](../../src/cya_detector/inference/runner.py)
returns one `{image_path, pred}` record per valid image. Publication writes the
run report before predictions using atomic replacements. Fatal runs publish
nothing; recognized invalid images allow partial output.

CLI exit codes are:

| Code | Meaning |
|---:|---|
| `0` | Every discovered image was scored |
| `1` | Fatal run failure |
| `2` | Command-line usage error |
| `3` | Partial success with one or more invalid images |

The default checkpoint is
`artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt`. If it is
missing, the current CLI prints a loud warning and falls back to a test stub that
returns `0.5`; those values are not model results. `--no-checkpoint` selects that
stub explicitly.

## Offline Data and Selection Architecture

Each eligible source has an immutable `source_original`. A fixed-Q96
`matched_clean` derivative removes label-correlated encoding history. Robustness
variants are created independently from that canonical clean parent: one
transformation and one parameter per derivative, never a chain of transformations.

The source-level split is 60% seed training, 25% self-training pool, 7.5%
selection validation, and 7.5% sealed final test. All derivatives of a source
remain in its split. The 14-cell development robustness matrix covers JPEG,
blur, two resize scales, noise, color jitter, and center crop at locked parameter
values.

Architecture selection uses the development score:

`0.50 x clean accuracy + 0.50 x mean independent-transform accuracy`

The sealed final test is different: it contained 141 direct fixed-Q96
matched-clean images and was read once on 2026-08-31. It did not contain a second
transformed bank. Development robustness results must not be presented as
final-test measurements.

## Recorded Architecture Decisions

| Candidate | Recorded result | Decision |
|---|---:|---|
| Controlled RINE, seeds 42/43/44 | 100.00% clean, 99.62% robustness, 99.81% locked mean | Retained; seed 42 packaged at 99.85% locked |
| RINE + frequency | 52.15% locked mean | Rejected; fast track disabled |
| RINE + Lab | 98.95% locked mean with AI-class regression | Rejected |
| RINE + PRNU-v2 | 33.43% locked mean after two-seed collapse | Rejected; PRNU remains diagnostic only |
| Global + texture patches | 93.13% robustness versus 99.80% controlled RINE | Rejected |
| Chromatic aberration / distortion | Required calibration and scene support absent | Not eligible |
| Temperature scaling | Fit converged to lower bound on an error-free clean set | Rejected; use `T=1` |

## Frozen Evidence and Remaining Limits

The one-time final test scored 140/141 images correctly: 99.29% overall,
100.00% AI-generated accuracy (69/69), 98.61% authentic accuracy (71/72), a
1.3889% authentic false-positive rate, and ECE 0.0189 at threshold `0.5`.

Target-hardware latency, peak memory, and model-cache/disk profiling remain
outstanding. The C2PA parser and fall-through behavior are unit-tested, but the
manifest-schema path has not yet been demonstrated end to end with a real signed
asset. Results outside the fully authentic versus fully AI-generated boundary are
not validated.
