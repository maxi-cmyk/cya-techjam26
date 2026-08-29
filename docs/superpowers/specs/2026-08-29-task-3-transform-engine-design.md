# Task 3 Transform Engine Design

## Purpose

Task 3 implements reproducible preprocessing, controlled training views, and independent
evaluation transformations. It keeps Task 2 matched-clean construction separate from both
training augmentation and robustness evaluation.

The implementation must preserve the central evaluation rule: every benchmark variant is
created directly from one matched-clean parent and receives exactly one transformation with
one parameter setting. Benchmark transformations are never chained.

## Scope

This task includes:

- deterministic benchmark transformation functions;
- optional lossless materialization of evaluation variants and provenance manifests;
- a lazy controlled training policy;
- SAFE as a separate training-only ablation;
- deterministic CLIP input preparation;
- command-line and Makefile entry points for materialization; and
- fixture tests for reproducibility, independence, dimensions, modes, and sampling balance.

This task does not include model loading, training loops, evaluation metrics, feature
extraction, or inference-time robustness probing.

## Separation of Responsibilities

The code must represent three distinct operations:

1. **Matched-clean construction:** Task 2 creates a label-independent canonical JPEG parent
   from immutable source bytes. Task 3 does not modify this builder.
2. **Training policy:** Task 3 selects either a clean parent or one temporary transformed view
   while a training sample is loaded. These views need not be saved.
3. **Evaluation transformation:** Task 3 applies one named benchmark cell directly to a
   matched-clean parent. The result may remain in memory or be materialized with provenance.

The corresponding package structure is:

```text
src/cya_detector/transforms/
├── __init__.py
├── benchmark.py
├── controlled.py
├── materialize.py
├── preprocessing.py
└── safe.py
```

`benchmark.py` owns exact pixel operations. `controlled.py` owns the primary training
schedule. `materialize.py` owns disk output and manifests. `preprocessing.py` owns RGB,
padding, and crop handling for CLIP. `safe.py` owns the separate SAFE ablation.

## Benchmark Cells

All benchmark operations accept a matched-clean RGB parent and return a new image plus a
record of the realized settings.

| Transform | Cells | Exact behaviour | Storage |
|---|---|---|---|
| JPEG | quality 90, 70, 50, 30 | Pillow JPEG, RGB, 4:4:4 subsampling, non-progressive, non-optimized, metadata stripped | JPEG |
| Gaussian blur | sigma 0.5, 1.0, 2.0 | Pillow Gaussian blur, where radius represents standard deviation | PNG |
| Resize round trip | scale 0.5, 0.25 | Bilinear downsample followed by bilinear restoration to the exact parent dimensions | PNG |
| Gaussian noise | sigma 0.02, 0.05, 0.10 | Independent RGB noise in normalized `[0, 1]` space, clipping and fixed rounding back to `uint8` | PNG |
| Colour jitter | fraction 0.20 | Independently sampled brightness, contrast, and saturation multipliers in `[0.8, 1.2]`, applied in that order | PNG |
| Centre crop | fraction 0.80 | Retain the central 80% of width and 80% of height without restoration | PNG |

The centre crop therefore retains approximately 64% of the parent area. The manifest records
the exact integer crop bounds and resulting dimensions.

### Resize contract

Pillow is the resize library and bilinear interpolation is used in both directions. The
intermediate dimension for an original dimension `d` and scale `s` is
`max(1, floor(d * s + 0.5))`. The upsample target is always the exact parent width and height.
The implementation records Pillow's exact runtime version, interpolation, dimension-rounding
rule, RGB handling, `uint8` range, and the library's fixed bilinear filtering behaviour. Resize
outputs are never JPEG encoded.

Reproducibility reports must include the exact dependency versions. Runs whose environment
metadata differs are not claimed to be cross-environment byte-identical.

### Randomness contract

Noise and colour jitter derive a local seed from the project seed, parent sample ID,
transformation name, and cell identifier using SHA-256. They do not consume shared global
random state. As a result, input order and worker scheduling cannot change output pixels.

The manifest stores the derived seed and all realized parameters. Colour jitter uses a fixed
brightness, contrast, saturation order.

## Materialized Evaluation Variants

The optional materializer writes one directory per transform cell, followed by a CSV manifest
and JSON report:

```text
output_root/
├── jpeg_q90/
├── blur_sigma_0.5/
├── resize_scale_0.5/
├── noise_sigma_0.02/
├── color_jitter_0.2/
├── center_crop_0.8/
├── transform_manifest.csv
└── transform_report.json
```

The complete run creates all configured cells; callers may request a declared subset for
fixtures or focused experiments. Every input record must identify a `matched_clean` view with
`transform=clean`. Other parents fail closed.

Every output row records:

- sample, parent, label, and split identifiers;
- parent and output paths and SHA-256 hashes;
- requested cell and realized settings;
- derived seed;
- input and output dimensions, mode, and format;
- transformation and preprocessing schema versions;
- Pillow version; and
- resize interpolation, filtering, rounding, colour, dtype, and restoration settings where
  applicable.

Non-JPEG transformations use PNG so persistence cannot add JPEG compression. Images are
written to temporary siblings and atomically replaced. Existing outputs with different hashes
cause an error unless explicit overwrite is enabled. The final manifest is published only
after every requested output succeeds.

## Controlled Training Policy

The primary sampler creates a deterministic epoch schedule with:

- 50% clean views and 50% transformed views;
- equal authentic and AI-generated representation; and
- uniform selection across the 14 configured transform cells.

When the epoch length is not exactly divisible, category counts may differ by at most one. A
seeded shuffle changes assignments between epochs while reproducing the same epoch when the
project seed, manifest, and epoch number are unchanged. The policy returns view specifications
and applies the selected operation lazily when the sample is loaded.

The controlled policy processes images in this order:

```text
matched-clean parent
→ clean or exactly one controlled transformation
→ pad if smaller than the model input
→ training crop
→ CLIP tensor conversion and normalization (owned by the later model integration)
```

## SAFE Ablation

SAFE is separately named and cannot be combined silently with the controlled policy. It is
training-only and follows this order:

1. pad if smaller than 336 by 336;
2. random 336 by 336 crop;
3. horizontal flip with probability 0.5;
4. brightness, contrast, and saturation jitter up to plus or minus 50%;
5. rotation sampled from minus 180 through plus 180 degrees with zero fill; and
6. 16 by 16 random masking, up to 75% coverage, with probability 0.5.

Random masking selects non-overlapping patches so realized coverage does not exceed the
declared maximum. SAFE uses local seeded generators rather than process-global randomness.

Validation and testing never use SAFE. SAFE parameters remain explicit configuration values
so later experiments can tune them without altering the benchmark contract.

## Deterministic Model-Input Preparation

Validation and test preparation converts to RGB, pads undersized dimensions, and applies a
fixed centre crop to the configured 336 by 336 model input. Padding is symmetric, with any odd
extra pixel placed on the right or bottom, and uses zero-valued RGB pixels. Images are never
upscaled merely to meet the model input size.

Training uses a seeded random crop after the selected training policy. Model-specific tensor
conversion and CLIP normalization remain Task 4 responsibilities, preventing Task 3 from
duplicating the model processor's normalization contract.

## Configuration and Interfaces

The existing benchmark cell lists in `configs/colab.json` remain authoritative. Task 3 adds
explicit schema fields for:

- transformation/preprocessing version;
- resize library, interpolation, filtering, and dimension rounding;
- benchmark output formats;
- model-input padding behaviour; and
- controlled and SAFE training-policy parameters.

Configuration validation rejects transform chaining, unknown cells, conflicting training
policies, and SAFE use outside training.

`scripts/materialize_transforms.py` exposes the materializer without placing transformation
logic in the script. Makefile targets provide a fixture materialization command and focused
Task 3 test command. No generated image dataset is committed to Git.

## Failure Behaviour

The transform engine raises clear errors for:

- unreadable parents;
- non-matched or already transformed parents;
- unsupported image modes that cannot be converted to RGB;
- undeclared transforms or settings;
- invalid dimensions or crop fractions;
- output collisions with different hashes; and
- controlled/SAFE policy conflicts.

Errors are not converted into clean or authentic predictions, and failed rows are not silently
omitted from a completed manifest.

## Verification

Fixture tests must demonstrate:

- each output has one matched-clean parent and exactly one transform cell;
- JPEG settings and storage format match the request;
- resize output dimensions exactly match the parent;
- centre-crop dimensions and bounds follow the 80%-per-dimension rule;
- seeded noise, colour jitter, SAFE operations, and epoch schedules reproduce exactly;
- input ordering does not affect random outputs;
- both labels receive matching clean/transformed and transform-cell distributions;
- grayscale, alpha-channel, very small, and odd-dimension inputs are handled consistently;
- SAFE is unavailable in validation and testing;
- chained or otherwise invalid parents fail; and
- repeated fixture materialization produces byte-identical files and manifests in the same
  recorded environment.

Task 3 is complete when the focused tests pass and a small fixture manifest can be regenerated
byte-for-byte under the recorded environment.
