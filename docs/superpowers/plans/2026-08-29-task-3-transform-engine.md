# Task 3 Transform Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible clean-or-one-transform training views, independent evaluation variants, deterministic CLIP crops, and an isolated SAFE training ablation.

**Architecture:** Pure Pillow/NumPy transformations support a lazy controlled-training scheduler and an optional disk materializer. Parent validation prevents chained evaluation transforms, SHA-256-derived local seeds remove processing-order dependence, and SAFE remains a separate training-only API.

**Tech Stack:** Python 3.10+, Pillow, NumPy, existing CSV/JSON manifest helpers, `unittest`, Ruff, Make.

**Spec:** `docs/superpowers/specs/2026-08-29-task-3-transform-engine-design.md`

## Global Constraints

- Every benchmark variant starts from `image_view=matched_clean` and `transform=clean`.
- Each variant receives exactly one of the 14 configured benchmark cells; chaining fails.
- The 14 cells are JPEG Q90/Q70/Q50/Q30, Gaussian blur 0.5/1.0/2.0, Resize
  0.5/0.25, Gaussian noise 0.02/0.05/0.10, Colour jitter 0.20, and Centre crop 0.80.
- Resize uses Pillow bilinear in both directions, `max(1, floor(d * scale + 0.5))` intermediate dimensions, and exact restoration dimensions.
- JPEG uses JPEG 4:4:4 storage; all other benchmark outputs use PNG.
- Random operations use local SHA-256-derived seeds, never shared global random state.
- Controlled training is 50% clean, 50% transformed, label-balanced, and uniform over cells to within one sample.
- SAFE is training-only and mutually exclusive with the controlled policy.
- Inputs smaller than 336 by 336 are symmetrically zero-padded, never upscaled, before cropping.
- Generated image datasets and artifacts must not be committed.
- Repeated fixture materialization in the same recorded environment must produce
  byte-identical image files and equivalent manifests after absolute paths are normalized.

## File Map

- `src/cya_detector/transforms/benchmark.py`: cells, seeds, parent checks, pixel operations.
- `src/cya_detector/transforms/materialize.py`: image output, provenance rows, reports.
- `src/cya_detector/transforms/preprocessing.py`: RGB, padding, centre/random crops.
- `src/cya_detector/transforms/controlled.py`: balanced epoch schedules and lazy views.
- `src/cya_detector/transforms/safe.py`: isolated SAFE training augmentation.
- `scripts/materialize_transforms.py`: thin materialization CLI.
- `configs/colab.json` and `src/cya_detector/config.py`: frozen settings and validation.
- `src/cya_detector/data/manifest.py`: Task 3 provenance columns.
- `tests/test_benchmark_transforms.py`, `tests/test_transform_materialization.py`, `tests/test_preprocessing.py`, `tests/test_controlled_sampler.py`, `tests/test_safe_transforms.py`: focused contracts.
- `Makefile` and `docs/planning/nextSteps.md`: workflow and verified status.

---

### Task 1: Freeze and Validate Task 3 Configuration

**Files:**
- Modify: `configs/colab.json`
- Modify: `src/cya_detector/config.py`
- Modify: `src/cya_detector/data/manifest.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `load_config(path) -> dict`.
- Produces: validated `transform_engine`, `training_policy`, and Task 3 manifest columns.

- [ ] **Step 1: Write failing contract tests**

```python
def test_task3_contract_is_frozen(self) -> None:
    config = load_config(CONFIG_PATH)
    engine = config["transform_engine"]
    self.assertEqual(engine["version"], "task3-v1")
    self.assertEqual(engine["resize_library"], "Pillow")
    self.assertEqual(engine["resize_interpolation"], "bilinear")
    self.assertEqual(engine["dimension_rounding"], "floor(d * scale + 0.5)")
    self.assertEqual(engine["non_jpeg_storage"], "PNG")

def test_training_policies_are_mutually_exclusive(self) -> None:
    config = load_config(CONFIG_PATH)
    config["training_policy"]["controlled"]["enabled"] = True
    config["training_policy"]["safe"]["enabled"] = True
    with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
        validate_config(config)
```

Also assert `MANIFEST_FIELDS` contains `parent_sha256`, `realized_parameters`,
`transform_version`, and `preprocessing_version`.

- [ ] **Step 2: Run tests to verify failure**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_config -v`

Expected: FAIL because the new sections and fields do not exist.

- [ ] **Step 3: Add exact configuration values and validation**

Add these sections to `configs/colab.json`:

```json
"transform_engine": {
  "version": "task3-v1",
  "preprocessing_version": "clip-crop-v1",
  "resize_library": "Pillow",
  "resize_interpolation": "bilinear",
  "resize_filtering": "pillow_bilinear_fixed",
  "dimension_rounding": "floor(d * scale + 0.5)",
  "jpeg_storage": "JPEG",
  "non_jpeg_storage": "PNG",
  "padding": "symmetric_zero"
},
"training_policy": {
  "controlled": {
    "enabled": true,
    "clean_fraction": 0.5,
    "transformed_fraction": 0.5,
    "balance_labels": true,
    "uniform_transform_cells": true
  },
  "safe": {
    "enabled": false,
    "horizontal_flip_probability": 0.5,
    "color_jitter_fraction": 0.5,
    "rotation_degrees": 180,
    "mask_patch_size": 16,
    "mask_max_fraction": 0.75,
    "mask_probability": 0.5
  }
}
```

Add both to `REQUIRED_SECTIONS`. Validate exact engine strings, fractions summing to one,
model input size 336, probability ranges, and policy mutual exclusion. Append the four
manifest fields without renaming Task 2 columns.

- [ ] **Step 4: Run contract and Task 2 regression tests**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_config tests.test_matched_clean tests.test_data_manifest -v`

Expected: PASS; Task 2 writers leave new columns blank.

- [ ] **Step 5: Commit**

```powershell
git add configs/colab.json src/cya_detector/config.py src/cya_detector/data/manifest.py tests/test_config.py
git commit -m "feat: freeze task 3 transform configuration"
```

---

### Task 2: Implement Independent Benchmark Operations

**Files:**
- Create: `src/cya_detector/transforms/__init__.py`
- Create: `src/cya_detector/transforms/benchmark.py`
- Create: `tests/test_benchmark_transforms.py`

**Interfaces:**
- Consumes: validated benchmark and engine configuration.
- Produces: `TransformCell`, `TransformResult`, `TransformContractError`, `benchmark_cells`, `derive_seed`, `validate_parent_record`, `apply_benchmark`.

- [ ] **Step 1: Write failing cell, parent, geometry, and seed tests**

```python
def test_config_expands_to_fourteen_cells(self) -> None:
    cells = benchmark_cells(load_config(CONFIG_PATH))
    self.assertEqual(len(cells), 14)
    self.assertEqual(len({cell.cell_id for cell in cells}), 14)

def test_rejects_transformed_parent(self) -> None:
    row = {"sample_id": "x", "image_view": "benchmark", "transform": "blur"}
    with self.assertRaisesRegex(TransformContractError, "matched_clean"):
        validate_parent_record(row)

def test_resize_restores_odd_parent_dimensions(self) -> None:
    result = apply_benchmark(self.gradient((73, 57)), self.resize_half, "a", 42)
    self.assertEqual(result.image.size, (73, 57))
    self.assertEqual(result.realized["intermediate_size"], [37, 29])

def test_center_crop_retains_eighty_percent_per_dimension(self) -> None:
    result = apply_benchmark(self.gradient((73, 57)), self.crop, "a", 42)
    self.assertEqual(result.image.size, (58, 46))
```

For noise and jitter, assert identical bytes and realized parameters for repeated sample/seed
pairs and different output for a different sample ID.

- [ ] **Step 2: Run tests to verify import failure**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_benchmark_transforms -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Define cells, results, seed derivation, and parent checks**

```python
@dataclass(frozen=True)
class TransformCell:
    name: str
    parameter: int | float
    cell_id: str
    output_format: str
    stochastic: bool = False

@dataclass(frozen=True)
class TransformResult:
    image: Image.Image
    realized: dict[str, Any]

def derive_seed(project_seed: int, sample_id: str, cell_id: str) -> int:
    digest = hashlib.sha256(f"{project_seed}:{sample_id}:{cell_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
```

`validate_parent_record` requires `matched_clean` and `clean`. `benchmark_cells` expands the
frozen configuration in stable order and assigns JPEG only to JPEG cells.

- [ ] **Step 4: Implement exact pixel operations**

```python
def apply_benchmark(
    image: Image.Image,
    cell: TransformCell,
    sample_id: str,
    project_seed: int,
) -> TransformResult:
    rgb = image.convert("RGB")
    seed = derive_seed(project_seed, sample_id, cell.cell_id)
    # Dispatch only declared names; every unknown cell raises TransformContractError.
```

Use Pillow `GaussianBlur(radius=sigma)`, bilinear resize, and `ImageEnhance` in brightness,
contrast, saturation order. Use NumPy `default_rng(seed).normal` in normalized float space,
clip, multiply by 255, and round with `floor(value + 0.5)`. JPEG round-trips through `BytesIO`
with `subsampling=0`, `optimize=False`, `progressive=False`, and empty EXIF. Record all realized
values and exact crop bounds.

- [ ] **Step 5: Run focused tests and lint**

```powershell
conda run -n cya-techjam26 python -m unittest tests.test_benchmark_transforms -v
conda run -n cya-techjam26 python -m ruff check src/cya_detector/transforms/benchmark.py tests/test_benchmark_transforms.py
```

Expected: PASS and no Ruff errors.

- [ ] **Step 6: Commit**

```powershell
git add src/cya_detector/transforms tests/test_benchmark_transforms.py
git commit -m "feat: add independent benchmark transforms"
```

---

### Task 3: Materialize Evaluation Variants and Provenance

**Files:**
- Create: `src/cya_detector/transforms/materialize.py`
- Create: `tests/test_transform_materialization.py`

**Interfaces:**
- Consumes: benchmark APIs plus existing manifest readers/writers and SHA-256 helper.
- Produces: `materialize_benchmarks(...) -> dict[str, Any]` and `TransformMaterializationError`.

- [ ] **Step 1: Write failing parentage and output tests**

```python
def test_materializes_cells_directly_from_clean_parents(self) -> None:
    report = materialize_benchmarks(
        input_manifest=self.manifest,
        output_root=self.root / "variants",
        output_manifest=self.root / "variants.csv",
        report_path=self.root / "report.json",
        config=load_config(CONFIG_PATH),
        cells=(self.jpeg_cell, self.resize_cell),
    )
    rows = read_manifest(self.root / "variants.csv")
    self.assertEqual(len(rows), 4)
    self.assertEqual({r["parent_id"] for r in rows}, self.parent_ids)
    self.assertEqual({r["image_view"] for r in rows}, {"benchmark"})
    self.assertEqual(report["cell_counts"], {"jpeg_q90": 2, "resize_scale_0.5": 2})

def test_chained_parent_does_not_publish_manifest(self) -> None:
    self.write_parent(image_view="benchmark", transform="blur")
    with self.assertRaises(TransformContractError):
        self.materialize()
    self.assertFalse((self.root / "variants.csv").exists())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_transform_materialization -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement atomic materialization**

```python
def materialize_benchmarks(
    *,
    input_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    report_path: Path,
    config: dict[str, Any],
    cells: Sequence[TransformCell] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]: ...
```

Sort parents and cells. Validate before opening, read only `image_path`, apply one cell, save to
a temporary sibling, verify it, hash it, then replace the destination. Use `.jpg` only for JPEG
and `.png` otherwise. Store requested and realized settings as compact sorted JSON. Publish CSV
and report only after all rows succeed.

- [ ] **Step 4: Add collision and regeneration tests**

Assert a differing existing output fails unless `overwrite=True`. Materialize twice into
separate roots; after excluding absolute path strings, assert identical hashes, realized
parameters, counts, and image bytes.

- [ ] **Step 5: Run focused and regression tests**

```powershell
conda run -n cya-techjam26 python -m unittest tests.test_transform_materialization tests.test_matched_clean -v
conda run -n cya-techjam26 python -m ruff check src/cya_detector/transforms/materialize.py tests/test_transform_materialization.py
```

Expected: PASS with no partial final manifest on deliberate failure.

- [ ] **Step 6: Commit**

```powershell
git add src/cya_detector/transforms/materialize.py tests/test_transform_materialization.py
git commit -m "feat: materialize benchmark transform variants"
```

---

### Task 4: Implement Deterministic Model-Input Preparation

**Files:**
- Create: `src/cya_detector/transforms/preprocessing.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: Pillow images and an integer input size.
- Produces: `to_rgb`, `pad_to_minimum`, `center_crop_input`, `random_crop_input`.

- [ ] **Step 1: Write failing RGB, padding, and crop tests**

```python
def test_small_image_is_padded_without_resizing(self) -> None:
    result = center_crop_input(Image.new("L", (333, 331), 255), size=336)
    self.assertEqual(result.size, (336, 336))
    self.assertEqual(result.mode, "RGB")
    self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))

def test_seeded_random_crop_repeats(self) -> None:
    first = random_crop_input(self.gradient(), 336, seed=123)
    second = random_crop_input(self.gradient(), 336, seed=123)
    self.assertEqual(first.tobytes(), second.tobytes())
```

Add an odd-padding assertion proving the extra pixel goes right/bottom and an RGBA conversion
test proving a stable black-background RGB composite.

- [ ] **Step 2: Run tests to verify failure**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_preprocessing -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement preprocessing helpers**

```python
def to_rgb(image: Image.Image) -> Image.Image: ...
def pad_to_minimum(image: Image.Image, size: int) -> Image.Image: ...
def center_crop_input(image: Image.Image, size: int) -> Image.Image: ...
def random_crop_input(image: Image.Image, size: int, *, seed: int) -> Image.Image: ...
```

Composite alpha onto black before RGB conversion. Compute left/top with floor division and put
remainders right/bottom. Use local `random.Random(seed)` for crop coordinates. Reject
nonpositive sizes.

- [ ] **Step 4: Run tests and commit**

```powershell
conda run -n cya-techjam26 python -m unittest tests.test_preprocessing -v
conda run -n cya-techjam26 python -m ruff check src/cya_detector/transforms/preprocessing.py tests/test_preprocessing.py
git add src/cya_detector/transforms/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add deterministic clip input crops"
```

---

### Task 5: Implement the Controlled Training Schedule

**Files:**
- Create: `src/cya_detector/transforms/controlled.py`
- Create: `tests/test_controlled_sampler.py`
- Modify: `src/cya_detector/transforms/__init__.py`

**Interfaces:**
- Consumes: matched-clean rows, benchmark cells/operations, and random input crop.
- Produces: `TrainingView`, `build_controlled_epoch`, `apply_training_view`.

- [ ] **Step 1: Write failing schedule tests**

```python
def test_epoch_is_label_and_view_balanced(self) -> None:
    schedule = build_controlled_epoch(
        self.records, self.cells, epoch_size=112, project_seed=42, epoch=0
    )
    counts = Counter((v.label, v.cell_id == "clean") for v in schedule)
    self.assertEqual(set(counts.values()), {28})

def test_cells_are_uniform_per_label(self) -> None:
    schedule = self.schedule(epoch=0)
    for label in ("authentic", "ai_generated"):
        counts = Counter(v.cell_id for v in schedule if v.label == label and v.cell_id != "clean")
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

def test_schedule_repeats_by_epoch(self) -> None:
    self.assertEqual(self.schedule(epoch=3), self.schedule(epoch=3))
    self.assertNotEqual(self.schedule(epoch=3), self.schedule(epoch=4))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_controlled_sampler -v`

Expected: FAIL because the scheduler does not exist.

- [ ] **Step 3: Implement scheduling and lazy application**

```python
@dataclass(frozen=True)
class TrainingView:
    sample_id: str
    label: str
    image_path: str
    cell_id: str
    seed: int

def build_controlled_epoch(
    records: Sequence[Mapping[str, str]], cells: Sequence[TransformCell], *,
    epoch_size: int, project_seed: int, epoch: int,
) -> tuple[TrainingView, ...]: ...

def apply_training_view(
    view: TrainingView, cells_by_id: Mapping[str, TransformCell], *, input_size: int,
) -> Image.Image: ...
```

Validate both labels and all parents. Allocate alternating label and clean/transformed slots.
Cycle through independently seeded/shuffled parent pools and a seeded/shuffled 14-cell list.
Schedule building must not open images. Lazy application opens exactly once, applies zero or
one benchmark transform, then performs a separately seeded random crop.

- [ ] **Step 4: Add error and laziness tests**

Reject missing labels, chained parents, unknown cell IDs, and nonpositive epoch sizes. Mock
`Image.open` to prove scheduling performs no read and application performs exactly one read.

- [ ] **Step 5: Run tests and commit**

```powershell
conda run -n cya-techjam26 python -m unittest tests.test_controlled_sampler -v
conda run -n cya-techjam26 python -m ruff check src/cya_detector/transforms/controlled.py tests/test_controlled_sampler.py
git add src/cya_detector/transforms/controlled.py src/cya_detector/transforms/__init__.py tests/test_controlled_sampler.py
git commit -m "feat: add balanced controlled transform sampler"
```

---

### Task 6: Implement the Isolated SAFE Ablation

**Files:**
- Create: `src/cya_detector/transforms/safe.py`
- Create: `tests/test_safe_transforms.py`
- Modify: `src/cya_detector/transforms/__init__.py`

**Interfaces:**
- Consumes: image, settings, sample ID, seed, epoch, and phase.
- Produces: `SafeSettings`, `apply_safe`, `validate_training_policy`, `SafePolicyError`.

- [ ] **Step 1: Write failing phase, seed, and masking tests**

```python
def test_safe_rejects_validation_and_test(self) -> None:
    for phase in ("selection_val", "final_test"):
        with self.assertRaisesRegex(SafePolicyError, "training-only"):
            apply_safe(self.image, self.settings, "sample", 42, 0, phase=phase)

def test_safe_repeats_for_same_sample_and_epoch(self) -> None:
    first = apply_safe(self.image, self.settings, "sample", 42, 3, phase="seed_train")
    second = apply_safe(self.image, self.settings, "sample", 42, 3, phase="seed_train")
    self.assertEqual(first.image.tobytes(), second.image.tobytes())
    self.assertEqual(first.realized, second.realized)

def test_mask_boxes_do_not_repeat_or_exceed_limit(self) -> None:
    result = apply_safe(self.image, self.always_mask, "sample", 42, 0, phase="seed_train")
    boxes = result.realized["mask_boxes"]
    self.assertEqual(len(boxes), len({tuple(box) for box in boxes}))
    self.assertLessEqual(result.realized["mask_fraction"], 0.75)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `conda run -n cya-techjam26 python -m unittest tests.test_safe_transforms -v`

Expected: FAIL because SAFE is absent.

- [ ] **Step 3: Implement SAFE with named sub-seeds**

```python
@dataclass(frozen=True)
class SafeSettings:
    input_size: int = 336
    flip_probability: float = 0.5
    color_jitter_fraction: float = 0.5
    rotation_degrees: float = 180.0
    mask_patch_size: int = 16
    mask_max_fraction: float = 0.75
    mask_probability: float = 0.5

def apply_safe(
    image: Image.Image, settings: SafeSettings, sample_id: str,
    project_seed: int, epoch: int, *, phase: str,
) -> TransformResult: ...
```

Require `phase == "seed_train"`. Derive named crop, flip, jitter, rotation, and mask sub-seeds
from `safe:{seed}:{epoch}:{sample_id}`. Apply the approved order. Rotate bilinearly with black
fill. Shuffle a 16 by 16 grid locally and choose mask boxes without replacement.

- [ ] **Step 4: Validate policy isolation**

Implement `validate_training_policy(config, *, phase: str) -> str`. Return `controlled` or
`safe`; reject both enabled, neither enabled, or SAFE outside `seed_train`. Test every branch.

- [ ] **Step 5: Run tests and commit**

```powershell
conda run -n cya-techjam26 python -m unittest tests.test_safe_transforms tests.test_preprocessing -v
conda run -n cya-techjam26 python -m ruff check src/cya_detector/transforms/safe.py tests/test_safe_transforms.py
git add src/cya_detector/transforms/safe.py src/cya_detector/transforms/__init__.py tests/test_safe_transforms.py
git commit -m "feat: add isolated safe training policy"
```

---

### Task 7: Expose the CLI and Verify Task 3 End to End

**Files:**
- Create: `scripts/materialize_transforms.py`
- Modify: `Makefile`
- Modify: `docs/planning/nextSteps.md`
- Modify: `tests/test_transform_materialization.py`

**Interfaces:**
- Consumes: `load_config`, `benchmark_cells`, `materialize_benchmarks`.
- Produces: materialization CLI plus `task3-test` and `task3-fixture` Make targets.

- [ ] **Step 1: Write a failing CLI smoke test**

```python
result = subprocess.run(
    [sys.executable, "scripts/materialize_transforms.py",
     "--input-manifest", str(self.manifest),
     "--output-root", str(self.root / "variants"),
     "--output-manifest", str(self.root / "variants.csv"),
     "--report", str(self.root / "report.json"),
     "--config", "configs/colab.json",
     "--cells", "resize_scale_0.5,noise_sigma_0.02"],
    cwd=REPO_ROOT, capture_output=True, text=True,
)
self.assertEqual(result.returncode, 0, result.stderr)
```

- [ ] **Step 2: Run the CLI test to verify failure**

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement the thin CLI and Make targets**

Parse all paths, comma-separated `--cells`, `--overwrite`, and `--config`. Resolve IDs against
`benchmark_cells`; call `materialize_benchmarks`; print image count, cell counts, manifest, and
report. Keep all pixel logic in the package.

```make
.PHONY: task3-test task3-fixture
TASK2_SELECTED_MANIFEST ?= $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv

task3-test:
	PYTHONPATH=src python -m unittest tests.test_config tests.test_benchmark_transforms tests.test_transform_materialization tests.test_preprocessing tests.test_controlled_sampler tests.test_safe_transforms -v

task3-fixture:
	python scripts/materialize_transforms.py --input-manifest $(TASK2_SELECTED_MANIFEST) --output-root $(ARTIFACT_ROOT)/task3/variants --output-manifest $(ARTIFACT_ROOT)/task3/transform_manifest.csv --report $(ARTIFACT_ROOT)/task3/transform_report.json --config configs/colab.json
```

The variable defaults to the fixed-Q96 pilot for a practical fixture run and must be overridden
with the selected Task 2 manifest when that decision is available.

- [ ] **Step 4: Run focused verification**

```powershell
conda run -n cya-techjam26 python -m unittest tests.test_benchmark_transforms tests.test_transform_materialization tests.test_preprocessing tests.test_controlled_sampler tests.test_safe_transforms -v
conda run -n cya-techjam26 python -m ruff check src/cya_detector/transforms scripts/materialize_transforms.py tests/test_benchmark_transforms.py tests/test_transform_materialization.py tests/test_preprocessing.py tests/test_controlled_sampler.py tests/test_safe_transforms.py
```

Expected: all focused tests pass and Ruff reports no Task 3 errors.

- [ ] **Step 5: Run the complete regression suite**

```powershell
conda run -n cya-techjam26 python -m unittest discover -s tests -v
conda run -n cya-techjam26 python scripts/smoke_check.py --config configs/colab.json
```

Expected: all tests pass and smoke validates configuration/imports. CPU-only CUDA output on the
local machine is informational.

- [ ] **Step 6: Update planning status from evidence**

Check only Task 3 bullets proven by tests. Leave real-data artifact or statistical-review work
unchecked until the Colab run exists. Add `make task3-test` and `make task3-fixture` beside the
Task 3 entry.

- [ ] **Step 7: Review and commit intended changes**

```powershell
git status --short
git diff --check
git diff --stat
git add scripts/materialize_transforms.py Makefile docs/planning/nextSteps.md tests/test_transform_materialization.py
git commit -m "feat: expose task 3 transform workflow"
```

- [ ] **Step 8: Preserve the Colab handoff**

After Task 2 selects the matched-clean policy, run in Colab:

```bash
cd /content/cya-techjam26
make task3-fixture ARTIFACT_ROOT=/content/cya-techjam26/artifacts
cp -r /content/cya-techjam26/artifacts/task3 /content/drive/MyDrive/cya-techjam26/artifacts/
```

Expected: Drive contains independent variant directories, manifest, and report. This real-data
run is an operational follow-up, not a prerequisite for the local implementation commit.
