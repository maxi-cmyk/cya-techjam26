# Task 9 Clean Texture Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean matched-image pilot that compares global-only RINE, local-only texture patches, and global-plus-local fusion without training or reevaluating the frozen CLIP backbone each epoch.

**Architecture:** Select at most four deterministic 112×112 source patches, convert them through the locked CLIP processor, and cache their frozen projected embeddings alongside compatible cached global RINE layer features. Train lightweight variant-specific heads for seeds 42/43/44, publish the shared-Drive artifact contract automatically, and make a deterministic clean-gate decision before any Task 3 robustness continuation.

**Tech Stack:** Python 3.11, PyTorch, Transformers CLIP, Pillow, NumPy, OpenCV, SciPy, `unittest`, Google Colab/Drive.

**Spec:** `docs/superpowers/specs/2026-08-30-task-9-clean-texture-pilot-design.md`

## Global Constraints

- Use only `seed_train` and `selection_val` from the existing fixed-Q96 matched-clean manifest.
- Never read `self_train_pool`, sealed `final_test`, source-original images, Task 3 variants, or Task 8B data.
- Freeze CLIP completely; train only RINE layer importance and Task 9 attention, projection, fusion, and classification parameters.
- Use exactly the configured seeds `[42, 43, 44]` for each of `global_only`, `local_only`, and `global_local`.
- Select at most four non-overlapping 112×112 source patches; the locked processor converts each to CLIP's 336×336 input.
- Keep large RINE and patch caches under `/content`; create Drive artifact directories only when publishing real outputs.
- Refuse completed-run overwrite unless the CLI receives `--overwrite`.
- The clean pilot can authorize a later robustness design but cannot retain Task 9 by itself.

---

## File Structure

- Modify `configs/colab.json`: freeze the Task 9 clean-pilot contract.
- Modify `src/cya_detector/config.py`: validate Task 9 keys, values, variants, and seeds.
- Modify `src/cya_detector/features/texture.py`: prepare padded source patches and stable cache identities.
- Create `src/cya_detector/models/texture.py`: masked attention and the three Task 9 heads.
- Create `src/cya_detector/training/texture_stage_d.py`: patch extraction, cache loading, head training, and per-seed publication.
- Create `src/cya_detector/evaluation/texture_gate.py`: compare variants and apply the clean gate.
- Create `scripts/train_texture_pilot.py`: one variant/seed extraction-and-training CLI.
- Create `scripts/compare_texture_pilot.py`: nine-run completeness check and comparison CLI.
- Create `notebooks/07_texture_stage_d.ipynb`: thin Colab launcher only.
- Modify `Makefile`: add fixture, run, seed-matrix, and comparison targets.
- Modify `docs/planning/nextSteps.md`: document the pilot commands and leave Task 9 incomplete until evidence exists.
- Modify `notebooks/README.md`: document notebook order and durable artifact copy behavior.
- Modify `tests/test_config.py` and `tests/test_features_texture.py`.
- Create `tests/test_texture_model.py`, `tests/test_texture_extraction.py`, `tests/test_texture_training.py`, and `tests/test_texture_gate.py`.

---

### Task 1: Freeze Configuration and Public Types

**Files:**
- Modify: `configs/colab.json`
- Modify: `src/cya_detector/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `validate_config(config: dict[str, Any]) -> None`
- Produces: `config["texture"]` with the exact keys and values below.

- [ ] **Step 1: Write failing configuration tests**

Add tests asserting the committed config contains exactly:

```python
expected = {
    "experiment_name": "clean_pilot_v1",
    "extractor_version": "texture-patches-v1",
    "patch_size": 112,
    "patch_count": 4,
    "aggregation": "masked_softmax_v1",
    "fusion_dimension": 256,
    "variants": ["global_only", "local_only", "global_local"],
    "seeds": [42, 43, 44],
}
self.assertEqual(self.config["texture"], expected)
```

Add table-driven failures for missing/unknown keys, boolean numeric values, `patch_size != 112`, `patch_count != 4`, nonpositive `fusion_dimension`, reordered/duplicate variants, reordered/duplicate seeds, and an experiment name other than `clean_pilot_v1`.

- [ ] **Step 2: Run the tests and verify the new contract fails**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_config -v
```

Expected: FAIL because `texture` is absent.

- [ ] **Step 3: Add the exact configuration and validation**

Add the JSON object above beside the existing `frequency` and `auxiliary` sections. In `config.py`, add `texture` to `REQUIRED_SECTIONS`, require the exact key set, reject booleans as integers, and freeze all approved values. Use explicit messages such as `Texture patch_size must remain 112` and `Texture variants must remain global_only/local_only/global_local`.

- [ ] **Step 4: Run configuration and bootstrap verification**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_config -v
python scripts/smoke_check.py --config configs/colab.json --allow-missing-dependencies
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add configs/colab.json src/cya_detector/config.py tests/test_config.py
git commit -m "feat: freeze task 9 clean pilot configuration"
```

---

### Task 2: Prepare Deterministic Local Patch Views

**Files:**
- Modify: `src/cya_detector/features/texture.py`
- Modify: `tests/test_features_texture.py`

**Interfaces:**
- Consumes: `select_texture_patches(image: np.ndarray, *, patch_size: int, top_k: int) -> PatchSelection`
- Produces:

```python
@dataclass(frozen=True)
class PreparedPatchViews:
    patches: tuple[np.ndarray, ...]
    patch_boxes: tuple[tuple[int, int, int, int], ...]
    availability_mask: tuple[bool, ...]
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]

def prepare_texture_patch_views(
    image: np.ndarray,
    *,
    patch_size: int,
    patch_count: int,
) -> PreparedPatchViews: ...

def texture_patch_cache_key(
    *,
    image_sha256: str,
    patch_boxes: tuple[tuple[int, int, int, int], ...],
    model_identifier: str,
    resolved_revision: str,
    preprocessing_version: str,
    extractor_version: str,
) -> str: ...
```

- [ ] **Step 1: Write failing patch-view tests**

Cover these literal fixtures:

```python
image = np.zeros((336, 336, 3), dtype=np.float32)
views = prepare_texture_patch_views(image, patch_size=112, patch_count=4)
self.assertEqual(len(views.patches), 4)
self.assertEqual(views.availability_mask, (True, True, True, True))
self.assertTrue(all(patch.shape == (112, 112, 3) for patch in views.patches))
```

Also assert:

- a 336×336 source exposes nine selector candidates and returns the ranked top four;
- a 100×101 source pads symmetrically to 112×112, with odd remainder on right/bottom;
- padding returns one real patch plus a four-position mask `(True, False, False, False)`;
- RGBA/grayscale and non-finite arrays are rejected or normalized according to `features.common` conventions;
- returned patches do not share writable memory with the source;
- cache keys repeat exactly and change for each listed identity field.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_features_texture -v
```

Expected: FAIL because the preparation and cache-key interfaces do not exist.

- [ ] **Step 3: Implement minimal patch preparation**

Pad only dimensions below `patch_size`. Call the existing selector on the padded array, crop selected boxes in ranked order, copy each crop, and append `False` mask entries without fake patch arrays until the mask length equals `patch_count`. Serialize the cache-key payload with sorted JSON and compact separators before SHA-256 hashing.

- [ ] **Step 4: Verify texture tests**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_features_texture -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/cya_detector/features/texture.py tests/test_features_texture.py
git commit -m "feat: prepare deterministic task 9 patch views"
```

---

### Task 3: Implement Masked Attention and Variant Heads

**Files:**
- Create: `src/cya_detector/models/texture.py`
- Create: `tests/test_texture_model.py`

**Interfaces:**
- Produces:

```python
TEXTURE_VARIANTS = ("global_only", "local_only", "global_local")

def masked_patch_weights(scores: Any, mask: Any) -> Any: ...

def build_texture_head(
    *,
    variant: str,
    layer_count: int,
    global_dimension: int,
    patch_dimension: int,
    fusion_dimension: int,
) -> Any: ...
```

Every head accepts:

```python
logits = model(
    global_features,  # [batch, layers, global_dimension]
    patch_features,   # [batch, patch_count, patch_dimension]
    patch_mask,       # [batch, patch_count], bool
)
```

and returns `[batch, 1]`. Inputs unused by a variant are still shape-validated so one training driver can serve all variants.

- [ ] **Step 1: Write failing model tests**

Test all three variants with batch size two, four RINE layers, global dimension eight, patch dimension six, and masks `[[1,1,1,1],[1,0,0,0]]`. Assert:

- output shape is `(2, 1)`;
- attention weights are finite, nonnegative, zero on absent patches, and sum to one per sample;
- changing masked patch feature values cannot change the output;
- an all-false mask raises `ValueError("Every sample requires at least one patch")`;
- bad global, patch, or mask shapes fail with explicit messages;
- unknown variants fail closed;
- parameters exist only in the constructed head and a separately supplied frozen encoder remains frozen.

- [ ] **Step 2: Run the new model tests and verify failure**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_model -v
```

Expected: FAIL because `cya_detector.models.texture` does not exist.

- [ ] **Step 3: Implement the lightweight heads**

Use the existing RINE normalization and softmax layer-importance formula for the global vector. Use `Linear(patch_dimension, fusion_dimension)`, `Tanh`, and `Linear(fusion_dimension, 1)` for patch scores. Apply masked softmax using the dtype minimum before softmax, then zero and renormalize masked weights defensively. Build:

- `global_only`: RINE aggregate to `Linear(global_dimension, 1)`;
- `local_only`: patch aggregate to `Linear(patch_dimension, 1)`;
- `global_local`: project global and local to `fusion_dimension`, apply GELU, concatenate, and classify with `Linear(2 * fusion_dimension, 1)`.

Expose `attention_weights(...)` and `global_importance_weights()` for reports without changing the forward return type.

- [ ] **Step 4: Run model and existing RINE tests**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_model tests.test_rine -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/cya_detector/models/texture.py tests/test_texture_model.py
git commit -m "feat: add task 9 attention and fusion heads"
```

---

### Task 4: Extract and Cache Frozen Global/Patch Features

**Files:**
- Create: `src/cya_detector/training/texture_stage_d.py`
- Create: `tests/test_texture_extraction.py`

**Interfaces:**
- Consumes: `LoadedClip`, `ManifestExample`, `extract_rine_features`, `prepare_texture_patch_views`, and `texture_patch_cache_key`.
- Produces:

```python
@dataclass(frozen=True)
class CachedTextureFeatures:
    example: ManifestExample
    global_cache_path: Path
    patch_cache_path: Path

def extract_texture_features(
    *,
    loaded_clip: LoadedClip,
    examples: list[ManifestExample],
    global_cache_root: Path,
    patch_cache_root: Path,
    matching_policy: str,
    preprocessing_version: str,
    rine_representation_version: str,
    texture_extractor_version: str,
    layers: tuple[int, ...],
    patch_size: int,
    patch_count: int,
    batch_size: int,
    device: str,
) -> tuple[list[CachedTextureFeatures], dict[str, Any]]: ...
```

Each patch cache file contains fixed tensors and metadata:

```python
{
    "patch_features": Tensor[patch_count, projection_dimension],
    "patch_mask": BoolTensor[patch_count],
    "patch_boxes": list[list[int]],
    "cache_contract": {...},
}
```

- [ ] **Step 1: Write failing extraction tests with a fake frozen encoder**

Use two temporary RGB images and a fake processor/model that returns deterministic projected embeddings. Assert:

- only `seed_train`/`selection_val` examples are accepted;
- missing SHA-256 fails before model invocation;
- first extraction writes atomic global and patch cache files;
- the second identical extraction reports cache hits and never calls the encoder;
- changed coordinates/version/revision force a miss;
- patch files have four rows with the correct boolean mask;
- non-finite encoder output prevents publication;
- report fields include counts, elapsed time, images/second, bytes, peak GPU memory, model revision, patch size/count, dimensions, and versions.

- [ ] **Step 2: Run extraction tests and verify failure**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_extraction -v
```

Expected: FAIL because the extraction module does not exist.

- [ ] **Step 3: Implement global reuse and patch batching**

Call `extract_rine_features` for the global cache contract. Decode each missing patch row once, prepare views, flatten available patches into batches for the locked processor, run `loaded_clip.model(...).image_embeds` in inference mode/autocast, and reconstruct fixed four-position tensors. Write `.tmp.pt` then replace. Validate cached metadata and tensor shapes before declaring a hit.

- [ ] **Step 4: Run extraction, texture, and RINE tests**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_extraction tests.test_features_texture tests.test_rine -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/cya_detector/training/texture_stage_d.py tests/test_texture_extraction.py
git commit -m "feat: cache frozen task 9 global and patch features"
```

---

### Task 5: Train Nine Runs, Publish Artifacts, and Apply the Gate

**Files:**
- Modify: `src/cya_detector/training/texture_stage_d.py`
- Create: `src/cya_detector/evaluation/texture_gate.py`
- Create: `scripts/train_texture_pilot.py`
- Create: `scripts/compare_texture_pilot.py`
- Create: `tests/test_texture_training.py`
- Create: `tests/test_texture_gate.py`

**Interfaces:**
- Produces:

```python
def train_texture_head(
    *,
    rows: list[CachedTextureFeatures],
    variant: str,
    seed: int,
    output_root: Path,
    overwrite: bool,
    run_configuration: dict[str, Any],
    **optimization: Any,
) -> dict[str, Any]: ...

def compare_texture_pilot(
    *,
    experiment_root: Path,
    seeds: tuple[int, ...],
    max_per_class_regression: float,
) -> dict[str, Any]: ...
```

- [ ] **Step 1: Write failing training/artifact tests**

Build tiny cached tensors for both labels in `seed_train` and `selection_val`. For every variant and seed, assert the trainer creates only on first output:

```text
<variant>/seed_<seed>/
├── checkpoints/best_clean.pt
├── checkpoints/latest.pt
├── predictions/selection_val.csv
├── reports/metrics.json
├── reports/training_history.json
└── metadata/run_metadata.json
```

Assert atomic JSON/CSV/checkpoint writes, finite-loss refusal, both-class requirement, exact split allowlist, deterministic repeatability, explicit overwrite requirement, and no `final_test` access.

- [ ] **Step 2: Write failing clean-gate tests**

Create nine fixture prediction files and assert:

```python
decision = compare_texture_pilot(
    experiment_root=root,
    seeds=(42, 43, 44),
    max_per_class_regression=0.01,
)
self.assertEqual(decision["decision"], "continue_to_robustness_design")
```

Add rejection fixtures for no mean accuracy improvement, authentic regression over 0.01, AI-generated regression over 0.01, no corrected global error, missing variant/seed, mismatched sample sets, and non-clean transform rows. Verify `comparison/` CSV/JSON plus root `metadata/artifact_manifest.json` contain file hashes and completion state.

- [ ] **Step 3: Run the new tests and verify failure**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_training tests.test_texture_gate -v
```

Expected: FAIL because trainer, gate, and CLIs do not exist.

- [ ] **Step 4: Implement cached-feature training**

Follow the existing RINE AdamW/BCE/warmup/accumulation/early-stopping contract. Load global and patch tensors once, train only the selected head, evaluate `selection_val` after each epoch, and write artifacts through temporary siblings. Map the CLI output root as:

```python
experiment_root = args.output_root / config["texture"]["experiment_name"]
run_root = experiment_root / args.variant / f"seed_{args.seed}"
```

Reject a nonconfigured variant/seed and a completed `run_root` unless `--overwrite` is set.

- [ ] **Step 5: Implement the comparison gate**

Read the best-clean predictions for all nine runs, require identical sample IDs/labels per seed, calculate ordinary and per-label metrics, count corrected/introduced errors, and apply the spec's strict conditions. Publish:

```text
comparison/global_local_comparison.json
comparison/per_seed_metrics.csv
comparison/latency_comparison.json
metadata/artifact_manifest.json
```

Use decision values `continue_to_robustness_design` and `reject_texture_clean_gate`. Do not add a Task 3 command here.

- [ ] **Step 6: Run training/gate tests and CLI help checks**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_training tests.test_texture_gate -v
python scripts/train_texture_pilot.py --help
python scripts/compare_texture_pilot.py --help
```

Expected: PASS; both help commands exit zero without importing pretrained weights.

- [ ] **Step 7: Commit Task 5**

```powershell
git add src/cya_detector/training/texture_stage_d.py src/cya_detector/evaluation/texture_gate.py scripts/train_texture_pilot.py scripts/compare_texture_pilot.py tests/test_texture_training.py tests/test_texture_gate.py
git commit -m "feat: train and gate task 9 clean texture pilot"
```

---

### Task 6: Colab Launcher, Commands, Documentation, and Full Verification

**Files:**
- Create: `notebooks/07_texture_stage_d.ipynb`
- Modify: `Makefile`
- Modify: `docs/planning/nextSteps.md`
- Modify: `notebooks/README.md`
- Test: all Task 9 tests and repository suite.

**Interfaces:**
- Produces Make targets `task9-test`, `task9-run`, `task9-matrix`, and `task9-compare`.

- [ ] **Step 1: Add a failing command-contract test**

In `tests/test_texture_training.py`, inspect `Makefile` and assert all four targets exist, `task9-matrix` invokes every configured variant and seed, caches default below `/content`, and output defaults below `$(ARTIFACT_ROOT)/task9`. Assert the notebook contains no embedded model/training implementation and references only scripts/Make targets.

- [ ] **Step 2: Run the contract test and verify failure**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_texture_training -v
```

Expected: FAIL because the targets/notebook are absent.

- [ ] **Step 3: Add Make targets and thin notebook**

Define overridable variables:

```make
TASK9_MANIFEST ?= $(ARTIFACT_ROOT)/task2/fixed_q96_manifest.csv
TASK9_OUTPUT_ROOT ?= $(ARTIFACT_ROOT)/task9
TASK9_GLOBAL_CACHE ?= /content/rine_feature_cache
TASK9_PATCH_CACHE ?= /content/texture_patch_cache
TASK9_VARIANT ?= global_only
TASK9_SEED ?= 42
```

`task9-run` calls the training CLI once. `task9-matrix` loops explicitly over the three variants and seeds without background concurrency, because simultaneous GPU extraction would duplicate model memory and race on shared caches. `task9-compare` runs only after nine successful runs. `task9-test` runs the focused Task 9 modules.

The notebook mounts Drive, installs the repository, verifies CUDA, stages the fixed-Q96 manifest/images locally, runs extraction/training on `/content`, and copies completed durable artifacts to the configured Drive artifact root. It must not create empty Drive directories before a run produces output.

- [ ] **Step 4: Update planning documentation**

Document the clean-only boundary, 112-pixel source patches, four-patch/five-view budget, three variants × three seeds, automatic shared-Drive structure, clean gate, and the fact that Task 3 robustness is a later continuation only after a pass. Do not mark the Task 9 training/evaluation boxes complete before a real Colab run.

- [ ] **Step 5: Run focused tests**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_config tests.test_features_texture tests.test_texture_model tests.test_texture_extraction tests.test_texture_training tests.test_texture_gate -v
```

Expected: PASS.

- [ ] **Step 6: Run lint and the full suite**

```powershell
python -m ruff check src/cya_detector/features/texture.py src/cya_detector/models/texture.py src/cya_detector/training/texture_stage_d.py src/cya_detector/evaluation/texture_gate.py scripts/train_texture_pilot.py scripts/compare_texture_pilot.py tests/test_features_texture.py tests/test_texture_model.py tests/test_texture_extraction.py tests/test_texture_training.py tests/test_texture_gate.py
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
python scripts/smoke_check.py --config configs/colab.json --allow-missing-dependencies
```

Expected: all commands PASS. Run filesystem-heavy tests with permission to create Windows temporary fixtures if the sandbox denies `%TEMP%`.

- [ ] **Step 7: Run a dependency-free fixture smoke**

Use the fake encoder fixture to execute extraction, one short epoch for all nine variant/seed combinations, comparison, and artifact publication under a temporary root. Verify no file is written beneath the real repository `artifacts/` or shared Drive during this test.

- [ ] **Step 8: Commit Task 6**

```powershell
git add Makefile notebooks/07_texture_stage_d.ipynb notebooks/README.md docs/planning/nextSteps.md tests/test_texture_training.py
git commit -m "docs: expose task 9 clean pilot workflow"
```

---

## Execution Notes for Speed

- Use one implementation worker per task, but do not run Tasks 2–5 concurrently because their interfaces are sequential and they edit overlapping files.
- Run specification review once after Task 1 and code-quality review once after Task 6; avoid a full dual-agent review after every small task unless a scoped test fails or an interface changes.
- Keep the first real Colab run to the existing fixed-Q96 pilot manifest.
- Run GPU extraction once, then train all nine lightweight runs from the same caches.
- Stop after the clean comparison if the decision is `reject_texture_clean_gate`.
- If the decision is `continue_to_robustness_design`, write a separate specification and cost estimate before materializing transformed Task 9 training inputs.
