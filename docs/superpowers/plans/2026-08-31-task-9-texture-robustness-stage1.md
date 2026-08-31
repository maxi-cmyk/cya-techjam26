# Task 9 Texture Robustness Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, frozen-checkpoint Stage-1 robustness evaluation that compares the clean Task 9 texture heads with both the internal global-only ablation and retained controlled RINE across nine texture-sensitive Task 3 transform cells and emits a deterministic continuation decision.

**Architecture:** Materialize transformed `selection_val` images directly from fixed-Q96 clean parents, extract transformed global/patch features once, and reuse them across all variants and seeds. Evaluate immutable best-clean checkpoints without training, join hash-verified controlled-RINE predictions on the identical parents and nine cells, then publish per-cell metrics and a hashed aggregate gate under a new Drive-safe artifact subtree.

**Tech Stack:** Python 3.11, PyTorch, Pillow, NumPy, existing Task 3 transform engine, existing frozen CLIP/RINE loader, `unittest`, Google Colab.

**Spec:** `docs/superpowers/specs/2026-08-31-task-9-texture-robustness-stage1-design.md`

## Global Constraints

- Input is fixed-Q96 matched-clean `selection_val` only; reject every other split/view/policy.
- Use exactly `jpeg_q90`, `jpeg_q70`, `jpeg_q50`, `jpeg_q30`, `blur_sigma_0.5`, `blur_sigma_1.0`, `blur_sigma_2.0`, `resize_scale_0.5`, and `resize_scale_0.25`.
- Every transformed row derives directly from clean; transform chaining is forbidden.
- Evaluate `global_only`, `local_only`, and `global_local` for seeds `42`, `43`, and `44`; `global_local` must pass against both `global_only` and the retained controlled-RINE parent. `local_only` is diagnostic.
- Restore the three controlled-RINE per-seed `predictions.csv` artifacts, require all 27 seed-cell partitions for the same nine cells and seeds, verify their source manifest/checkpoint/output hashes, and recompute the nine-cell comparator score. Do not substitute the persisted 14-cell aggregate (`0.9981`).
- Recompute patch selection from each transformed image; never reuse clean coordinates.
- Checkpoints, threshold, and clean calibration are immutable. No training or fine-tuning is permitted.
- Never read `seed_train`, `self_train_pool`, or sealed `final_test` in this continuation.
- Large transformed images and feature caches stay under `/content`; durable predictions/reports go below `artifacts/task9/clean_pilot_v1/robustness_stage1_v1`.
- Publish atomically, validate hashes before resuming, and write `metadata/artifact_manifest.json` last.
- A pass authorizes only the remaining Task 3 robustness cells, not final retention or final-test evaluation.

---

### Task 1: Freeze the Stage-1 Contract

**Files:**
- Modify: `configs/colab.json`
- Modify: `src/cya_detector/config.py`
- Create: `src/cya_detector/evaluation/texture_robustness.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_texture_robustness.py`

**Interfaces:**
- Consumes: `benchmark_cells(config)` and locked Task 9 variants/seeds.
- Produces: `STAGE1_CELL_IDS`, `RobustnessContract`, and `validate_robustness_contract(config) -> RobustnessContract`.

- [ ] **Step 1: Write failing configuration and contract tests**

```python
def test_texture_robustness_contract_is_exact_and_locked(self):
    contract = validate_robustness_contract(load_config(CONFIG_PATH))
    self.assertEqual(contract.cell_ids, STAGE1_CELL_IDS)
    self.assertEqual(contract.variants, ("global_only", "local_only", "global_local"))
    self.assertEqual(contract.seeds, (42, 43, 44))
    self.assertEqual(contract.controlling_comparators, ("global_only", "controlled_rine"))
    self.assertEqual(contract.aggregate_class_tolerance, 0.01)
    self.assertEqual(contract.worst_cell_tolerance, 0.03)

def test_contract_rejects_unknown_missing_or_deferred_cells(self):
    for bad in (("jpeg_q90",), STAGE1_CELL_IDS + ("noise_sigma_0.02",)):
        candidate = copy.deepcopy(self.config)
        candidate["texture_robustness_stage1"]["cell_ids"] = list(bad)
        with self.assertRaises(ConfigError):
            validate_config(candidate)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_config tests.test_texture_robustness -v"`

Expected: FAIL because the configuration section and robustness module do not exist.

- [ ] **Step 3: Add the exact configuration and immutable contract**

Add:

```json
"texture_robustness_stage1": {
  "experiment_name": "robustness_stage1_v1",
  "cell_ids": ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "blur_sigma_0.5", "blur_sigma_1.0", "blur_sigma_2.0", "resize_scale_0.5", "resize_scale_0.25"],
  "aggregate_class_tolerance": 0.01,
  "worst_cell_tolerance": 0.03
}
```

Define a frozen dataclass and fail closed if configuration values differ from the constants or refer to cells absent from `benchmark_cells(config)`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: PASS with no new warning or failure.

- [ ] **Step 5: Commit**

```powershell
git add configs/colab.json src/cya_detector/config.py src/cya_detector/evaluation/texture_robustness.py tests/test_config.py tests/test_texture_robustness.py
git commit -m "feat: freeze texture robustness stage 1 contract"
```

---

### Task 2: Materialize the Locked Selection Robustness Views

**Files:**
- Create: `scripts/materialize_texture_robustness.py`
- Modify: `src/cya_detector/evaluation/texture_robustness.py`
- Modify: `tests/test_texture_robustness.py`

**Interfaces:**
- Consumes: fixed-Q96 manifest and Task 3 `materialize_benchmarks`.
- Produces: `materialize_texture_stage1(*, input_manifest: Path, output_root: Path, output_manifest: Path, report_path: Path, config: dict, overwrite: bool = False) -> dict[str, Any]`.

- [ ] **Step 1: Write failing materialization-boundary tests**

```python
def test_materializes_exact_stage1_cells_directly_from_selection_clean(self):
    report = materialize_texture_stage1(**self.materialize_args)
    rows = read_manifest(self.output_manifest)
    self.assertEqual(set(report["cell_counts"]), set(STAGE1_CELL_IDS))
    self.assertEqual({row["split"] for row in rows}, {"selection_val"})
    self.assertEqual({row["image_view"] for row in rows}, {"benchmark"})
    self.assertTrue(all(row["parent_id"] in self.clean_parent_ids for row in rows))

def test_rejects_forbidden_split_or_chained_parent_before_writing(self):
    for field, value in (("split", "final_test"), ("image_view", "benchmark")):
        self.mutate_parent(field, value)
        with self.assertRaises(TextureRobustnessError):
            materialize_texture_stage1(**self.materialize_args)
        self.assertFalse(self.output_manifest.exists())
```

Also test fixed-Q96 policy, exact row count, deterministic reruns, parent/output hashes, and no partial CSV/JSON publication after a late failure.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_texture_robustness -v"`

Expected: FAIL because `materialize_texture_stage1` and the CLI are absent.

- [ ] **Step 3: Implement validation-first materialization and CLI**

Filter and validate parents before calling the existing Task 3 materializer. Select cells by exact ID from `benchmark_cells(config)` and assert the materializer report returns the exact cell set and expected `parent_count × 9` rows. Write output CSV and JSON through temporary siblings.

CLI contract:

```text
python scripts/materialize_texture_robustness.py \
  --input-manifest artifacts/task2/fixed_q96_manifest.csv \
  --output-root /content/task9_robustness_stage1/images \
  --output-manifest /content/task9_robustness_stage1/transformed_selection_val.csv \
  --report /content/task9_robustness_stage1/materialization_report.json
```

- [ ] **Step 4: Verify GREEN and compatibility**

Run the Step 2 command, then:

`conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_benchmark_transforms tests.test_transform_materialization -v"`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/materialize_texture_robustness.py src/cya_detector/evaluation/texture_robustness.py tests/test_texture_robustness.py
git commit -m "feat: materialize texture robustness stage 1 views"
```

---

### Task 3: Extract Transformed Features and Evaluate Frozen Checkpoints

**Files:**
- Create: `scripts/evaluate_texture_robustness.py`
- Modify: `src/cya_detector/evaluation/texture_robustness.py`
- Modify: `tests/test_texture_robustness.py`

**Interfaces:**
- Consumes: transformed manifest, materialization report, nine clean run roots, frozen CLIP/RINE loader.
- Produces: `evaluate_texture_stage1(*, transformed_manifest: Path, clean_experiment_root: Path, cache_root: Path, output_root: Path, config: dict, device: str, overwrite: bool = False) -> dict[str, Any]` and 81 prediction CSV slices.

- [ ] **Step 1: Write failing extraction/evaluation tests**

Use a fake encoder and tiny real checkpoints. Assert transformed patch boxes differ when transformed energy ordering differs, while repeat evaluation is byte-identical. Assert the encoder is called once per transformed image set rather than once per seed/variant.

```python
def test_evaluates_all_81_slices_with_shared_transformed_features(self):
    summary = evaluate_texture_stage1(**self.evaluation_args)
    self.assertEqual(summary["completed_slices"], 81)
    expected_views = sum(1 + row.available_patch_count for row in self.expected_rows)
    self.assertEqual(self.fake_encoder.encoded_image_count, expected_views)

def test_never_reuses_clean_patch_coordinates(self):
    evaluate_texture_stage1(**self.evaluation_args)
    self.assertEqual(self.recorded_patch_boxes, expected_boxes_from_transformed_pixels)
```

Also test immutable checkpoint hashes, exact sample/label alignment, fixed threshold/calibration, forbidden splits, cache identity fields, non-finite refusal, atomic slice writes, and hash-verified resume.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 2 focused command. Expected: FAIL because frozen robustness evaluation is absent.

- [ ] **Step 3: Implement transformed extraction and inference**

Reuse the established texture preparation and model-head builders, but introduce an evaluation-specific cache contract that requires benchmark view, clean parent provenance, and exact cell identity. Load `checkpoints/best_clean.pt` only; hash before and after evaluation and refuse changes. Produce one CSV per variant/seed/cell with the fields frozen by the spec.

- [ ] **Step 4: Run focused and Task 9 compatibility tests**

```powershell
conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_texture_robustness tests.test_texture_extraction tests.test_texture_training -v"
python scripts/evaluate_texture_robustness.py --help
```

Expected: PASS; help exits zero without loading model weights.

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_texture_robustness.py src/cya_detector/evaluation/texture_robustness.py tests/test_texture_robustness.py
git commit -m "feat: evaluate frozen texture heads under transforms"
```

---

### Task 4: Verify Controlled RINE and Compute the Locked Robustness Gate

**Files:**
- Create: `scripts/compare_texture_robustness.py`
- Modify: `src/cya_detector/evaluation/texture_robustness.py`
- Modify: `tests/test_texture_robustness.py`

**Interfaces:**
- Consumes: clean comparison artifacts, 81 validated texture prediction slices, and three persisted controlled-RINE per-seed prediction files containing all 27 required seed-cell partitions from the completed Task 3 robustness run.
- Produces: `compare_texture_stage1(*, clean_experiment_root: Path, robustness_root: Path, controlled_rine_root: Path, config: dict) -> dict[str, Any]` and the complete report/manifest tree.

- [ ] **Step 1: Write failing metric and gate tests**

```python
def test_retains_only_when_locked_score_and_every_regression_gate_pass(self):
    report = compare_texture_stage1(**self.comparison_args)
    self.assertEqual(report["decision"], "retain_texture_for_full_robustness")
    self.assertGreater(report["comparators"]["global_only"]["locked_score_delta"], 0.0)
    self.assertGreater(report["comparators"]["controlled_rine"]["locked_score_delta"], 0.0)

def test_recomputes_controlled_rine_on_the_exact_nine_cell_subset(self):
    report = compare_texture_stage1(**self.comparison_args)
    self.assertEqual(report["comparators"]["controlled_rine"]["cell_count"], 9)

def test_missing_or_mismatched_controlled_rine_blocks_comparison(self):
    self.remove_one_controlled_rine_slice()
    with self.assertRaises(TextureRobustnessPrerequisiteError):
        compare_texture_stage1(**self.comparison_args)

def test_rejects_each_gate_independently_against_either_comparator(self):
    for comparator in ("global_only", "controlled_rine"):
        for mutation in self.gate_mutations(comparator):
            self.reset_comparison_fixture()
            mutation()
            self.assertEqual(
                compare_texture_stage1(**self.comparison_args)["decision"],
                "reject_texture_robustness_stage1",
            )
```

Test equal macro weighting, paired bootstrap determinism, ECE/confusion matrices, corrected/introduced errors against both comparators, local-only diagnostic reporting, latency/memory finiteness, exact 81 texture-slice plus 27 controlled-RINE-partition completeness, comparator provenance hashes, and manifest-last hash publication.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 2 focused command. Expected: FAIL because comparison is absent.

- [ ] **Step 3: Implement metrics, deterministic gate, and publication**

Validate the controlled-RINE source manifest, checkpoint, prediction hashes, exact seeds/cells, and sample/label alignment before computing any decision. Recompute its clean and nine-cell robustness means from per-sample artifacts. Compute texture means exactly as specified and require `global_local` to pass every improvement and regression condition against both controlling comparators. Average seeds within cells for worst-cell checks. Use the existing evaluation metric helpers where their label semantics match. Write `per_cell_metrics.csv`, `per_seed_robustness.csv`, `robustness_comparison.json`, `latency_and_memory.json`, `failure_analysis.csv`, then hash every required durable artifact and comparator reference and publish `artifact_manifest.json` last.

- [ ] **Step 4: Verify GREEN and CLI**

```powershell
conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_texture_robustness tests.test_texture_gate -v"
python scripts/compare_texture_robustness.py --help
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/compare_texture_robustness.py src/cya_detector/evaluation/texture_robustness.py tests/test_texture_robustness.py
git commit -m "feat: gate texture robustness stage 1"
```

---

### Task 5: Colab Launcher, Commands, Documentation, and Full Verification

**Files:**
- Create: `notebooks/08_texture_robustness_stage1.ipynb`
- Modify: `Makefile`
- Modify: `notebooks/README.md`
- Modify: `docs/planning/nextSteps.md`
- Modify: `tests/test_texture_robustness.py`

**Interfaces:**
- Produces Make targets `task9-robustness-test`, `task9-robustness-materialize`, `task9-robustness-evaluate`, and `task9-robustness-compare`.

- [ ] **Step 1: Write failing command/notebook contract tests**

Assert all targets exist; the notebook contains no transform/model/metric/gate implementation; `/content` holds images/caches; Drive receives only completed durable artifacts; checkpoint restore covers all variants/seeds; and compare follows successful materialize/evaluate steps.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 2 focused command. Expected: FAIL because launcher and targets are absent.

- [ ] **Step 3: Add sequential Make targets and thin resumable notebook**

Use overridable variables rooted at the clean Drive artifacts and `/content`. The notebook mounts Drive, checks out the exact branch/commit, restores the Task 2 bundle, nine clean runs, and the controlled-RINE manifests/checkpoints/predictions required for the exact nine-cell comparison; invokes the three CLIs; inspects an initial slice; resumes verified predictions; and copies completed report artifacts only. It must never create a completed marker before comparison succeeds.

- [ ] **Step 4: Update documentation with the real clean result and pending robustness status**

Record clean means (`global_only` 0.993939, `local_only` 0.953535, `global_local` 1.0), controlled RINE (`1.0` clean and `0.9981` locked across the full 14-cell matrix), the requirement to recompute its nine-cell subset, the verified clean decision, the Stage-1 matrix/gate, and the fact that final retention/final test remain pending. Do not mark robustness complete before the real Colab run.

- [ ] **Step 5: Run focused tests, lint, and smoke checks**

```powershell
conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_config tests.test_benchmark_transforms tests.test_transform_materialization tests.test_features_texture tests.test_texture_model tests.test_texture_extraction tests.test_texture_training tests.test_texture_gate tests.test_texture_robustness -v"
python -m ruff check src/cya_detector/evaluation/texture_robustness.py scripts/materialize_texture_robustness.py scripts/evaluate_texture_robustness.py scripts/compare_texture_robustness.py tests/test_texture_robustness.py
conda run -n cya-techjam26 python scripts/smoke_check.py --config configs/colab.json --allow-missing-dependencies
```

Expected: no new failures beyond the separately ledgered pre-existing Windows Task 8B assertion; lint for new files is clean; smoke passes.

- [ ] **Step 6: Run full suite and dependency-free end-to-end fixture**

```powershell
conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest discover -s tests -v"
conda run -n cya-techjam26 cmd /d /c "set PYTHONPATH=src&&python -m unittest tests.test_texture_robustness.TextureRobustnessFixtureSmokeTests -v"
```

The fixture must execute materialization → shared fake extraction → 81 frozen texture evaluations → three hash-verified controlled-RINE files containing 27 seed-cell partitions → two-comparator gate → hashed publication under a temporary root and prove no writes beneath real `artifacts/` or Drive paths.

- [ ] **Step 7: Commit**

```powershell
git add Makefile notebooks/08_texture_robustness_stage1.ipynb notebooks/README.md docs/planning/nextSteps.md tests/test_texture_robustness.py
git commit -m "docs: expose texture robustness stage 1 workflow"
```
