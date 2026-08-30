from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from cya_detector.data.dataset import ManifestExample
from cya_detector.evaluation.texture_gate import compare_texture_pilot
from cya_detector.models.clip_baseline import LoadedClip
from cya_detector.training.texture_stage_d import (
    APPROVED_MATCHING_POLICY,
    LOCKED_TEXTURE_SEEDS,
    LOCKED_TEXTURE_VARIANTS,
    CachedTextureFeatures,
    extract_texture_features,
    train_texture_head,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"
MAKEFILE_PATH = REPO_ROOT / "Makefile"
NOTEBOOK_PATH = REPO_ROOT / "notebooks/07_texture_stage_d.ipynb"

_MAKE_TARGET_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)\s*:(?!=)(?P<deps>[^\n]*)\n(?P<recipe>(?:\t[^\n]*\n?)*)",
    re.MULTILINE,
)


def _parse_make_targets(text: str) -> dict[str, str]:
    """Map each Makefile target name to its tab-indented recipe body."""

    return {match.group("name"): match.group("recipe") for match in _MAKE_TARGET_PATTERN.finditer(text)}


class TextureTrainingTests(unittest.TestCase):
    """The breaks caught here are accidental cache use, partial publishing, and reruns."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"texture-training-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.rows = self._cached_rows()
        self.configuration = {"texture": {"fusion_dimension": 4}}

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _cached_rows(self, *, include_final_test: bool = False, nonfinite: bool = False) -> list[CachedTextureFeatures]:
        rows: list[CachedTextureFeatures] = []
        specifications = [
            ("train-authentic", "seed_train", "authentic", -1.0),
            ("train-ai", "seed_train", "ai_generated", 1.0),
            ("val-authentic", "selection_val", "authentic", -0.8),
            ("val-ai", "selection_val", "ai_generated", 0.8),
        ]
        if include_final_test:
            specifications.append(("final", "final_test", "authentic", -0.5))
        for index, (sample_id, split, label, value) in enumerate(specifications):
            example = ManifestExample(
                sample_id=sample_id,
                source_id=f"source-{sample_id}",
                parent_id=f"parent-{sample_id}",
                image_path=self.root / f"{sample_id}.png",
                sha256=f"sha-{sample_id}",
                label=label,
                split=split,
                image_view="matched_clean",
                transform="clean",
                transform_parameter="",
                metadata={
                    "dataset_name": "fixture",
                    "generator_name": "unknown",
                    "generator_checkpoint": "unknown",
                    "capture_source": "fixture",
                },
            )
            global_path = self.root / "cache" / f"{sample_id}.global.pt"
            patch_path = self.root / "cache" / f"{sample_id}.patch.pt"
            global_path.parent.mkdir(parents=True, exist_ok=True)
            global_tensor = torch.full((2, 3), value + index * 0.01, dtype=torch.float32)
            if nonfinite and sample_id == "train-authentic":
                global_tensor[0, 0] = torch.nan
            torch.save(global_tensor, global_path)
            global_path.with_suffix(".meta.json").write_text(
                json.dumps({"matching_policy": "fixed_q96"}), encoding="utf-8"
            )
            torch.save(
                {
                    "patch_features": torch.full((4, 2), value, dtype=torch.float32),
                    "patch_mask": torch.tensor([True, True, False, False]),
                    "cache_contract": {"matching_policy": "fixed_q96"},
                },
                patch_path,
            )
            rows.append(CachedTextureFeatures(example, global_path, patch_path, "fixed_q96"))
        return rows

    def _train(self, *, root: Path, rows: list[CachedTextureFeatures] | None = None, variant: str = "global_local", seed: int = 42, overwrite: bool = False, **changes):
        from cya_detector.training.texture_stage_d import train_texture_head

        task4_extraction_report = changes.pop(
            "task4_extraction_report", {"elapsed_seconds": 0.01, "peak_gpu_memory_bytes": 0}
        )
        learning_rate = changes.pop("learning_rate", 0.01)

        return train_texture_head(
            rows=self.rows if rows is None else rows,
            variant=variant,
            seed=seed,
            output_root=root,
            overwrite=overwrite,
            run_configuration=self.configuration,
            device="cpu",
            learning_rate=learning_rate,
            weight_decay=0.0,
            warmup_fraction=0.0,
            max_epochs=2,
            early_stopping_patience=2,
            physical_batch_size=2,
            effective_batch_size=2,
            threshold=0.5,
            task4_extraction_report=task4_extraction_report,
            **changes,
        )

    def test_every_configured_head_run_publishes_a_complete_atomic_artifact_set(self) -> None:
        expected_relative = {
            Path("checkpoints/best_clean.pt"),
            Path("checkpoints/latest.pt"),
            Path("predictions/selection_val.csv"),
            Path("reports/metrics.json"),
            Path("reports/training_history.json"),
            Path("metadata/run_metadata.json"),
        }
        for variant in ("global_only", "local_only", "global_local"):
            for seed in (42, 43, 44):
                with self.subTest(variant=variant, seed=seed):
                    output_root = self.root / "runs"
                    run_root = output_root / variant / f"seed_{seed}"
                    self.assertFalse(run_root.exists())
                    summary = self._train(root=output_root, variant=variant, seed=seed)
                    self.assertEqual(summary["status"], "completed")
                    self.assertEqual({path.relative_to(run_root) for path in run_root.rglob("*") if path.is_file()}, expected_relative)
                    self.assertFalse(any(".tmp" in path.name for path in output_root.rglob("*")))
                    self.assertEqual(json.loads((run_root / "reports/metrics.json").read_text())["selection_split"], "selection_val")
                    with (run_root / "predictions/selection_val.csv").open(newline="", encoding="utf-8") as handle:
                        prediction_rows = list(csv.DictReader(handle))
                    self.assertTrue(prediction_rows)
                    self.assertTrue(
                        all(row["matching_policy"] == APPROVED_MATCHING_POLICY for row in prediction_rows)
                    )

    def test_completed_run_requires_explicit_overwrite_and_repeat_is_deterministic(self) -> None:
        first_root = self.root / "first"
        second_root = self.root / "second"
        self._train(root=first_root)
        with self.assertRaisesRegex(FileExistsError, "completed"):
            self._train(root=first_root)
        self._train(root=second_root)
        first = first_root / "global_local" / "seed_42" / "predictions" / "selection_val.csv"
        second = second_root / "global_local" / "seed_42" / "predictions" / "selection_val.csv"
        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), hashlib.sha256(second.read_bytes()).hexdigest())
        self._train(root=first_root, overwrite=True)

    def test_failed_overwrite_keeps_the_completed_canonical_run(self) -> None:
        output_root = self.root / "interrupted-overwrite"
        first = self._train(root=output_root)
        run_root = Path(first["run_root"])
        metadata_path = run_root / "metadata" / "run_metadata.json"
        original_metadata = json.loads(metadata_path.read_text())
        from cya_detector.training import texture_stage_d

        original_replace = texture_stage_d._replace_published_file
        replacement_count = 0

        def interrupt_metadata_commit(source: Path, destination: Path) -> None:
            nonlocal replacement_count
            replacement_count += 1
            if replacement_count == 6:
                raise OSError("interrupted replacement")
            original_replace(source, destination)

        with patch(
            "cya_detector.training.texture_stage_d._replace_published_file",
            side_effect=interrupt_metadata_commit,
        ):
            with self.assertRaisesRegex(OSError, "interrupted replacement"):
                self._train(
                    root=output_root,
                    overwrite=True,
                    task4_extraction_report={"elapsed_seconds": 2.0, "peak_gpu_memory_bytes": 0},
                )
        self.assertTrue(metadata_path.is_file())
        self.assertEqual(json.loads(metadata_path.read_text()), original_metadata)
        self.assertIn("artifact_sha256", original_metadata)
        self.assertTrue(any(
            hashlib.sha256((run_root / relative).read_bytes()).hexdigest() != digest
            for relative, digest in original_metadata["artifact_sha256"].items()
        ))

    def test_rejects_nonfinite_features_before_creating_a_run_directory(self) -> None:
        output_root = self.root / "nonfinite-runs"
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._train(root=output_root, rows=self._cached_rows(nonfinite=True))
        self.assertFalse(output_root.exists())

    def test_refuses_nonfinite_model_outputs_without_publishing(self) -> None:
        class NaNHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, global_features, patch_features, patch_mask):
                return global_features[:, :1, :1].reshape(-1, 1) * self.weight * float("nan")

        output_root = self.root / "nan-loss-runs"
        with patch("cya_detector.training.texture_stage_d.build_texture_head", return_value=NaNHead()):
            with self.assertRaisesRegex(ValueError, "non-finite training logits"):
                self._train(root=output_root)
        self.assertFalse((output_root / "global_local" / "seed_42").exists())
        self.assertFalse(any(".tmp" in path.name for path in output_root.rglob("*")))

    def test_rejects_nonclean_and_final_test_rows_without_publishing_or_reading_them(self) -> None:
        output_root = self.root / "invalid-runs"
        rows_with_final = self._cached_rows(include_final_test=True)
        final_row = next(row for row in rows_with_final if row.example.split == "final_test")
        final_row.global_cache_path.unlink()
        final_row.patch_cache_path.unlink()
        with self.assertRaisesRegex(ValueError, "seed_train and selection_val"):
            self._train(root=output_root, rows=rows_with_final)
        invalid = self._cached_rows()
        invalid[0] = CachedTextureFeatures(
            example=invalid[0].example.__class__(
                **{**invalid[0].example.__dict__, "transform": "jpeg"}
            ),
            global_cache_path=invalid[0].global_cache_path,
            patch_cache_path=invalid[0].patch_cache_path,
            matching_policy=invalid[0].matching_policy,
        )
        with self.assertRaisesRegex(ValueError, "matched-clean"):
            self._train(root=output_root, rows=invalid)
        self.assertFalse(output_root.exists())

    def test_requires_both_training_labels_and_configured_variant_and_seed(self) -> None:
        only_authentic = [row for row in self.rows if row.example.label == "authentic"]
        with self.assertRaisesRegex(ValueError, "both classes"):
            self._train(root=self.root / "bad-labels", rows=only_authentic)
        with self.assertRaisesRegex(ValueError, "configured texture variant"):
            self._train(root=self.root / "bad-variant", variant="unknown")
        with self.assertRaisesRegex(ValueError, "configured texture seed"):
            self._train(root=self.root / "bad-seed", seed=45)

    def test_public_boundary_keeps_the_locked_seed_set_even_if_caller_configuration_changes(self) -> None:
        original = self.configuration
        self.configuration = {"texture": {"variants": ["global_local"], "seeds": [45], "fusion_dimension": 4}}
        try:
            with self.assertRaisesRegex(ValueError, "configured texture seed"):
                self._train(root=self.root / "caller-seed", seed=45)
        finally:
            self.configuration = original

    def test_requires_fixed_q96_cached_provenance_before_loading_features(self) -> None:
        rows = list(self.rows)
        object.__setattr__(rows[0], "matching_policy", "uniform_q95_q100")
        with self.assertRaisesRegex(ValueError, "fixed_q96"):
            self._train(root=self.root / "uniform", rows=rows)

    def test_rejects_spoofed_wrapper_provenance_when_cache_contract_is_not_fixed_q96(self) -> None:
        rows = self._cached_rows()
        rows[0].global_cache_path.with_suffix(".meta.json").write_text(
            json.dumps({"matching_policy": "uniform_q95_q100"}), encoding="utf-8"
        )
        payload = torch.load(rows[0].patch_cache_path, weights_only=True)
        payload["cache_contract"]["matching_policy"] = "uniform_q95_q100"
        torch.save(payload, rows[0].patch_cache_path)
        with self.assertRaisesRegex(ValueError, "fixed_q96"):
            self._train(root=self.root / "spoofed-wrapper", rows=rows)

    def test_missing_cached_provenance_fails_closed(self) -> None:
        rows = [
            CachedTextureFeatures(row.example, row.global_cache_path, row.patch_cache_path)
            for row in self.rows
        ]
        with self.assertRaisesRegex(ValueError, "fixed_q96"):
            self._train(root=self.root / "missing-policy", rows=rows)

    def test_rejects_nonfinite_masked_cache_entries(self) -> None:
        rows = self._cached_rows()
        payload = torch.load(rows[0].patch_cache_path, weights_only=True)
        payload["patch_features"][2, 0] = torch.nan
        torch.save(payload, rows[0].patch_cache_path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._train(root=self.root / "masked-nonfinite", rows=rows)

    def test_records_real_inference_measurements_and_task4_extraction_report(self) -> None:
        summary = self._train(
            root=self.root / "measured",
            task4_extraction_report={"elapsed_seconds": 1.25, "peak_gpu_memory_bytes": 0},
        )
        metrics = json.loads((Path(summary["run_root"]) / "reports" / "metrics.json").read_text())
        self.assertGreater(metrics["inference"]["latency_seconds"], 0.0)
        self.assertGreaterEqual(metrics["inference"]["peak_memory_bytes"], 0)
        self.assertEqual(metrics["task4_extraction"]["elapsed_seconds"], 1.25)

    def test_writes_a_trainable_payload_from_task4_cache_rows(self) -> None:
        from cya_detector.training.texture_stage_d import (
            read_cached_texture_features_payload,
            write_cached_texture_features_payload,
        )

        payload_path = self.root / "task4-cache-rows.json"
        report = {"elapsed_seconds": 1.25, "peak_gpu_memory_bytes": 0}
        write_cached_texture_features_payload(
            payload_path, rows=self.rows, task4_extraction_report=report
        )
        rows, restored_report = read_cached_texture_features_payload(payload_path)
        self.assertEqual(restored_report, report)
        self.assertEqual(rows, self.rows)
        self.assertEqual(json.loads(payload_path.read_text())["matching_policy"], "fixed_q96")

    def test_cuda_determinism_fails_closed_when_torch_cannot_guarantee_it(self) -> None:
        def deterministic_only(enabled: bool, *, warn_only: bool) -> None:
            if warn_only:
                return
            raise RuntimeError("determinism unavailable")

        with patch("torch.use_deterministic_algorithms", side_effect=deterministic_only):
            with self.assertRaisesRegex(RuntimeError, "determinism unavailable"):
                self._train(root=self.root / "determinism")


class TextureCommandContractTests(unittest.TestCase):
    """The break caught here is a launcher that reimplements training instead of calling the locked CLIs."""

    def setUp(self) -> None:
        self.makefile_text = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.targets = _parse_make_targets(self.makefile_text)
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_makefile_declares_all_four_task9_targets(self) -> None:
        for target in ("task9-test", "task9-run", "task9-matrix", "task9-compare"):
            with self.subTest(target=target):
                self.assertIn(target, self.targets, f"Makefile is missing target {target!r}")

    def test_task9_matrix_invokes_every_configured_variant_and_seed_combination(self) -> None:
        recipe = self.targets.get("task9-matrix", "")
        self.assertTrue(recipe, "task9-matrix has no recipe body")
        for variant in self.config["texture"]["variants"]:
            with self.subTest(variant=variant):
                self.assertIn(variant, recipe)
        for seed in self.config["texture"]["seeds"]:
            with self.subTest(seed=seed):
                self.assertIn(str(seed), recipe)

    def test_task9_caches_default_under_content_and_output_defaults_under_artifact_root(self) -> None:
        self.assertRegex(self.makefile_text, r"(?m)^TASK9_GLOBAL_CACHE\s*\?=\s*/content(/|\s|$)")
        self.assertRegex(self.makefile_text, r"(?m)^TASK9_PATCH_CACHE\s*\?=\s*/content(/|\s|$)")
        self.assertRegex(
            self.makefile_text,
            r"(?m)^TASK9_OUTPUT_ROOT\s*\?=\s*\$\(ARTIFACT_ROOT\)/task9\s*$",
        )

    def test_task9_run_and_compare_targets_reference_the_locked_training_and_gate_clis(self) -> None:
        run_recipe = self.targets.get("task9-run", "")
        self.assertIn("train_texture_pilot.py", run_recipe)
        compare_recipe = self.targets.get("task9-compare", "")
        self.assertIn("compare_texture_pilot.py", compare_recipe)

    def test_notebook_is_a_thin_launcher_with_no_inlined_model_or_training_implementation(self) -> None:
        self.assertTrue(NOTEBOOK_PATH.is_file(), f"{NOTEBOOK_PATH} does not exist")
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        self.assertTrue(code_cells, "notebook has no code cells")
        source_text = "\n".join("".join(cell["source"]) for cell in code_cells)

        forbidden_patterns = (
            r"class\s+\w+\s*\(",
            r"nn\.Module",
            r"nn\.Linear",
            r"nn\.Sequential",
            r"def\s+forward\s*\(",
            r"def\s+train_\w*\s*\(",
            r"loss\.backward\(\)",
            r"optimizer\.step\(\)",
            r"zero_grad\(\)",
            r"BCEWithLogitsLoss",
            r"torch\.optim",
            r"masked_patch_weights",
            r"build_texture_head",
            r"AdamW",
        )
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertNotRegex(source_text, pattern)

        for script in ("extract_texture_features.py", "train_texture_pilot.py", "compare_texture_pilot.py"):
            with self.subTest(script=script):
                self.assertIn(script, source_text)


class _FixtureProcessor:
    """A dependency-free stand-in for the locked CLIP processor."""

    def __call__(self, *, images, return_tensors: str):
        array = torch.as_tensor(np.array(images.convert("RGB"), copy=True), dtype=torch.float32)
        pixels = array.permute(2, 0, 1).unsqueeze(0) / 255.0
        return {"pixel_values": torch.nn.functional.interpolate(pixels, size=(16, 16))}


class _FixtureEncoder:
    """A dependency-free stand-in for the frozen CLIP vision tower."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=3, projection_dim=2)
        self.calls = 0

    def __call__(self, *, pixel_values, output_hidden_states=False, return_dict=False):
        self.calls += 1
        count = pixel_values.shape[0]
        embeds = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
        states = tuple(torch.full((count, 1, 3), float(layer)) for layer in range(3))
        return SimpleNamespace(image_embeds=embeds, hidden_states=states)


class TextureFixtureSmokeTests(unittest.TestCase):
    """The break caught here is any pipeline stage that requires pretrained weights, a GPU, or
    that reaches outside its declared temporary root into the real repository artifacts or a
    Drive-like path."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"texture-e2e-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _snapshot(directory: Path) -> set[tuple[str, int]]:
        if not directory.is_dir():
            return set()
        return {
            (str(path.relative_to(directory)), path.stat().st_size)
            for path in directory.rglob("*")
            if path.is_file()
        }

    def _examples(self) -> list[ManifestExample]:
        specifications = [
            ("train-authentic", "seed_train", "authentic"),
            ("train-ai", "seed_train", "ai_generated"),
            ("val-authentic", "selection_val", "authentic"),
            ("val-ai", "selection_val", "ai_generated"),
        ]
        examples: list[ManifestExample] = []
        for sample_id, split, label in specifications:
            image_path = self.root / "images" / f"{sample_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
            examples.append(
                ManifestExample(
                    sample_id=sample_id,
                    source_id=f"source-{sample_id}",
                    parent_id=f"parent-{sample_id}",
                    image_path=image_path,
                    sha256=f"sha-{sample_id}",
                    label=label,
                    split=split,
                    image_view="matched_clean",
                    transform="clean",
                    transform_parameter="",
                    metadata={
                        "dataset_name": "fixture",
                        "generator_name": "unknown",
                        "generator_checkpoint": "unknown",
                        "capture_source": "fixture",
                    },
                )
            )
        return examples

    def test_full_pipeline_runs_under_a_temporary_root_without_touching_real_artifacts_or_drive(self) -> None:
        artifacts_root = REPO_ROOT / "artifacts"
        baseline_artifacts = self._snapshot(artifacts_root)

        encoder = _FixtureEncoder()
        loaded_clip = LoadedClip(encoder, _FixtureProcessor(), "fixture-model", "requested", "resolved", 2)
        rows, extraction_report = extract_texture_features(
            loaded_clip=loaded_clip,
            examples=self._examples(),
            global_cache_root=self.root / "global_cache",
            patch_cache_root=self.root / "patch_cache",
            matching_policy="fixed_q96",
            preprocessing_version="prep-v1",
            rine_representation_version="rine-v1",
            texture_extractor_version="texture-v1",
            layers=(1, 2),
            patch_size=16,
            patch_count=4,
            batch_size=2,
            device="cpu",
        )
        self.assertEqual(len(rows), 4)
        self.assertGreater(encoder.calls, 0)

        output_root = self.root / "runs"
        for variant in LOCKED_TEXTURE_VARIANTS:
            for seed in LOCKED_TEXTURE_SEEDS:
                with self.subTest(variant=variant, seed=seed):
                    summary = train_texture_head(
                        rows=rows,
                        variant=variant,
                        seed=seed,
                        output_root=output_root,
                        overwrite=False,
                        run_configuration={"texture": {"fusion_dimension": 4}},
                        device="cpu",
                        learning_rate=0.01,
                        weight_decay=0.0,
                        warmup_fraction=0.0,
                        max_epochs=1,
                        early_stopping_patience=1,
                        physical_batch_size=2,
                        effective_batch_size=2,
                        threshold=0.5,
                        task4_extraction_report=extraction_report,
                    )
                    self.assertEqual(summary["status"], "completed")

        decision = compare_texture_pilot(
            experiment_root=output_root,
            seeds=LOCKED_TEXTURE_SEEDS,
            max_per_class_regression=0.99,
        )
        self.assertIn(decision["decision"], ("continue_to_robustness_design", "reject_texture_clean_gate"))
        self.assertTrue((output_root / "comparison" / "global_local_comparison.json").is_file())
        self.assertTrue((output_root / "comparison" / "per_seed_metrics.csv").is_file())
        self.assertTrue((output_root / "comparison" / "latency_comparison.json").is_file())
        self.assertTrue((output_root / "metadata" / "artifact_manifest.json").is_file())

        self.assertEqual(
            self._snapshot(artifacts_root),
            baseline_artifacts,
            "the fixture pipeline must not write beneath the real repository artifacts/ directory",
        )
        drive_like = [path for path in self.root.rglob("*") if "drive" in path.name.lower()]
        self.assertEqual(drive_like, [], "the fixture pipeline must not create any Drive-like path")


if __name__ == "__main__":
    unittest.main()
