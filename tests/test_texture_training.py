from __future__ import annotations

import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch

from cya_detector.data.dataset import ManifestExample
from cya_detector.training.texture_stage_d import CachedTextureFeatures


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
            torch.save(
                {
                    "patch_features": torch.full((4, 2), value, dtype=torch.float32),
                    "patch_mask": torch.tensor([True, True, False, False]),
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

        return train_texture_head(
            rows=self.rows if rows is None else rows,
            variant=variant,
            seed=seed,
            output_root=root,
            overwrite=overwrite,
            run_configuration=self.configuration,
            device="cpu",
            learning_rate=0.01,
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
        metadata_path = Path(first["run_root"]) / "metadata" / "run_metadata.json"
        original_metadata = metadata_path.read_bytes()
        with patch(
            "cya_detector.training.texture_stage_d._replace_published_file",
            side_effect=OSError("interrupted replacement"),
        ):
            with self.assertRaisesRegex(OSError, "interrupted replacement"):
                self._train(root=output_root, overwrite=True)
        self.assertTrue(metadata_path.is_file())
        self.assertEqual(metadata_path.read_bytes(), original_metadata)

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

    def test_cuda_determinism_fails_closed_when_torch_cannot_guarantee_it(self) -> None:
        def deterministic_only(enabled: bool, *, warn_only: bool) -> None:
            if warn_only:
                return
            raise RuntimeError("determinism unavailable")

        with patch("torch.use_deterministic_algorithms", side_effect=deterministic_only):
            with self.assertRaisesRegex(RuntimeError, "determinism unavailable"):
                self._train(root=self.root / "determinism")


if __name__ == "__main__":
    unittest.main()
