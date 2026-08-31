from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import cya_detector.evaluation.texture_robustness as robustness_module
from cya_detector.config import ConfigError, load_config
from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    read_manifest,
    sha256_file,
    write_manifest,
)
from cya_detector.evaluation.texture_robustness import (
    STAGE1_CELL_IDS,
    TextureRobustnessError,
    TextureRobustnessPrerequisiteError,
    compare_texture_stage1,
    evaluate_texture_stage1,
    materialize_texture_stage1,
    validate_robustness_contract,
)
from cya_detector.models.texture import build_texture_head
from cya_detector.models.clip_baseline import LoadedClip
from cya_detector.features.texture import prepare_texture_patch_views

from tests.test_texture_training import MAKEFILE_PATH, _parse_make_targets

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"
STAGE1_NOTEBOOK_PATH = REPO_ROOT / "notebooks/10_texture_robustness_stage1.ipynb"


class TextureRobustnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_texture_robustness_contract_is_exact_and_locked(self) -> None:
        contract = validate_robustness_contract(self.config)

        self.assertEqual(contract.cell_ids, STAGE1_CELL_IDS)
        self.assertEqual(
            contract.variants,
            ("global_only", "local_only", "global_local"),
        )
        self.assertEqual(contract.seeds, (42, 43, 44))
        self.assertEqual(
            contract.controlling_comparators,
            ("global_only", "controlled_rine"),
        )
        self.assertEqual(contract.aggregate_class_tolerance, 0.01)
        self.assertEqual(contract.worst_cell_tolerance, 0.03)

    def test_texture_robustness_contract_is_immutable(self) -> None:
        contract = validate_robustness_contract(self.config)

        with self.assertRaises(FrozenInstanceError):
            contract.worst_cell_tolerance = 0.5

    def test_contract_rejects_cells_absent_from_benchmark_contract(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["benchmark_transforms"]["jpeg_quality"] = [70, 50, 30]

        with self.assertRaises(ConfigError):
            validate_robustness_contract(candidate)


class TextureRobustnessMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(CONFIG_PATH)
        self.clean_parent_ids = {
            "authentic__matched_clean__fixed_q96",
            "ai_generated__matched_clean__fixed_q96",
        }
        self.input_manifest = self.root / "fixed_q96_manifest.csv"
        self.output_root = self.root / "images"
        self.output_manifest = self.root / "transformed_selection_val.csv"
        self.report_path = self.root / "materialization_report.json"
        self._write_parents()

    @property
    def materialize_args(self) -> dict[str, object]:
        return {
            "input_manifest": self.input_manifest,
            "output_root": self.output_root,
            "output_manifest": self.output_manifest,
            "report_path": self.report_path,
            "config": self.config,
        }

    def _write_parents(self) -> None:
        rows = []
        for label, color in (("authentic", "red"), ("ai_generated", "blue")):
            image_path = self.root / f"{label}.jpg"
            Image.new("RGB", (32, 24), color).save(
                image_path,
                format="JPEG",
                quality=96,
                subsampling=0,
                optimize=False,
                progressive=False,
                exif=b"",
            )
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update(
                {
                    "sample_id": f"{label}__matched_clean__fixed_q96",
                    "source_id": label,
                    "parent_id": f"{label}__source_original",
                    "source_path": str((self.root / f"source-{label}.png").resolve()),
                    "image_path": str(image_path.resolve()),
                    "clean_image_path": str(image_path.resolve()),
                    "image_view": "matched_clean",
                    "sha256": sha256_file(image_path),
                    "label": label,
                    "split": "selection_val",
                    "dataset_name": "fixture",
                    "transform": "clean",
                    "transform_seed": "42",
                    "normalization_codec": "JPEG",
                    "normalization_quality": "96",
                    "output_storage_format": "JPEG",
                }
            )
            rows.append(row)
        write_manifest(self.input_manifest, list(reversed(rows)))

    def _mutate_parent(self, field: str, value: str) -> None:
        rows = read_manifest(self.input_manifest)
        rows[-1][field] = value
        write_manifest(self.input_manifest, rows)

    def test_materializes_exact_stage1_cells_directly_from_selection_clean(self) -> None:
        report = materialize_texture_stage1(**self.materialize_args)
        rows = read_manifest(self.output_manifest)

        self.assertEqual(set(report["cell_counts"]), set(STAGE1_CELL_IDS))
        self.assertEqual(len(rows), len(self.clean_parent_ids) * len(STAGE1_CELL_IDS))
        self.assertEqual({row["split"] for row in rows}, {"selection_val"})
        self.assertEqual({row["image_view"] for row in rows}, {"benchmark"})
        self.assertEqual({row["parent_id"] for row in rows}, self.clean_parent_ids)
        self.assertEqual(report["expected_image_count"], len(rows))
        self.assertEqual(report["stage1_cell_ids"], list(STAGE1_CELL_IDS))

    def test_records_verified_parent_and_output_hashes(self) -> None:
        report = materialize_texture_stage1(**self.materialize_args)

        parent_hashes = {
            row["sample_id"]: row["sha256"] for row in read_manifest(self.input_manifest)
        }
        for row in read_manifest(self.output_manifest):
            self.assertEqual(row["parent_sha256"], parent_hashes[row["parent_id"]])
            self.assertEqual(row["sha256"], sha256_file(Path(row["image_path"])))
        self.assertEqual(report["input_manifest_sha256"], sha256_file(self.input_manifest))
        self.assertEqual(report["output_manifest_sha256"], sha256_file(self.output_manifest))

    def test_rejects_forbidden_split_view_policy_or_chaining_before_writing(self) -> None:
        cases = (
            ("split", "final_test"),
            ("split", "self_train_pool"),
            ("image_view", "benchmark"),
            ("transform", "blur"),
            ("normalization_quality", "95"),
            ("sample_id", "wrong-policy-parent"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self._write_parents()
                self._mutate_parent(field, value)
                with self.assertRaises(TextureRobustnessError):
                    materialize_texture_stage1(**self.materialize_args)
                self.assertFalse(self.output_root.exists())
                self.assertFalse(self.output_manifest.exists())
                self.assertFalse(self.report_path.exists())

    def test_filters_seed_train_without_reading_or_materializing_it(self) -> None:
        rows = read_manifest(self.input_manifest)
        seed_row = dict(rows[0])
        seed_row.update(
            {
                "sample_id": "sealed-seed-parent__matched_clean__fixed_q96",
                "split": "seed_train",
                "image_path": str(self.root / "must-not-be-read.jpg"),
                "clean_image_path": str(self.root / "must-not-be-read.jpg"),
                "sha256": "0" * 64,
            }
        )
        write_manifest(self.input_manifest, rows + [seed_row])

        report = materialize_texture_stage1(**self.materialize_args)

        output_rows = read_manifest(self.output_manifest)
        self.assertEqual(report["parent_count"], 2)
        self.assertEqual(report["source_row_count"], 3)
        self.assertEqual(report["ignored_seed_train_count"], 1)
        self.assertEqual(len(output_rows), 18)
        self.assertNotIn(seed_row["sample_id"], {row["parent_id"] for row in output_rows})

    def test_rejects_parent_hash_mismatch_before_writing(self) -> None:
        self._mutate_parent("sha256", "0" * 64)

        with self.assertRaisesRegex(TextureRobustnessError, "SHA-256"):
            materialize_texture_stage1(**self.materialize_args)

        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.output_manifest.exists())
        self.assertFalse(self.report_path.exists())

    def test_rerun_is_byte_identical(self) -> None:
        first_report = materialize_texture_stage1(**self.materialize_args)
        first_manifest_bytes = self.output_manifest.read_bytes()
        first_report_bytes = self.report_path.read_bytes()
        first_image_hashes = {
            row["sample_id"]: row["sha256"] for row in read_manifest(self.output_manifest)
        }

        second_report = materialize_texture_stage1(**self.materialize_args)

        self.assertEqual(second_report, first_report)
        self.assertEqual(self.output_manifest.read_bytes(), first_manifest_bytes)
        self.assertEqual(self.report_path.read_bytes(), first_report_bytes)
        self.assertEqual(
            {row["sample_id"]: row["sha256"] for row in read_manifest(self.output_manifest)},
            first_image_hashes,
        )

    def test_late_validation_failure_does_not_publish_csv_or_json(self) -> None:
        def invalid_materializer(**kwargs: object) -> dict[str, object]:
            write_manifest(Path(kwargs["output_manifest"]), [])
            Path(kwargs["report_path"]).write_text("{}", encoding="utf-8")
            return {"parent_count": 2, "image_count": 0, "cell_counts": {}}

        with patch(
            "cya_detector.evaluation.texture_robustness.materialize_benchmarks",
            side_effect=invalid_materializer,
        ):
            with self.assertRaises(TextureRobustnessError):
                materialize_texture_stage1(**self.materialize_args)

        self.assertFalse(self.output_manifest.exists())
        self.assertFalse(self.report_path.exists())
        self.assertEqual(list(self.root.glob(".*stage1*")), [])

    def test_rejects_adversarial_materialized_provenance_mutations(self) -> None:
        original_materializer = robustness_module.materialize_benchmarks

        def mutate_json(field: str, key: str, value: object):
            def mutation(row: dict[str, str]) -> None:
                payload = json.loads(row[field])
                payload[key] = value
                row[field] = json.dumps(payload, sort_keys=True, separators=(",", ":"))

            return mutation

        mutations = (
            (
                "label",
                lambda row: row.__setitem__(
                    "label",
                    "authentic" if row["label"] == "ai_generated" else "ai_generated",
                ),
            ),
            ("source_id", lambda row: row.__setitem__("source_id", "spoofed-source")),
            ("transform", lambda row: row.__setitem__("transform", "noise")),
            (
                "declared_parameter",
                mutate_json("transform_parameter", "parameter", 999),
            ),
            (
                "realized_parameter",
                mutate_json("realized_parameters", "output_format", "GIF"),
            ),
            ("transform_seed", lambda row: row.__setitem__("transform_seed", "123")),
            (
                "output_storage_format",
                lambda row: row.__setitem__("output_storage_format", "GIF"),
            ),
            (
                "transform_version",
                lambda row: row.__setitem__("transform_version", "spoofed-v0"),
            ),
            (
                "preprocessing_version",
                lambda row: row.__setitem__("preprocessing_version", "spoofed-v0"),
            ),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                def adversarial_materializer(**kwargs: object) -> dict[str, object]:
                    report = original_materializer(**kwargs)
                    manifest_path = Path(kwargs["output_manifest"])
                    rows = read_manifest(manifest_path)
                    mutation(rows[0])
                    write_manifest(manifest_path, rows)
                    report["output_manifest_sha256"] = sha256_file(manifest_path)
                    return report

                with patch(
                    "cya_detector.evaluation.texture_robustness.materialize_benchmarks",
                    side_effect=adversarial_materializer,
                ):
                    with self.assertRaises(TextureRobustnessError):
                        materialize_texture_stage1(**self.materialize_args)

                self.assertFalse(self.output_manifest.exists())
                self.assertFalse(self.report_path.exists())

    def test_second_publication_failure_rolls_back_new_manifest(self) -> None:
        real_replace = robustness_module.os.replace

        def fail_report_replace(source: object, destination: object) -> None:
            if Path(destination) == self.report_path:
                raise OSError("injected report publication failure")
            real_replace(source, destination)

        with patch(
            "cya_detector.evaluation.texture_robustness.os.replace",
            side_effect=fail_report_replace,
        ):
            with self.assertRaises(TextureRobustnessError):
                materialize_texture_stage1(**self.materialize_args)

        self.assertFalse(self.output_manifest.exists())
        self.assertFalse(self.report_path.exists())

    def test_second_publication_failure_restores_previous_manifest(self) -> None:
        materialize_texture_stage1(**self.materialize_args)
        previous_report = self.report_path.read_bytes()
        previous_manifest = b"previous-valid-manifest\n"
        self.output_manifest.write_bytes(previous_manifest)
        real_replace = robustness_module.os.replace

        def fail_report_replace(source: object, destination: object) -> None:
            if Path(destination) == self.report_path:
                raise OSError("injected report publication failure")
            real_replace(source, destination)

        with patch(
            "cya_detector.evaluation.texture_robustness.os.replace",
            side_effect=fail_report_replace,
        ):
            with self.assertRaises(TextureRobustnessError):
                materialize_texture_stage1(**self.materialize_args)

        self.assertEqual(self.output_manifest.read_bytes(), previous_manifest)
        self.assertEqual(self.report_path.read_bytes(), previous_report)

    def test_cli_materializes_the_locked_matrix(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/materialize_texture_robustness.py",
                "--input-manifest",
                str(self.input_manifest),
                "--output-root",
                str(self.output_root),
                "--output-manifest",
                str(self.output_manifest),
                "--report",
                str(self.report_path),
                "--config",
                str(CONFIG_PATH),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(read_manifest(self.output_manifest)), 18)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["stage1_cell_ids"], list(STAGE1_CELL_IDS))
        self.assertIn("Materialized 18 images", result.stdout)


class TextureRobustnessEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(CONFIG_PATH)
        self.input_manifest = self.root / "fixed_q96_manifest.csv"
        self.output_root = self.root / "images"
        self.output_manifest = self.root / "transformed_selection_val.csv"
        self.report_path = self.root / "materialization_report.json"
        rows = []
        for label, color in (("authentic", "red"), ("ai_generated", "blue")):
            image_path = self.root / f"{label}.jpg"
            Image.new("RGB", (224, 224), color).save(
                image_path, format="JPEG", quality=96, subsampling=0,
                optimize=False, progressive=False, exif=b"",
            )
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update(
                {
                    "sample_id": f"{label}__matched_clean__fixed_q96",
                    "source_id": label,
                    "parent_id": f"{label}__source_original",
                    "source_path": str((self.root / f"source-{label}.png").resolve()),
                    "image_path": str(image_path.resolve()),
                    "clean_image_path": str(image_path.resolve()),
                    "image_view": "matched_clean", "sha256": sha256_file(image_path),
                    "label": label, "split": "selection_val", "dataset_name": "fixture",
                    "transform": "clean", "transform_seed": "42",
                    "normalization_codec": "JPEG", "normalization_quality": "96",
                    "output_storage_format": "JPEG",
                }
            )
            rows.append(row)
        write_manifest(self.input_manifest, rows)
        materialize_texture_stage1(**self.materialize_args)
        self.clean_root = self.root / "clean-runs"
        self.cache_root = self.root / "cache"
        self.prediction_root = self.root / "predictions"
        self._write_clean_runs()

    @property
    def materialize_args(self) -> dict[str, object]:
        return {
            "input_manifest": self.input_manifest, "output_root": self.output_root,
            "output_manifest": self.output_manifest, "report_path": self.report_path,
            "config": self.config,
        }

    def _write_clean_runs(self) -> None:
        import torch

        for variant in ("global_only", "local_only", "global_local"):
            for seed in (42, 43, 44):
                run = self.clean_root / variant / f"seed_{seed}"
                checkpoint = run / "checkpoints" / "best_clean.pt"
                checkpoint.parent.mkdir(parents=True)
                model = build_texture_head(
                    variant=variant,
                    layer_count=4,
                    global_dimension=3,
                    patch_dimension=3,
                    fusion_dimension=256,
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "stage": "texture_stage_d",
                        "variant": variant,
                        "seed": seed,
                    },
                    checkpoint,
                )
                prediction = run / "predictions" / "selection_val.csv"
                prediction.parent.mkdir(parents=True)
                with prediction.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=(
                            "sample_id", "source_id", "parent_id", "split", "label",
                            "logit", "probability", "checkpoint", "seed",
                            "matching_policy", "transform", "transform_parameter",
                            "dataset_name", "generator_name", "generator_checkpoint",
                            "capture_source",
                        ),
                    )
                    writer.writeheader()
                    for label in ("authentic", "ai_generated"):
                        writer.writerow(
                            {
                                "sample_id": f"{label}__matched_clean__fixed_q96",
                                "source_id": label,
                                "parent_id": f"{label}__source_original",
                                "split": "selection_val",
                                "label": label,
                                "logit": "-1" if label == "authentic" else "1",
                                "probability": "0.25" if label == "authentic" else "0.75",
                                "checkpoint": "best_clean",
                                "seed": seed,
                                "matching_policy": "fixed_q96",
                                "transform": "clean",
                                "transform_parameter": "",
                                "dataset_name": "fixture",
                                "generator_name": "unknown",
                                "generator_checkpoint": "unknown",
                                "capture_source": "unknown",
                            }
                        )
                metadata = run / "metadata" / "run_metadata.json"
                metadata.parent.mkdir(parents=True)
                metadata.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "stage": "texture_stage_d",
                            "variant": variant,
                            "seed": seed,
                            "matching_policy": "fixed_q96",
                            "optimization": {"threshold": 0.5},
                            "run_configuration": {
                                "texture": {"experiment_name": "clean_pilot_v1"},
                                "evaluation": {"threshold": 0.5},
                            },
                            "artifact_sha256": {
                                "checkpoints/best_clean.pt": hashlib.sha256(
                                    checkpoint.read_bytes()
                                ).hexdigest(),
                                "predictions/selection_val.csv": hashlib.sha256(
                                    prediction.read_bytes()
                                ).hexdigest(),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

    @property
    def evaluation_args(self) -> dict[str, object]:
        return {
            "transformed_manifest": self.output_manifest,
            "materialization_report": self.report_path,
            "clean_experiment_root": self.clean_root,
            "cache_root": self.cache_root,
            "output_root": self.prediction_root,
            "config": self.config,
            "device": "cpu",
        }

    def _fake_features(self, **kwargs: object):
        import torch

        rows = read_manifest(Path(kwargs["transformed_manifest"]))
        return [
            {
                "manifest_row": row,
                "global_features": torch.ones((4, 3)),
                "patch_features": torch.ones((4, 3)),
                "patch_mask": torch.ones(4, dtype=torch.bool),
                "patch_boxes": ((0, 0, 112, 112),) * 4,
                "global_feature_sha256": "a" * 64,
                "patch_feature_sha256": "b" * 64,
                "cache_contract_sha256": "c" * 64,
            }
            for row in rows
        ], {"encoded_image_count": len(rows) * 5, "cache_hit_count": 0}

    def test_evaluates_all_81_slices_with_one_shared_extraction(self) -> None:
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ) as extraction:
            summary = evaluate_texture_stage1(**self.evaluation_args)

        self.assertEqual(summary["completed_slices"], 81)
        self.assertEqual(extraction.call_count, 1)
        self.assertEqual(len(list(self.prediction_root.rglob("*.csv"))), 81)
        slice_metadata = json.loads(
            next(self.prediction_root.rglob("*.meta.json")).read_text(encoding="utf-8")
        )
        self.assertEqual(slice_metadata["inference"]["sample_count"], 2)
        self.assertGreaterEqual(slice_metadata["inference"]["latency_seconds"], 0.0)

    def test_resume_requires_valid_slice_hashes_and_keeps_checkpoints_immutable(self) -> None:
        checkpoints = list(self.clean_root.rglob("best_clean.pt"))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            evaluate_texture_stage1(**self.evaluation_args)
            resumed = evaluate_texture_stage1(**self.evaluation_args)
        self.assertEqual(resumed["resumed_slices"], 81)
        self.assertEqual(
            before,
            {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints},
        )

        corrupt = next(self.prediction_root.rglob("*.csv"))
        corrupt.write_text("partial", encoding="utf-8")
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            repaired = evaluate_texture_stage1(**self.evaluation_args)
        self.assertEqual(repaired["resumed_slices"], 80)

    def test_changed_clean_predictions_or_run_configuration_invalidate_nine_slices(self) -> None:
        run = self.clean_root / "global_only" / "seed_42"
        predictions = run / "predictions" / "selection_val.csv"
        metadata_path = run / "metadata" / "run_metadata.json"
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            evaluate_texture_stage1(**self.evaluation_args)
            with predictions.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["probability"] = "0.125"
            with predictions.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["artifact_sha256"]["predictions/selection_val.csv"] = hashlib.sha256(
                predictions.read_bytes()
            ).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            replaced = evaluate_texture_stage1(**self.evaluation_args)

        self.assertEqual(replaced["resumed_slices"], 72)
        cell = self.prediction_root / "global_only" / "seed_42" / "jpeg_q90.csv"
        with cell.open(newline="", encoding="utf-8") as stream:
            updated_rows = list(csv.DictReader(stream))
        authentic = next(row for row in updated_rows if row["label"] == "authentic")
        self.assertEqual(float(authentic["paired_clean_probability"]), 0.125)

        metadata["run_configuration"]["evaluation"]["threshold_note"] = "review-fix"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            config_replaced = evaluate_texture_stage1(**self.evaluation_args)
        self.assertEqual(config_replaced["resumed_slices"], 72)

    def test_rejects_forbidden_split_before_feature_extraction_or_publication(self) -> None:
        rows = read_manifest(self.output_manifest)
        rows[0]["split"] = "final_test"
        write_manifest(self.output_manifest, rows)
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank"
        ) as extraction:
            with self.assertRaisesRegex(TextureRobustnessError, "selection_val"):
                evaluate_texture_stage1(**self.evaluation_args)
        extraction.assert_not_called()
        self.assertFalse(self.prediction_root.exists())

    def test_rejects_changed_clean_threshold_and_nonfinite_logits_without_publication(self) -> None:
        import torch

        run = self.clean_root / "global_only" / "seed_42"
        metadata_path = run / "metadata" / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["optimization"]["threshold"] = 0.4
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            with self.assertRaisesRegex(TextureRobustnessError, "threshold mismatch"):
                evaluate_texture_stage1(**self.evaluation_args)
        self.assertFalse(self.prediction_root.exists())

        metadata["optimization"]["threshold"] = 0.5
        checkpoint = run / "checkpoints" / "best_clean.pt"
        payload = torch.load(checkpoint, weights_only=False)
        first = next(iter(payload["model_state_dict"]))
        payload["model_state_dict"][first].fill_(float("nan"))
        torch.save(payload, checkpoint)
        metadata["artifact_sha256"]["checkpoints/best_clean.pt"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            with self.assertRaisesRegex(TextureRobustnessError, "non-finite outputs"):
                evaluate_texture_stage1(**self.evaluation_args)
        self.assertFalse(self.prediction_root.exists())

    def test_real_extraction_recomputes_transformed_boxes_once_and_binds_cache_contract(self) -> None:
        import numpy as np
        import torch

        class FakeProcessor:
            def __call__(self, *, images, return_tensors):
                array = np.asarray(images, dtype=np.float32)
                value = float(array.mean() / 255.0)
                return {"pixel_values": torch.full((1, 3, 8, 8), value)}

        class FakeModel:
            def __init__(self):
                self.config = type(
                    "Config", (), {"num_hidden_layers": 24, "hidden_size": 3}
                )()
                self.encoded_image_count = 0

            def __call__(
                self, *, pixel_values, output_hidden_states=False, return_dict=False
            ):
                batch = pixel_values.shape[0]
                self.encoded_image_count += batch
                base = pixel_values.mean(dim=(1, 2, 3)).view(batch, 1, 1)
                result = type("Output", (), {})()
                result.image_embeds = base.repeat(1, 1, 3).reshape(batch, 3)
                if output_hidden_states:
                    result.hidden_states = tuple(
                        (base + index).repeat(1, 1, 3) for index in range(25)
                    )
                return result

        manifest_rows = read_manifest(self.output_manifest)
        selected = manifest_rows[:1]
        transformed = np.zeros((224, 224, 3), dtype=np.uint8)
        transformed[112:, 112:] = np.indices((112, 112)).sum(axis=0)[..., None] % 2 * 255
        image_path = Path(selected[0]["image_path"])
        Image.fromarray(transformed).save(image_path, format="JPEG", quality=95)
        selected[0]["sha256"] = sha256_file(image_path)
        fake_model = FakeModel()
        loaded = LoadedClip(
            model=fake_model, processor=FakeProcessor(), identifier="fixture/clip",
            requested_revision="main", resolved_revision="a" * 40,
            embedding_dimension=3,
        )
        with patch(
            "cya_detector.evaluation.texture_robustness.load_frozen_clip",
            return_value=loaded,
        ):
            extracted, report = robustness_module._extract_transformed_feature_bank(
                transformed_manifest=self.output_manifest,
                rows=selected,
                cache_root=self.cache_root,
                config=self.config,
                device="cpu",
            )

        expected = prepare_texture_patch_views(
            transformed,
            patch_size=self.config["texture"]["patch_size"],
            patch_count=self.config["texture"]["patch_count"],
        ).patch_boxes
        self.assertEqual(extracted[0]["patch_boxes"], expected)
        self.assertEqual(fake_model.encoded_image_count, 1 + len(expected))
        self.assertEqual(report["encoded_image_count"], fake_model.encoded_image_count)
        cache_file = next((self.cache_root / "patch").rglob("*.pt"))
        cache_contract = torch.load(cache_file, weights_only=True)["cache_contract"]
        for key in (
            "input_sha256", "parent_sha256", "cell_contract", "resolved_revision",
            "preprocessing_version", "texture_extractor_version", "matching_policy",
            "patch_size", "patch_count", "patch_boxes",
        ):
            self.assertIn(key, cache_contract)

    def test_malformed_matching_caches_are_rejected_and_reextracted(self) -> None:
        import torch

        manifest_rows = read_manifest(self.output_manifest)[:1]

        class FakeProcessor:
            def __call__(self, *, images, return_tensors):
                return {"pixel_values": torch.ones((1, 3, 8, 8))}

        class FakeModel:
            def __init__(self):
                self.config = type(
                    "Config", (), {"num_hidden_layers": 24, "hidden_size": 3}
                )()
                self.encoded_image_count = 0

            def __call__(
                self, *, pixel_values, output_hidden_states=False, return_dict=False
            ):
                batch = pixel_values.shape[0]
                self.encoded_image_count += batch
                result = type("Output", (), {})()
                result.image_embeds = torch.ones((batch, 3))
                if output_hidden_states:
                    result.hidden_states = tuple(
                        torch.ones((batch, 1, 3)) * index for index in range(25)
                    )
                return result

        model = FakeModel()
        loaded = LoadedClip(
            model=model, processor=FakeProcessor(), identifier="fixture/clip",
            requested_revision="main", resolved_revision="b" * 40,
            embedding_dimension=3,
        )
        extractor_args = {
            "transformed_manifest": self.output_manifest,
            "rows": manifest_rows,
            "cache_root": self.cache_root,
            "config": self.config,
            "device": "cpu",
        }
        with patch(
            "cya_detector.evaluation.texture_robustness.load_frozen_clip",
            return_value=loaded,
        ):
            robustness_module._extract_transformed_feature_bank(**extractor_args)
            global_path = next((self.cache_root / "global").rglob("*.pt"))
            patch_path = next((self.cache_root / "patch").rglob("*.pt"))
            global_payload = torch.load(global_path, weights_only=True)
            patch_payload = torch.load(patch_path, weights_only=True)
            global_payload["global_features"] = torch.ones((3, 3))
            patch_payload["patch_features"] = torch.ones((3, 3))
            patch_payload["patch_boxes"] = [[99, 99, 112, 112]]
            patch_payload["patch_mask"] = torch.tensor([True, False, False, False])
            torch.save(global_payload, global_path)
            torch.save(patch_payload, patch_path)
            _, repaired = robustness_module._extract_transformed_feature_bank(**extractor_args)

        self.assertEqual(repaired["cache_hit_count"], 0)
        repaired_global = torch.load(global_path, weights_only=True)
        repaired_patch = torch.load(patch_path, weights_only=True)
        self.assertEqual(tuple(repaired_global["global_features"].shape), (4, 3))
        self.assertEqual(tuple(repaired_patch["patch_features"].shape), (4, 3))
        self.assertEqual(
            repaired_patch["patch_boxes"],
            repaired_patch["cache_contract"]["patch_boxes"],
        )
        self.assertEqual(
            repaired_patch["patch_mask"].tolist(),
            repaired_patch["cache_contract"]["availability_mask"],
        )


_PARENTS = (
    ("authentic", "authentic__source_original"),
    ("ai_generated", "ai_generated__source_original"),
)


def _probability_for(label: str, *, correct: bool) -> float:
    wants_positive = label == "ai_generated"
    predicted_positive = wants_positive if correct else not wants_positive
    return 0.75 if predicted_positive else 0.25


class TextureRobustnessComparisonTests(unittest.TestCase):
    """The break caught here is a gate that trusts an unverified or incomplete
    controlled-RINE comparator, or that lets the treatment tie or lose against a
    comparator without rejecting."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(CONFIG_PATH)
        self.clean_root = self.root / "clean-runs"
        self.robustness_root = self.root / "robustness-predictions"
        self.rine_root = self.root / "controlled-rine"
        self.reset_comparison_fixture()

    @property
    def comparison_args(self) -> dict[str, object]:
        return {
            "clean_experiment_root": self.clean_root,
            "robustness_root": self.robustness_root,
            "controlled_rine_root": self.rine_root,
            "config": self.config,
        }

    def reset_comparison_fixture(self) -> None:
        import shutil

        shutil.rmtree(self.clean_root, ignore_errors=True)
        shutil.rmtree(self.robustness_root, ignore_errors=True)
        shutil.rmtree(self.rine_root, ignore_errors=True)
        self._write_clean_runs()
        wrong = {("global_only", "jpeg_q90", 42, "ai_generated")}
        for variant in ("global_only", "local_only", "global_local"):
            self._write_variant_predictions(variant, wrong={unit[1:] for unit in wrong if unit[0] == variant})
        self._write_rine_predictions(wrong={("jpeg_q70", 43, "ai_generated")})

    def _write_clean_runs(self) -> None:
        import torch

        for variant in ("global_only", "local_only", "global_local"):
            for seed in (42, 43, 44):
                run = self.clean_root / variant / f"seed_{seed}"
                checkpoint = run / "checkpoints" / "best_clean.pt"
                checkpoint.parent.mkdir(parents=True)
                model = build_texture_head(
                    variant=variant, layer_count=4, global_dimension=3,
                    patch_dimension=3, fusion_dimension=256,
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(), "stage": "texture_stage_d",
                        "variant": variant, "seed": seed,
                    },
                    checkpoint,
                )
                prediction = run / "predictions" / "selection_val.csv"
                prediction.parent.mkdir(parents=True)
                with prediction.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=(
                            "sample_id", "source_id", "parent_id", "split", "label",
                            "logit", "probability", "checkpoint", "seed",
                            "matching_policy", "transform", "transform_parameter",
                            "dataset_name", "generator_name", "generator_checkpoint",
                            "capture_source",
                        ),
                    )
                    writer.writeheader()
                    for label, parent_id in _PARENTS:
                        probability = _probability_for(label, correct=True)
                        writer.writerow(
                            {
                                "sample_id": parent_id, "source_id": label, "parent_id": parent_id,
                                "split": "selection_val", "label": label,
                                "logit": "1" if probability >= 0.5 else "-1",
                                "probability": probability, "checkpoint": "best_clean", "seed": seed,
                                "matching_policy": "fixed_q96", "transform": "clean",
                                "transform_parameter": "", "dataset_name": "fixture",
                                "generator_name": "unknown", "generator_checkpoint": "unknown",
                                "capture_source": "unknown",
                            }
                        )
                metadata = run / "metadata" / "run_metadata.json"
                metadata.parent.mkdir(parents=True)
                metadata.write_text(
                    json.dumps(
                        {
                            "status": "completed", "stage": "texture_stage_d", "variant": variant,
                            "seed": seed, "matching_policy": "fixed_q96",
                            "optimization": {"threshold": 0.5},
                            "run_configuration": {
                                "texture": {"experiment_name": "clean_pilot_v1"},
                                "evaluation": {"threshold": 0.5},
                            },
                            "artifact_sha256": {
                                "checkpoints/best_clean.pt": hashlib.sha256(
                                    checkpoint.read_bytes()
                                ).hexdigest(),
                                "predictions/selection_val.csv": hashlib.sha256(
                                    prediction.read_bytes()
                                ).hexdigest(),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

    def _write_variant_predictions(
        self, variant: str, *, wrong: set[tuple[str, int, str]],
    ) -> None:
        for seed in (42, 43, 44):
            for cell_id in STAGE1_CELL_IDS:
                path = self.robustness_root / variant / f"seed_{seed}" / f"{cell_id}.csv"
                metadata_path = path.with_suffix(".meta.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for label, parent_id in _PARENTS:
                    correct = (cell_id, seed, label) not in wrong
                    probability = _probability_for(label, correct=correct)
                    clean_probability = _probability_for(label, correct=True)
                    rows.append(
                        {
                            "sample_id": f"{parent_id}__{cell_id}", "parent_id": parent_id,
                            "source_id": label, "split": "selection_val", "label": label,
                            "cell_id": cell_id, "cell_parameters": json.dumps({"cell_id": cell_id}),
                            "variant": variant, "seed": seed,
                            "logit": "1" if probability >= 0.5 else "-1", "probability": probability,
                            "prediction": int(probability >= 0.5),
                            "paired_clean_probability": clean_probability,
                            "paired_clean_prediction": int(clean_probability >= 0.5),
                            "checkpoint_sha256": "checkpoint-hash", "input_sha256": "input-hash",
                            "parent_sha256": "parent-hash", "global_feature_sha256": "global-hash",
                            "patch_feature_sha256": "patch-hash", "cache_contract_sha256": "cache-hash",
                            "patch_boxes": "[]", "available_patch_count": 1,
                            "matching_policy": "fixed_q96", "transform": "benchmark",
                            "transform_parameter": json.dumps({"cell_id": cell_id}),
                        }
                    )
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=robustness_module._EVALUATION_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                metadata_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "contract": {"variant": variant, "seed": seed, "cell_id": cell_id},
                            "row_count": len(rows),
                            "csv_sha256": sha256_file(path),
                        }
                    ),
                    encoding="utf-8",
                )

    def _write_rine_predictions(self, *, wrong: set[tuple[str, int, str]]) -> None:
        import torch

        for seed in (42, 43, 44):
            run = self.rine_root / f"seed_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stage": "controlled_rine_robustness", "seed": seed,
                    "matching_policy": "fixed_q96", "threshold": 0.5,
                    "transform_cells": list(STAGE1_CELL_IDS) + ["noise_sigma_0.02"],
                    "manifest_sha256": "controlled-rine-manifest-hash",
                },
                run / "best_50_50.pt",
            )
            path = run / "best_50_50_predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "sample_id", "source_id", "parent_id", "split", "label",
                        "logit", "probability", "checkpoint", "seed",
                        "matching_policy", "transform", "transform_parameter",
                        "dataset_name", "generator_name", "generator_checkpoint",
                        "capture_source",
                    ),
                )
                writer.writeheader()
                for label, parent_id in _PARENTS:
                    probability = _probability_for(label, correct=True)
                    writer.writerow(
                        {
                            "sample_id": parent_id, "source_id": label, "parent_id": parent_id,
                            "split": "selection_val", "label": label,
                            "logit": "1" if probability >= 0.5 else "-1", "probability": probability,
                            "checkpoint": "controlled_rine", "seed": seed,
                            "matching_policy": "fixed_q96", "transform": "clean",
                            "transform_parameter": "", "dataset_name": "fixture",
                            "generator_name": "unknown", "generator_checkpoint": "unknown",
                            "capture_source": "unknown",
                        }
                    )
                for cell_id in STAGE1_CELL_IDS:
                    for label, parent_id in _PARENTS:
                        correct = (cell_id, seed, label) not in wrong
                        probability = _probability_for(label, correct=correct)
                        writer.writerow(
                            {
                                "sample_id": f"{parent_id}__{cell_id}", "source_id": label,
                                "parent_id": parent_id, "split": "selection_val", "label": label,
                                "logit": "1" if probability >= 0.5 else "-1", "probability": probability,
                                "checkpoint": "controlled_rine", "seed": seed,
                                "matching_policy": "fixed_q96", "transform": "benchmark",
                                "transform_parameter": json.dumps({"cell_id": cell_id}),
                                "dataset_name": "fixture", "generator_name": "unknown",
                                "generator_checkpoint": "unknown", "capture_source": "unknown",
                            }
                        )

    def test_retains_only_when_locked_score_and_every_regression_gate_pass(self) -> None:
        report = compare_texture_stage1(**self.comparison_args)

        self.assertEqual(report["decision"], "retain_texture_for_full_robustness")
        self.assertGreater(report["comparators"]["global_only"]["locked_score_delta"], 0.0)
        self.assertGreater(report["comparators"]["controlled_rine"]["locked_score_delta"], 0.0)

    def test_recomputes_controlled_rine_on_the_exact_nine_cell_subset(self) -> None:
        report = compare_texture_stage1(**self.comparison_args)

        self.assertEqual(report["comparators"]["controlled_rine"]["cell_count"], 9)

    def remove_one_controlled_rine_slice(self) -> None:
        (self.rine_root / "seed_43" / "best_50_50_predictions.csv").unlink()

    def test_missing_or_mismatched_controlled_rine_blocks_comparison(self) -> None:
        self.remove_one_controlled_rine_slice()

        with self.assertRaises(TextureRobustnessPrerequisiteError):
            compare_texture_stage1(**self.comparison_args)

    def gate_mutations(self, comparator: str) -> list[Callable[[], None]]:
        def equalize_locked_score() -> None:
            if comparator == "controlled_rine":
                self._write_rine_predictions(wrong=set())
            else:
                self._write_variant_predictions(comparator, wrong=set())

        def regress_treatment_authentic_accuracy() -> None:
            self._write_variant_predictions(
                "global_local", wrong={(STAGE1_CELL_IDS[0], 42, "authentic")}
            )

        def regress_treatment_single_cell() -> None:
            wrong = {
                (STAGE1_CELL_IDS[0], seed, label)
                for seed in (42, 43, 44) for label, _ in _PARENTS
            }
            self._write_variant_predictions("global_local", wrong=wrong)

        return [
            equalize_locked_score,
            regress_treatment_authentic_accuracy,
            regress_treatment_single_cell,
        ]

    def test_rejects_each_gate_independently_against_either_comparator(self) -> None:
        for comparator in ("global_only", "controlled_rine"):
            for mutation in self.gate_mutations(comparator):
                with self.subTest(comparator=comparator, mutation=mutation.__name__):
                    self.reset_comparison_fixture()
                    mutation()
                    self.assertEqual(
                        compare_texture_stage1(**self.comparison_args)["decision"],
                        "reject_texture_robustness_stage1",
                    )

    def test_publishes_report_artifacts_and_manifest_last(self) -> None:
        output_root = self.root / "reports-output"

        report = compare_texture_stage1(**self.comparison_args, output_root=output_root)

        self.assertTrue((output_root / "reports" / "robustness_comparison.json").is_file())
        self.assertTrue((output_root / "reports" / "per_cell_metrics.csv").is_file())
        self.assertTrue((output_root / "reports" / "per_seed_robustness.csv").is_file())
        self.assertTrue((output_root / "reports" / "failure_analysis.csv").is_file())
        manifest = json.loads(
            (output_root / "metadata" / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["decision"], report["decision"])


class TextureRobustnessCommandContractTests(unittest.TestCase):
    """The break caught here is a launcher that reimplements transform, model, inference,
    metric, or gate logic instead of calling the locked Stage-1 CLIs."""

    def setUp(self) -> None:
        self.makefile_text = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.targets = _parse_make_targets(self.makefile_text)

    def test_makefile_declares_all_four_stage1_targets(self) -> None:
        for target in (
            "task9-robustness-test", "task9-robustness-materialize",
            "task9-robustness-evaluate", "task9-robustness-compare",
        ):
            with self.subTest(target=target):
                self.assertIn(target, self.targets, f"Makefile is missing target {target!r}")

    def test_stage1_targets_reference_the_locked_clis(self) -> None:
        self.assertIn(
            "materialize_texture_robustness.py", self.targets.get("task9-robustness-materialize", "")
        )
        self.assertIn(
            "evaluate_texture_robustness.py", self.targets.get("task9-robustness-evaluate", "")
        )
        self.assertIn(
            "compare_texture_robustness.py", self.targets.get("task9-robustness-compare", "")
        )

    def test_stage1_caches_default_under_content_and_output_defaults_under_artifact_root(self) -> None:
        self.assertRegex(
            self.makefile_text, r"(?m)^TASK9_ROBUSTNESS_IMAGE_ROOT\s*\?=\s*/content(/|\s|$)"
        )
        self.assertRegex(
            self.makefile_text, r"(?m)^TASK9_ROBUSTNESS_CACHE_ROOT\s*\?=\s*/content(/|\s|$)"
        )
        self.assertRegex(
            self.makefile_text,
            r"(?m)^TASK9_ROBUSTNESS_OUTPUT_ROOT\s*\?=\s*\$\(ARTIFACT_ROOT\)/task9/clean_pilot_v1/robustness_stage1_v1\s*$",
        )

    def test_stage1_evaluate_and_compare_depend_on_the_locked_clean_experiment_root(self) -> None:
        evaluate_recipe = self.targets.get("task9-robustness-evaluate", "")
        compare_recipe = self.targets.get("task9-robustness-compare", "")
        self.assertIn("$(TASK9_ROBUSTNESS_CLEAN_ROOT)", evaluate_recipe)
        self.assertIn("$(TASK9_ROBUSTNESS_CLEAN_ROOT)", compare_recipe)
        self.assertIn("$(TASK9_ROBUSTNESS_CONTROLLED_RINE_ROOT)", compare_recipe)

    def test_clean_launcher_is_referenced_only_by_its_renamed_path(self) -> None:
        self.assertNotIn("07_texture_stage_d.ipynb", self.makefile_text)
        for path in (REPO_ROOT / "notebooks" / "README.md", REPO_ROOT / "docs" / "planning" / "nextSteps.md"):
            with self.subTest(path=path):
                self.assertNotIn("07_texture_stage_d.ipynb", path.read_text(encoding="utf-8"))
        self.assertTrue((REPO_ROOT / "notebooks" / "09_texture_stage_d.ipynb").is_file())

    def test_stage1_notebook_is_a_thin_launcher_with_no_inlined_transform_model_or_gate_logic(self) -> None:
        self.assertTrue(STAGE1_NOTEBOOK_PATH.is_file(), f"{STAGE1_NOTEBOOK_PATH} does not exist")
        notebook = json.loads(STAGE1_NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        self.assertTrue(code_cells, "notebook has no code cells")
        source_text = "\n".join("".join(cell["source"]) for cell in code_cells)

        forbidden_patterns = (
            r"class\s+\w+\s*\(",
            r"nn\.Module",
            r"nn\.Linear",
            r"nn\.Sequential",
            r"def\s+forward\s*\(",
            r"def\s+evaluate_texture_stage1\s*\(",
            r"def\s+compare_texture_stage1\s*\(",
            r"def\s+materialize_texture_stage1\s*\(",
            r"prepare_texture_patch_views",
            r"build_texture_head",
            r"binary_metrics",
            r"apply_benchmark",
        )
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertNotRegex(source_text, pattern)

        for script in (
            "materialize_texture_robustness.py", "evaluate_texture_robustness.py",
            "compare_texture_robustness.py",
        ):
            with self.subTest(script=script):
                self.assertIn(script, source_text)

    def test_stage1_notebook_never_reads_seed_train_self_train_pool_or_final_test(self) -> None:
        notebook = json.loads(STAGE1_NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        source_text = "\n".join("".join(cell["source"]) for cell in code_cells)
        for forbidden in ("seed_train", "self_train_pool", "final_test"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source_text)


class TextureRobustnessFixtureSmokeTests(TextureRobustnessEvaluationTests):
    """The break caught here is any Stage-1 stage that needs pretrained weights, a GPU, or that
    reaches outside its declared temporary root into the real repository artifacts or a Drive-like
    path. Covers materialization, shared fake extraction, all 81 frozen texture evaluations, the
    three hash-verified controlled-RINE files with their 27 seed-cell partitions, the two-comparator
    gate, and hashed publication."""

    def setUp(self) -> None:
        super().setUp()
        self.rine_root = self.root / "controlled-rine"
        self._write_rine_predictions()

    @staticmethod
    def _snapshot(directory: Path) -> set[tuple[str, int]]:
        if not directory.is_dir():
            return set()
        return {
            (str(path.relative_to(directory)), path.stat().st_size)
            for path in directory.rglob("*")
            if path.is_file()
        }

    def _write_rine_predictions(self) -> None:
        import torch

        for seed in (42, 43, 44):
            run = self.rine_root / f"seed_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stage": "controlled_rine_robustness", "seed": seed,
                    "matching_policy": "fixed_q96", "threshold": 0.5,
                    "transform_cells": list(STAGE1_CELL_IDS),
                    "manifest_sha256": "controlled-rine-manifest-hash",
                },
                run / "best_50_50.pt",
            )
            path = run / "best_50_50_predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "sample_id", "source_id", "parent_id", "split", "label",
                        "logit", "probability", "checkpoint", "seed",
                        "matching_policy", "transform", "transform_parameter",
                        "dataset_name", "generator_name", "generator_checkpoint",
                        "capture_source",
                    ),
                )
                writer.writeheader()
                parents = [
                    (label, f"{label}__matched_clean__fixed_q96") for label in ("authentic", "ai_generated")
                ]
                for label, parent_id in parents:
                    probability = _probability_for(label, correct=True)
                    writer.writerow(
                        {
                            "sample_id": parent_id, "source_id": label, "parent_id": parent_id,
                            "split": "selection_val", "label": label,
                            "logit": "1" if probability >= 0.5 else "-1", "probability": probability,
                            "checkpoint": "controlled_rine", "seed": seed,
                            "matching_policy": "fixed_q96", "transform": "clean",
                            "transform_parameter": "", "dataset_name": "fixture",
                            "generator_name": "unknown", "generator_checkpoint": "unknown",
                            "capture_source": "unknown",
                        }
                    )
                for cell_id in STAGE1_CELL_IDS:
                    for label, parent_id in parents:
                        probability = _probability_for(label, correct=True)
                        writer.writerow(
                            {
                                "sample_id": f"{parent_id}__{cell_id}", "source_id": label,
                                "parent_id": parent_id, "split": "selection_val", "label": label,
                                "logit": "1" if probability >= 0.5 else "-1", "probability": probability,
                                "checkpoint": "controlled_rine", "seed": seed,
                                "matching_policy": "fixed_q96", "transform": "benchmark",
                                "transform_parameter": json.dumps({"cell_id": cell_id}),
                                "dataset_name": "fixture", "generator_name": "unknown",
                                "generator_checkpoint": "unknown", "capture_source": "unknown",
                            }
                        )

    def test_full_pipeline_runs_under_a_temporary_root_without_touching_real_artifacts_or_drive(
        self,
    ) -> None:
        real_artifacts_before = self._snapshot(REPO_ROOT / "artifacts")

        with patch(
            "cya_detector.evaluation.texture_robustness._extract_transformed_feature_bank",
            side_effect=self._fake_features,
        ):
            evaluation_summary = evaluate_texture_stage1(**self.evaluation_args)
        self.assertEqual(evaluation_summary["completed_slices"], 81)
        self.assertEqual(len(list(self.prediction_root.rglob("*.csv"))), 81)

        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                self.assertTrue((self.rine_root / f"seed_{seed}" / "best_50_50.pt").is_file())
                with (self.rine_root / f"seed_{seed}" / "best_50_50_predictions.csv").open(
                    newline="", encoding="utf-8"
                ) as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(
                    len({(row["parent_id"], row["transform_parameter"]) for row in rows if row["transform"] != "clean"}),
                    len(STAGE1_CELL_IDS) * 2,
                )

        compare_output = self.root / "compare-output"
        report = compare_texture_stage1(
            clean_experiment_root=self.clean_root,
            robustness_root=self.prediction_root,
            controlled_rine_root=self.rine_root,
            config=self.config,
            output_root=compare_output,
        )
        self.assertIn(
            report["decision"],
            ("retain_texture_for_full_robustness", "reject_texture_robustness_stage1"),
        )
        self.assertEqual(report["comparators"]["controlled_rine"]["cell_count"], len(STAGE1_CELL_IDS))
        manifest = json.loads(
            (compare_output / "metadata" / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["decision"], report["decision"])

        self.assertEqual(self._snapshot(REPO_ROOT / "artifacts"), real_artifacts_before)
        self.assertFalse((REPO_ROOT / "artifacts" / "task9").exists())


if __name__ == "__main__":
    unittest.main()
