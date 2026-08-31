from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
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
    evaluate_texture_stage1,
    materialize_texture_stage1,
    validate_robustness_contract,
)
from cya_detector.models.texture import build_texture_head
from cya_detector.models.clip_baseline import LoadedClip
from cya_detector.features.texture import prepare_texture_patch_views

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"


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


if __name__ == "__main__":
    unittest.main()
