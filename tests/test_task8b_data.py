from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from cya_detector.data.manifest import DatasetContractError, write_manifest
from cya_detector.data.task8b import (
    TASK8B_REQUIRED_COLUMNS,
    assign_task8b_splits,
    build_task8b_manifest,
)
from cya_detector.data.task8b_audit import audit_task8b_readiness
from cya_detector.data.task8b_archive import extract_genimage_ai_sample
from cya_detector.data.task8b_matched import build_task8b_matched_views
from cya_detector.data.task8b_prepare import (
    Task8BPreparationError,
    prepare_task8b_inventory,
)
from cya_detector.features.prnu_reference import build_training_prnu_references


REPO_ROOT = Path(__file__).resolve().parents[1]


def patterned_image(path: Path, index: int) -> None:
    image = Image.new("RGB", (128, 96), ((index * 37) % 255, 80, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5 + index, 8, 50, 60), outline=(255, 255, 255), width=2)
    draw.line((0, index % 30, 127, 95 - index % 30), fill=(0, 0, 0), width=2)
    for offset in range(index + 1):
        x = (offset * 11 + index * 7) % 120
        draw.rectangle((x, 70, min(x + 5, 127), 90), fill=(255, 255, 0))
    image.save(path)


def inventory_row(relative_path: str, *, device: str = "", generator: str = "") -> dict[str, str]:
    if device:
        return {
            "relative_path": relative_path,
            "dataset_name": "premier",
            "source_subset": "N3",
            "label": "authentic",
            "license_status": "cc-by-sa-4.0",
            "processing_state": "native_camera",
            "device_id": device,
            "camera_make": "Example",
            "camera_model": device,
            "lens_model": "unknown",
            "focal_length": "unknown",
            "content_category": "natural",
            "generator_paradigm": "",
            "generator_name": "",
            "generator_checkpoint": "",
            "decoder_family": "",
        }
    return {
        "relative_path": relative_path,
        "dataset_name": "genimage",
        "source_subset": generator,
        "label": "ai_generated",
        "license_status": "cc-by-nc-sa-4.0",
        "processing_state": "native_generator_export",
        "device_id": "",
        "camera_make": "",
        "camera_model": "",
        "lens_model": "",
        "focal_length": "",
        "content_category": "natural",
        "generator_paradigm": "diffusion",
        "generator_name": generator,
        "generator_checkpoint": "v1",
        "decoder_family": "vae",
    }


def write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(TASK8B_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


class Task8BDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, list[dict[str, str]]]:
        rows: list[dict[str, str]] = []
        index = 0
        for device in ("camera-a", "camera-b", "camera-c"):
            for image_index in range(2):
                relative = f"premier/{device}-{image_index}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                patterned_image(path, index)
                rows.append(inventory_row(relative, device=device))
                index += 1
        for generator in ("sd14", "glide", "biggan"):
            for image_index in range(2):
                relative = f"genimage_ai/{generator}/{image_index}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                patterned_image(path, index)
                rows.append(inventory_row(relative, generator=generator))
                index += 1
        inventory = root / "sources.csv"
        write_inventory(inventory, rows)
        return inventory, rows

    def _extracted_fixture(self, root: Path) -> None:
        index = 0
        for device in ("D01_Apple_iPhone", "D02_Google_Pixel", "F01_Samsung_Galaxy"):
            for image_index in range(3):
                path = root / f"premier/N1/{device}/{image_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                patterned_image(path, index)
                index += 1
        for generator in ("ADM", "BigGAN", "GLIDE"):
            for image_index in range(3):
                path = root / f"genimage_ai/{generator}/train/ai/cat/{image_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                patterned_image(path, index)
                index += 1
        nature = root / "genimage_ai/ADM/train/nature/cat/excluded.png"
        nature.parent.mkdir(parents=True, exist_ok=True)
        patterned_image(nature, index)

    def test_preparation_builds_balanced_reviewable_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._extracted_fixture(root)
            output = root / "sources.csv"
            report_path = root / "inventory_preparation.json"
            report = prepare_task8b_inventory(
                task8b_root=root,
                output_path=output,
                report_path=report_path,
                supported_extensions={".png"},
                max_per_generator=2,
                seed=42,
            )

            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertTrue(report["review_required"])
        self.assertEqual(report["selected"]["authentic"], 6)
        self.assertEqual(report["selected"]["ai_generated"], 6)
        self.assertEqual(report["selected"]["devices"], 3)
        self.assertEqual(report["selected"]["generator_counts"], {"ADM": 2, "BigGAN": 2, "GLIDE": 2})
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(len(rows), 12)
        self.assertNotIn("nature", " ".join(row["relative_path"] for row in rows))

    def test_preparation_does_not_overwrite_reviewed_inventory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._extracted_fixture(root)
            output = root / "sources.csv"
            output.write_text("reviewed\n", encoding="utf-8")
            with self.assertRaisesRegex(Task8BPreparationError, "already exists"):
                prepare_task8b_inventory(
                    task8b_root=root,
                    output_path=output,
                    report_path=root / "report.json",
                    supported_extensions={".png"},
                    max_per_generator=2,
                    seed=42,
                )

    def test_archive_extraction_is_bounded_deterministic_and_ai_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            archive_path = root / "ADM.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(8):
                    image = source / f"ai-{index}.png"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    patterned_image(image, index)
                    archive.write(image, f"ADM/train/ai/category/{index}.png")
                nature = source / "nature.png"
                patterned_image(nature, 20)
                archive.write(nature, "ADM/train/nature/category/excluded.png")

            report = extract_genimage_ai_sample(
                archive_path=archive_path,
                generator_name="ADM",
                output_root=root / "genimage_ai",
                report_path=root / "extract.json",
                limit=3,
                seed=42,
                supported_extensions={".png"},
            )
            extracted = sorted((root / "genimage_ai/ADM").rglob("*.png"))

        self.assertEqual(report["eligible_ai_members"], 8)
        self.assertEqual(report["selected_count"], 3)
        self.assertEqual(len(extracted), 3)
        self.assertTrue(all("nature" not in str(path) for path in extracted))

    def test_manifest_and_grouped_splits_preserve_license_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, _ = self._fixture(root)
            records, audit = build_task8b_manifest(
                dataset_root=root,
                inventory_path=inventory,
                allow_noncommercial_genimage=True,
                supported_extensions={".png"},
                minimum_images_per_device=2,
                perceptual_distance=0,
            )
            assigned, report = assign_task8b_splits(
                records,
                seed=42,
                fractions={"seed_train": 0.75, "selection_val": 0.125, "heldout_test": 0.125},
            )

        self.assertEqual(audit["label_counts"], {"ai_generated": 6, "authentic": 6})
        self.assertEqual(audit["device_count"], 3)
        self.assertEqual(audit["generator_count"], 3)
        self.assertEqual(report["split_group_overlap_count"], 0)
        self.assertTrue(all(row["license_verified"] == "true" for row in assigned))
        for group in {row["split_group_id"] for row in assigned}:
            self.assertEqual(len({row["split"] for row in assigned if row["split_group_id"] == group}), 1)
        for device in {row["device_id"] for row in assigned if row["device_id"]}:
            self.assertEqual(
                len({row["split"] for row in assigned if row["device_id"] == device}),
                1,
            )

    def test_matched_views_are_deterministic_equal_size_lossless_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, label in enumerate(("authentic", "ai_generated")):
                image_path = root / f"source-{index}.png"
                image = Image.new("RGB", (320 + index, 300 + index), (20, 40, 60))
                ImageDraw.Draw(image).rectangle((40, 50, 220, 250), fill=(index * 80, 90, 120))
                image.save(image_path)
                rows.append(
                    {
                        "sample_id": f"source-{index}__source_original",
                        "source_id": f"source-{index}",
                        "source_path": str(image_path),
                        "image_path": str(image_path),
                        "image_view": "source_original",
                        "sha256": str(index),
                        "eligible_for_split": "true",
                        "license_verified": "true",
                        "label": label,
                        "split": "seed_train",
                        "dataset_name": "premier" if index == 0 else "genimage",
                        "physical_source_status": (
                            "native_camera" if index == 0 else "native_generator_export"
                        ),
                        "split_group_id": f"group-{index}",
                        "format": "PNG",
                        "original_format": "PNG",
                        "width": 320 + index,
                        "height": 300 + index,
                    }
                )
            source_manifest = root / "source.csv"
            write_manifest(source_manifest, rows)
            report = build_task8b_matched_views(
                source_manifest=source_manifest,
                output_root=root / "views",
                output_manifest=root / "matched.csv",
                report_path=root / "matched.json",
                size=256,
                seed=42,
                perceptual_distance=0,
            )
            with (root / "matched.csv").open(newline="", encoding="utf-8") as stream:
                matched = list(csv.DictReader(stream))

        self.assertEqual(report["materialized_count"], 2)
        self.assertTrue(report["all_output_file_sizes_equal"])
        self.assertTrue(all(row["width"] == row["height"] == "256" for row in matched))
        self.assertTrue(all(row["format"] == "TIFF" for row in matched))
        self.assertTrue(all("resize=false" in row["realized_parameters"] for row in matched))

    def test_readiness_audit_separates_source_prnu_and_training_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, _ = self._fixture(root)
            records, _ = build_task8b_manifest(
                dataset_root=root,
                inventory_path=inventory,
                allow_noncommercial_genimage=True,
                supported_extensions={".png"},
                minimum_images_per_device=2,
                perceptual_distance=0,
            )
            assigned, _ = assign_task8b_splits(
                records,
                seed=42,
                fractions={"seed_train": 0.75, "selection_val": 0.125, "heldout_test": 0.125},
            )
            manifest = root / "manifest.csv"
            write_manifest(manifest, assigned)
            report = audit_task8b_readiness(
                manifest_path=manifest,
                output_path=root / "readiness.json",
                readiness={
                    "minimum_rows_per_label": 6,
                    "minimum_authentic_devices": 3,
                    "minimum_generator_families": 3,
                    "minimum_prnu_training_devices": 1,
                    "max_label_count_ratio": 1.0,
                    "max_nuisance_balanced_accuracy": 1.0,
                    "minimum_ca_metadata_fraction": 0.0,
                    "minimum_ca_edge_rich_fraction": 0.0,
                },
                minimum_images_per_device=2,
            )

        self.assertTrue(report["source_ready"])
        self.assertTrue(report["prnu_reference"]["ready"])
        self.assertTrue(report["training_ready"])
        self.assertEqual(report["nuisance"]["status"], "evaluated")
        self.assertFalse(report["final_test_read"])

    def test_genimage_requires_noncommercial_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, _ = self._fixture(root)
            with self.assertRaisesRegex(DatasetContractError, "non-commercial"):
                build_task8b_manifest(
                    dataset_root=root,
                    inventory_path=inventory,
                    allow_noncommercial_genimage=False,
                    supported_extensions={".png"},
                    minimum_images_per_device=2,
                )

    def test_genimage_nature_branch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "genimage_ai/sd14/nature/example.png"
            path.parent.mkdir(parents=True)
            patterned_image(path, 1)
            inventory = root / "sources.csv"
            write_inventory(
                inventory,
                [inventory_row("genimage_ai/sd14/nature/example.png", generator="sd14")],
            )
            with self.assertRaisesRegex(DatasetContractError, "nature/ImageNet"):
                build_task8b_manifest(
                    dataset_root=root,
                    inventory_path=inventory,
                    allow_noncommercial_genimage=True,
                    supported_extensions={".png"},
                    minimum_images_per_device=2,
                )

    def test_prnu_references_read_seed_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = []
            for index in range(3):
                path = root / f"image-{index}.png"
                patterned_image(path, index)
                image_paths.append(path)
            rows = []
            for index, path in enumerate(image_paths):
                rows.append(
                    {
                        "sample_id": f"sample-{index}",
                        "source_id": f"source-{index}",
                        "image_path": str(path),
                        "sha256": str(index),
                        "label": "authentic",
                        "split": "seed_train" if index < 2 else "heldout_test",
                        "dataset_name": "premier",
                        "eligible_for_split": "true",
                        "license_verified": "true",
                        "physical_source_status": "native_camera",
                        "device_id": "camera-a",
                    }
                )
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            report = build_training_prnu_references(
                manifest_path=manifest,
                output_root=root / "fingerprints",
                report_path=root / "report.json",
                minimum_images_per_device=2,
                reference_size=64,
            )

        self.assertEqual(report["reference_count"], 1)
        self.assertEqual(report["references"][0]["image_count"], 2)
        self.assertFalse(report["selection_or_heldout_rows_read"])

    def test_manifest_cli_writes_under_requested_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, _ = self._fixture(root)
            config = json.loads(
                (REPO_ROOT / "configs/colab.json").read_text(encoding="utf-8")
            )
            config["task8b"]["minimum_images_per_device"] = 2
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            artifact_root = root / "artifacts/task8b"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/build_task8b_manifest.py"),
                    "--config",
                    str(config_path),
                    "--dataset-root",
                    str(root),
                    "--inventory",
                    str(inventory),
                    "--manifest",
                    str(artifact_root / "manifests/source_manifest_split.csv"),
                    "--audit-report",
                    str(artifact_root / "audits/source_audit.json"),
                    "--split-report",
                    str(artifact_root / "audits/split_report.json"),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((artifact_root / "manifests/source_manifest_split.csv").is_file())
            self.assertTrue((artifact_root / "audits/source_audit.json").is_file())
            self.assertTrue((artifact_root / "audits/split_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
