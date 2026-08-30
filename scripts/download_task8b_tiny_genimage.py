#!/usr/bin/env python3
"""Download a bounded generator-labelled Tiny-GenImage AI-only sample."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import write_json  # noqa: E402


DATASET_ID = "TheKernel01/Tiny-GenImage"
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_ID}"
API_ROOT = "https://datasets-server.huggingface.co"
GENERATOR_CODES = {
    "ADM": 1,
    "BigGAN": 2,
    "GLIDE": 3,
    "Midjourney": 4,
    "SD15": 6,
    "VQDM": 7,
    "Wukong": 8,
}


class TinyGenImageDownloadError(RuntimeError):
    """Raised when the bounded Tiny-GenImage download cannot be verified."""


def _read_json(url: str, attempts: int = 4) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                return json.load(response)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _collect_rows(
    generators: tuple[str, ...],
    per_generator: int,
    split: str,
    skip_per_generator: int,
) -> dict[str, list[dict[str, Any]]]:
    collected = {generator: [] for generator in generators}
    by_code = {GENERATOR_CODES[generator]: generator for generator in generators}
    offset = 0
    required = per_generator + skip_per_generator
    while any(len(rows) < required for rows in collected.values()):
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_ID,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        payload = _read_json(f"{API_ROOT}/rows?{query}")
        rows = payload.get("rows", [])
        if not rows:
            raise TinyGenImageDownloadError(
                "Dataset rows ended before every generator reached the requested count"
            )
        for item in rows:
            row = item["row"]
            generator = by_code.get(row["generator"])
            if generator and row["label"] == 1 and len(collected[generator]) < required:
                collected[generator].append(
                    {
                        "row_idx": item["row_idx"],
                        "image": row["image"],
                    }
                )
        offset += len(rows)
    return {
        generator: rows[skip_per_generator:required]
        for generator, rows in collected.items()
    }


def _download_one(item: dict[str, Any], target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(item["image"]["src"], timeout=90) as response:  # noqa: S310
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(target)
            return {
                "row_idx": item["row_idx"],
                "path": str(target.resolve()),
                "declared_width": item["image"].get("width"),
                "declared_height": item["image"].get("height"),
                "bytes": target.stat().st_size,
            }
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download_sample(
    *,
    output_root: Path,
    report_path: Path,
    generators: tuple[str, ...],
    per_generator: int,
    split: str,
    workers: int,
    skip_per_generator: int = 0,
    append: bool = False,
) -> dict[str, Any]:
    if per_generator < 1 or workers < 1 or skip_per_generator < 0:
        raise TinyGenImageDownloadError(
            "per_generator and workers must be positive; skip_per_generator cannot be negative"
        )
    unknown = set(generators) - set(GENERATOR_CODES)
    if unknown:
        raise TinyGenImageDownloadError(f"Unsupported generators: {sorted(unknown)}")
    existing = [
        output_root / generator
        for generator in generators
        if (output_root / generator).exists()
    ]
    if existing and not append:
        raise TinyGenImageDownloadError(
            f"Generator destinations already exist; review before replacing: {existing}"
        )

    selected = _collect_rows(generators, per_generator, split, skip_per_generator)
    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {generator: [] for generator in generators}
    with tempfile.TemporaryDirectory(prefix="tiny-genimage-", dir=output_root) as temporary:
        temporary_root = Path(temporary)
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for generator, rows in selected.items():
                for item in rows:
                    source_suffix = Path(
                        urllib.parse.urlparse(item["image"]["src"]).path
                    ).suffix.lower()
                    suffix = source_suffix if source_suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
                    target = (
                        temporary_root
                        / generator
                        / split
                        / "ai"
                        / f"tiny_genimage_{item['row_idx']:08d}{suffix}"
                    )
                    future = executor.submit(_download_one, item, target)
                    futures[future] = generator
            for future in as_completed(futures):
                generator = futures[future]
                records[generator].append(future.result())
        for generator in generators:
            source = temporary_root / generator
            destination = output_root / generator
            if not append:
                source.replace(destination)
                continue
            for source_file in source.rglob("*"):
                if not source_file.is_file():
                    continue
                target = destination / source_file.relative_to(source)
                if target.exists():
                    raise TinyGenImageDownloadError(
                        f"Append would overwrite an existing image: {target}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file.replace(target)

        for rows in records.values():
            for row in rows:
                temporary_path = Path(row["path"])
                relative = temporary_path.relative_to(temporary_root)
                row["path"] = str((output_root / relative).resolve())

    for generator in records:
        records[generator].sort(key=lambda row: row["row_idx"])
    report = {
        "dataset_id": DATASET_ID,
        "dataset_page": DATASET_PAGE,
        "declared_license": "cc-by-nc-sa-4.0",
        "approved_use": "noncommercial_research_hackathon",
        "source_kind": "third_party_genimage_repackaging",
        "split": split,
        "per_generator": per_generator,
        "skip_per_generator": skip_per_generator,
        "append": append,
        "generator_counts": {
            generator: len(rows) for generator, rows in records.items()
        },
        "total_count": sum(len(rows) for rows in records.values()),
        "output_root": str(output_root.resolve()),
        "rows": records,
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["ADM", "BigGAN", "Midjourney", "Wukong"],
    )
    parser.add_argument("--per-generator", type=int, default=150)
    parser.add_argument("--skip-per-generator", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = download_sample(
            output_root=args.output_root,
            report_path=args.report,
            generators=tuple(args.generators),
            per_generator=args.per_generator,
            split=args.split,
            workers=args.workers,
            skip_per_generator=args.skip_per_generator,
            append=args.append,
        )
    except Exception as exc:
        print(f"TINY-GENIMAGE DOWNLOAD ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"generator_counts": report["generator_counts"], "total_count": report["total_count"]}, indent=2))
    print(f"Download report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
