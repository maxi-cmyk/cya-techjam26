# Task 8B Native Physical-Signal Dataset

Task 8B is a separate, non-blocking evidence track for PRNU first and chromatic
aberration second. It does not replace SID_Set and does not authorize retraining
Tasks 4–7. The RINE backbone remains frozen; only physical-feature projections
and fusion weights may be trained after the data audit passes.

## Verified sources and licenses

The source pages and license text were rechecked on 2026-08-30.

| Source | Accepted Task 8B content | License | Restriction |
|---|---|---|---|
| [PREMIER Dataset v3](https://sites.google.com/unitn.it/premier/resources/datasets) | Accessible native image subsets N1 and N2; N3 is optional if it becomes accessible | CC BY-SA 4.0 | Preserve attribution and comply with ShareAlike when distributing adapted dataset material |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | `ai` branches only, grouped by generator | CC BY-NC-SA 4.0 plus the published dataset terms | Non-commercial research, teaching, and scientific-publication use only |
| [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) | Approved storage-limited third-party repackaging of GenImage AI rows | Declared CC BY-NC-SA 4.0 | Preserve Hugging Face row IDs and repackaging provenance; nuisance audit must pass before training |

PREMIER altered/social-media subsets and GenImage `nature`/ImageNet branches are
excluded. Mixed, AI-edited, composited, tampered, and ambiguous-origin images
remain outside the binary contract.

## Explicit assumptions

1. This hackathon run is non-commercial research, so GenImage's
   CC BY-NC-SA 4.0 restriction is acceptable. GenImage must be replaced or
   separately licensed before commercial use.
2. PREMIER metadata can identify each included physical device. Rows without a
   device ID are rejected.
3. GenImage generator identity can be preserved for every included AI image.
   Rows without a generator name are rejected.
4. The first implementation supports JPEG, PNG, and TIFF sources that Pillow can
   decode deterministically. RAW and HEIC originals may be preserved in Drive,
   but they must not enter the manifest until a decoder and version are pinned.
5. PREMIER is sufficient to begin the PRNU track. It is not assumed to provide
   adequate lens model, focal-length, corrected/uncorrected, or calibration-target
   coverage for chromatic aberration. CA remains deferred unless its audit passes.
6. Dataset presence, download completion, and upstream metadata accuracy require
   manual verification in Drive/Colab; repository tests validate the contract with
   licensed-style fixtures but do not prove the external files are present.
7. The initial readiness thresholds are conservative pilot gates: at least 500
   eligible rows per label, 10 authentic devices, four generator families, five
   seed-training devices suitable for references, and no more than a 1.25 class
   count ratio. These are predeclared in `configs/colab.json` and must not be
   relaxed after looking at model results.
8. A nuisance-only balanced accuracy above 0.65 blocks physical fusion training.
   It does not invalidate the native source collection; it means export, format,
   resolution, or file-size matching still needs work.

## Public downloads and local storage

Use the official public folders:

- [Accessible PREMIER Google Drive folder](https://drive.google.com/drive/folders/1Fb1ayRJnHGGdUI2TGfzZGQ9MRXh8a6Rj)
- [GenImage Google Drive folder](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing)

The prepared Task 8B transport lives in the writable hackathon data folder
[`task8b`](https://drive.google.com/drive/folders/1ANvP41AjiTztc0Hhv3rT-XjKguG0YNUp).
Its canonical payload is the ordered set
`task8b_manifest.tar.gz.upload-aa` through
`task8b_manifest.tar.gz.upload-aw`: 23 chunks totalling 2,135,569,689 bytes.
Together they reconstruct a SHA-256-locked archive containing `sources.csv` and
exactly the 1,280 licensed inventory images. The Drive `premier/N1` and
`premier/N2` directory placeholders do not need to contain individual images;
`06_task8b_native_physical.ipynb` downloads, verifies, combines, and safely
extracts the chunks into `/content/hackathon_data/raw/task8b`. It then verifies
all inventory paths and refuses to continue if either local PREMIER split is
empty.

The accessible PREMIER source folder was manually checked on 2026-08-30 and
does not contain N3, even though the project page describes that subset. Prepare
N1 and N2 now; N3 is not required and may be added later only if it becomes
publicly accessible under the documented license. For GenImage, acquire each
selected generator's `train/ai` and `val/ai` branches. Do not download or include
`nature` branches for Task 8B.

### Storage-limited GenImage pilot

Do not use Google Drive's **Download all** action on the GenImage root. The
official release is million-scale and the Task 8B pilot does not need every
generator or every image. The four generator `.zip` files obtained from the
linked subfolder were inspected on 2026-08-30 and proved to be only the final
central-directory volumes of multi-part ZIP archives. They contain no local AI
payload and are unusable without all companion volumes. Do not treat a readable
`zipinfo` listing as proof that an archive is extractable.

The original intended four-generator selection was:

1. BigGAN
2. ADM
3. Stable Diffusion V1.4
4. VQDM

This gives four independent generator groups and covers GAN, pixel-space
diffusion, latent diffusion, and vector-quantized diffusion families. If the
release separates training and validation archives, either branch is acceptable
because Task 8B creates new generator-grouped splits; preserve the upstream
branch name in the path. Extract only entries below `train/ai` or `val/ai`.

Those unusable final volumes were permanently removed after explicit approval.

Use the bounded ZIP extractor only with a complete, independently extractable
archive. Run `unzip -t <archive>` first; multi-part warnings or bad local-header
offsets block extraction. Downloading the dozens of missing GenImage companion
volumes is not recommended for the storage-limited pilot. A smaller public
repackaging or replacement source requires a separate provenance and license
decision before it enters `sources.csv`.

Tiny-GenImage was explicitly approved for the local pilot on 2026-08-30. The
bounded downloader retained 160 rows each from ADM, BigGAN, Midjourney, and
Wukong through the Hugging Face Dataset Viewer API. The resulting 640-image
synthetic pool occupies about 42 MB rather than downloading the full 8.36 GB
Parquet export. It is recorded as a third-party GenImage repackaging, not as
bytes downloaded from the official multi-part archives.

Do not unzip a generator archive. Leave it compressed in `_incoming` or point to
it through a mounted personal-Drive shortcut, then run one bounded extraction per
archive:

```bash
make task8b-extract-genimage \
  GENIMAGE_ARCHIVE="/content/hackathon_data/raw/task8b/_incoming/ADM.zip" \
  GENIMAGE_GENERATOR="ADM" \
  GENIMAGE_LIMIT=200
```

Repeat for the other selected generators. The extractor reads the ZIP directory,
deterministically selects only supported images under an `ai` branch, rejects
unsafe paths and oversized members, verifies each selected image, and writes an
audit report. It never expands the full archive and refuses to overwrite an
existing generator directory.

The expected extracted layout is:

```text
/content/hackathon_data/raw/task8b/
├── premier/
│   ├── N1/<Dxx_device_folder>/...
│   ├── N2/<Fxx_device_folder>/...
│   └── N3/<device_folder>/...           optional; omit when unavailable
└── genimage_ai/
    ├── ADM/{train,val}/ai/<category>/...
    ├── BigGAN/{train,val}/ai/<category>/...
    ├── Stable_Diffusion_v1.4/{train,val}/ai/<category>/...
    └── VQDM/{train,val}/ai/<category>/...
```

Keep the upstream files unchanged inside those wrapper directories. PREMIER
device folders must begin with their real alphanumeric identifier, such as `D40`
in N1 or `F01` in N2. The inventory helper deliberately rejects files when it
cannot infer both a PREMIER subset and device ID. Use at least 10 authentic
devices, at least four generator families,
and at least 500 usable files per class. The default inventory cap is 830 images
per generator and class counts are balanced automatically.

Task 8B reuses the existing data and artifact roots from `configs/colab.json`.
No additional Drive root is introduced.

```text
/content/hackathon_data/
└── raw/task8b/                    required Colab-local source

/content/cya-techjam26/artifacts/
└── task8b/
    ├── manifests/
    ├── audits/
    ├── features/
    ├── fingerprints/
    ├── checkpoints/
    └── reports/

Google Drive artifacts root (ID 1uv0sa041-6N-Vg8tdtb5in0GgWBR-SFz)
└── task8b/                        durable API-synced result copy

/content/drive/MyDrive/cya-techjam26-data/
└── raw/task8b/                    optional personal source copy
```

Raw images stay under `hackathon_data`. Manifests, reports, features,
fingerprints, and checkpoints stay under `artifacts/task8b`. Both local roots
are ignored by Git. The notebook syncs durable Task 8B outputs directly through
the authenticated Drive API; mounting My Drive is optional. The reproducible
`matched_views/images` cache stays local rather than creating more than one
thousand Drive files.

## Curated inventory contract

Place `sources.csv` at `hackathon_data/raw/task8b/sources.csv`. Each row must
contain all of these columns:

Start from [`task8b_sources_template.csv`](task8b_sources_template.csv), remove
the example rows, and populate it from the upstream metadata inventories.

```text
relative_path,dataset_name,source_subset,label,license_status,processing_state,
device_id,camera_make,camera_model,lens_model,focal_length,content_category,
generator_paradigm,generator_name,generator_checkpoint,decoder_family
```

Accepted authentic example:

```csv
premier/N1/D01/image001.jpg,premier,N1,authentic,cc-by-sa-4.0,native_camera,D01,Apple,iPhone4s,unknown,unknown,natural,,,,
```

Accepted AI example:

```csv
genimage_ai/Stable_Diffusion_v1.4/ai/example.png,genimage,Stable_Diffusion_v1.4,ai_generated,cc-by-nc-sa-4.0,native_generator_export,,,,,natural,diffusion,Stable_Diffusion_v1.4,v1.4,VAE
```

The importer rejects unsafe paths, missing files, undeclared or mismatched
licenses, unsupported formats, duplicate inventory paths, non-native PREMIER
rows, GenImage real/nature rows, missing device IDs, and missing generator names.
PREMIER devices with fewer than the configured minimum number of images remain
in the audit but are ineligible for splitting.

Current local preparation evidence: 640 synthetic rows and 640 balanced PREMIER
inventory rows produced a 1,280-row manifest. After duplicate exclusions, 535
authentic and 640 synthetic rows remain eligible across 13 devices and ADM,
GLIDE, Midjourney, and Wukong. BigGAN is retained only as a low-resolution stress
source. The source-original nuisance model reaches 1.0 balanced accuracy, so
those files are never used as the binary training view.

The matched manifest uses deterministic 256 px crop-only RGB views saved as
metadata-stripped, uncompressed TIFF. No source is resized. Three near-uniform
crops are excluded by the label-independent RGB-standard-deviation rule, and
same-label near duplicates spanning splits are resolved to one primary. The
result has 1,164 eligible rows, identical 196,748-byte files, no eligible split
overlap, and nuisance balanced accuracy 0.50.

PRNU validation uses ten seed-training devices, ten disjoint reference images
per device, and 316 query images. It never uses authentic/AI labels or reads
selection/held-out rows. Classical multi-image correlation reaches AUC 0.538;
the existing single-image proxy reaches AUC 0.543. Both miss the predeclared
0.60 gate. CA coverage is zero. The final decision is therefore no physical
feature retained and no fusion training.

## Split and leakage contract

- Authentic rows group by complete physical device ID. Variable EXIF model
  strings can never split one sensor across multiple groups.
- AI rows group by complete generator family. A generator can never cross splits.
- Task 8B uses its own `seed_train`, `selection_val`, and `heldout_test` splits.
  It never reads the competition `final_test` split.
- Exact or near duplicates spanning different device/generator groups are marked
  `excluded_review` rather than assigned.
- Multi-image PRNU references use eligible PREMIER `seed_train` rows only.
  Selection and held-out images never contribute to a device fingerprint.

## Commands

After extracting the datasets into the local layout, generate the draft
inventory:

```bash
cd /content/cya-techjam26
make task8b-inventory
```

Review both files before continuing:

```text
/content/hackathon_data/raw/task8b/sources.csv
/content/cya-techjam26/artifacts/task8b/audits/inventory_preparation.json
```

Correct inferred metadata against the upstream metadata, resolve every rejected
path that should be included, and confirm the license fields. The command will
not overwrite an existing reviewed `sources.csv`; use the CLI's `--overwrite`
flag only when intentionally regenerating the draft. Then run:

```bash
make task8b-prepare
make task8b-prnu-references
make task8b-matched
make task8b-prnu-validate
make task8b-decision
```

`task8b-prepare` builds the split manifest and requires the source-readiness audit
to pass. Its report distinguishes `source_ready`, `prnu_reference.ready`,
`chromatic_aberration.ready`, and `training_ready`. The second command reruns the
source gate and then builds training-only reference fingerprints. The remaining
commands materialize and audit the matched view, validate PRNU without binary
labels, and record the fail-closed retain/reject decision. The final notebook
cell upserts the durable result directories into the existing Google Drive
artifacts root for persistent storage.

None of these Make commands downloads third-party data or uploads results automatically.
The thin Colab launcher
[`06_task8b_native_physical.ipynb`](../../notebooks/06_task8b_native_physical.ipynb)
performs local staging, runs these gates, and syncs completed Task 8B artifacts
back into the existing Drive artifact root.

## Readiness meanings

- `source_ready`: licenses, provenance, class volume, device/generator coverage,
  split completeness, and leakage checks pass.
- `prnu_reference.ready`: `source_ready` passes and enough eligible
  seed-training devices have the minimum image count.
- `chromatic_aberration.ready`: `source_ready` passes and the configured lens,
  focal-length, and edge-rich metadata fractions pass. Corrected/uncorrected
  calibration coverage and estimator validation are still required afterward.
- `training_ready`: `source_ready` passes and a nuisance-only model trained on
  `seed_train` cannot exceed the declared balanced-accuracy threshold on either
  `selection_val` or `heldout_test`.
