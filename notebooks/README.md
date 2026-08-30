# Colab notebooks

Notebooks are thin launchers for the Python modules in `src/`; training logic does not live only in a notebook.

Start with `00_colab_setup.ipynb`, using the official Google Colab VS Code extension:

1. open the shared dataset folder `1c-IVvAiHlApA49CtU3QQH9XqQDmkbO8U`, add a shortcut at `My Drive/hackathon_data`, and keep that shortcut name stable;
2. open or create a notebook in this directory;
3. select **Kernel -> Colab -> Auto Connect** and sign in;
4. select a GPU-backed runtime;
5. run **Colab: Mount Google Drive to Server...** from the command palette;
6. clone/pull this repository into `/content/cya-techjam26` on the remote runtime;
7. run `make install-colab` from that remote checkout;
8. copy the active dataset archive or subset from Drive into `/content/hackathon_data`;
9. run `make smoke` before any extraction or training command;
10. keep temporary caches under `/content` and sync checkpoints/metrics to `/content/drive/MyDrive/cya-techjam26/artifacts`.

The Colab virtual machine is disposable. Drive holds source data and durable outputs, while `/content` holds the active repository, extracted dataset, feature cache, and other high-I/O working files.

Task 8B can use source files extracted directly at
`/content/hackathon_data/raw/task8b`, without modifying the shared Drive. An
optional personal source copy can live at
`My Drive/cya-techjam26-data/raw/task8b`. Results are written first to
`/content/cya-techjam26/artifacts/task8b`. Copy completed results into the existing
Drive artifact root at `My Drive/cya-techjam26/artifacts/task8b`. The repository
does not download or upload the licensed datasets automatically; follow
[`docs/data/task8b_dataset.md`](../docs/data/task8b_dataset.md) for the inventory
and license checks.

After setup passes, use `01_task2_data_contract.ipynb` to audit the immutable SID sources, freeze grouped splits, and compare the two matched-clean encoding candidates. Its real-data assertions are intentionally strict and stop before derivation when source counts, corruption, C2PA scanning, or cross-label duplicate checks fail.

Then use `02_stage_a_clip.ipynb` to regenerate disposable matched-clean pilots when needed, run frozen-CLIP Stage A for both matching policies and three seeds, create locked Task 5 clean-selection reports, and sync completed artifacts to Drive.

After Stage A selects a matching policy, `03_rine_stage_b.ipynb` runs the frozen-CLIP intermediate-layer ablation on the same splits and seeds. It records a provisional clean decision while leaving final 50/50 retention pending the Task 3 transform rows.

`04_frequency_stage1.ipynb` extracts the deterministic FFT/DCT/residual feature bank, trains magnitude-only and bounded-phase variants across three seeds, audits nuisance overlap, and leaves the Stage 1 early exit disabled.

`05_auxiliary_stage_c.ipynb` extracts RGB/Lab correlation plus confidence-masked PRNU/optics diagnostics. On the matched-Q96 handoff it trains only the three color variants; physical-family training remains blocked until label-independent source-original eligibility is available.

`06_task8b_native_physical.ipynb` uses manually downloaded PREMIER and GenImage
AI-only files from Colab-local storage (or an optional personal Drive copy),
creates a draft inventory that requires manual review, builds the
device/generator-grouped manifest, builds the 256 px matched TIFF view, runs the
source and matched nuisance gates, compares single-image and multi-image PRNU
without binary labels, records the retain/reject decision, and syncs reports to
the existing Drive artifact root. It stops before model fitting whenever source,
nuisance, or physical-estimator validation fails. The current licensed pilot
finishes with no physical feature retained.

`07_texture_stage_d.ipynb` runs after Task 2's fixed-Q96 pilot manifest exists.
It is a thin launcher only: every cell calls `scripts/extract_texture_features.py`,
`scripts/train_texture_pilot.py`, or `scripts/compare_texture_pilot.py` (the same
`task9-*` Make targets), and no model or training code is inlined in the notebook
itself. It trains and compares only on the fixed-Q96 matched-clean manifest's
`seed_train` and `selection_val` splits, never on `self_train_pool`, sealed
`final_test`, source-original images, Task 3 robustness variants, or Task 8B
data; CLIP stays fully frozen throughout.

The notebook mounts Drive, stages the manifest and matched-clean images locally,
extracts and caches frozen global RINE features plus up to four non-overlapping
112 px source patches (upsampled by the locked CLIP processor to 336 px, for a
five-view-per-image encoding budget) exactly once under `/content`, then trains
the three variants (`global_only`, `local_only`, `global_local`) across seeds
42/43/44 — nine runs total reusing the same caches. Each run is copied to the
shared Drive artifact root (`My Drive/cya-techjam26/artifacts/task9`) only after
it completes, so an interrupted or not-yet-started run never creates an empty
Drive directory; rerunning the notebook after a runtime reset skips any run
already found complete either locally or in that Drive tree. After all nine
runs finish, it applies the deterministic clean gate. A
`continue_to_robustness_design` decision only authorizes a later, separately
specified Task 3 robustness continuation — it does not retain Task 9 by itself.
A `reject_texture_clean_gate` decision means Task 9 remains tested-and-rejected
and RINE (Task 6) remains the retained global representation. No real Colab run
of this notebook has been recorded yet.
