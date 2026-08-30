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
8. run the notebook-specific preparation cell to stage its active dataset under `/content/hackathon_data`;
9. run `make smoke` before any extraction or training command;
10. keep temporary caches under `/content` and sync checkpoints/metrics to `/content/drive/MyDrive/cya-techjam26/artifacts`.

The Colab virtual machine is disposable. Drive holds source data and durable outputs, while `/content` holds the active repository, extracted dataset, feature cache, and other high-I/O working files.

Task 8B can use source files extracted directly at
`/content/hackathon_data/raw/task8b`, without modifying the shared Drive. An
optional personal source copy can live at
`My Drive/cya-techjam26-data/raw/task8b`. Results are written first to
`/content/cya-techjam26/artifacts/task8b`. The final Task 8B cell uses the same
authenticated Drive API session to upsert durable results into the existing
[`artifacts`](https://drive.google.com/drive/folders/1uv0sa041-6N-Vg8tdtb5in0GgWBR-SFz)
root, so mounting My Drive is not required. Follow
[`docs/data/task8b_dataset.md`](../docs/data/task8b_dataset.md) for the inventory
and license checks.

After setup passes, use `01_task2_data_contract.ipynb` to audit the immutable SID sources, freeze grouped splits, and compare the two matched-clean encoding candidates. Its real-data assertions are intentionally strict and stop before derivation when source counts, corruption, C2PA scanning, or cross-label duplicate checks fail.

Then use `02_stage_a_clip.ipynb` to regenerate disposable matched-clean pilots when needed, run frozen-CLIP Stage A for both matching policies and three seeds, create locked Task 5 clean-selection reports, and sync completed artifacts to Drive.

After Stage A selects a matching policy, `03_rine_stage_b.ipynb` runs the frozen-CLIP intermediate-layer ablation on the same splits and seeds. It records a provisional clean decision while leaving final 50/50 retention pending the Task 3 transform rows.

`04_frequency_stage1.ipynb` extracts the deterministic FFT/DCT/residual feature bank, trains magnitude-only and bounded-phase variants across three seeds, audits nuisance overlap, and leaves the Stage 1 early exit disabled.

`05_auxiliary_stage_c.ipynb` extracts RGB/Lab correlation plus confidence-masked PRNU/optics diagnostics. On the matched-Q96 handoff it trains only the three color variants; physical-family training remains blocked until label-independent source-original eligibility is available.

`06_task8b_native_physical.ipynb` authenticates to the prepared Task 8B Drive
folder, downloads the ordered `task8b_manifest.tar.gz.upload-*` chunks, verifies
the reconstructed archive, and safely extracts the licensed PREMIER and GenImage
AI-only files into Colab-local storage. The preparation cell validates all 1,280
`sources.csv` paths and requires populated local PREMIER N1 and N2 trees before
the notebook creates the reviewed inventory and builds the
device/generator-grouped manifest, builds the 256 px matched TIFF view, runs the
source and matched nuisance gates, compares single-image and multi-image PRNU
without binary labels, records the retain/reject decision, then runs the bounded
native-coordinate PRNU v2 estimator. Original evidence is synced to
`artifacts/task8b`; v2 reports and fingerprints are synced to the sibling
`artifacts/task8b_v2` folder under the same Drive root. Neither upload requires
a mounted Drive. Durable
audits, manifests, fingerprints, reports, features, and checkpoints are synced;
the reproducible matched-view image cache remains local. It stops before model
fitting whenever source, nuisance, or physical-estimator validation fails. The
current licensed pilot finishes with no physical feature retained.

`07_robustness_rerun.ipynb` runs the post-Task-3 robustness milestone without
opening `final_test`. It materializes and validates all independent transform
cells, evaluates the existing Stage A and RINE checkpoints, retrains controlled
RINE across seeds 42/43/44, extracts the retained frequency magnitude/residual
and Lab features, applies incremental fusion gates, and freezes the evidence
needed for the Task 9 handoff. Task 10 calibration and packaging remain out of
scope.

`08_prnu_v2_binary.ipynb` is a clean Colab GPU **Run all** workflow. It mounts
Drive, stages only the selected 2,000 raw sources, byte-verifies regenerated
fixed-Q96 views against the Notebook 07 manifest, and rebuilds or safely reuses
the 19,460-image transform bank on Colab-local storage. Before doing that work,
it requires a label-free PREMIER device-signal pass at the same 256 px protocol,
reusing a verified Drive report or producing one from already-staged Task 8B
data. It restores each durable controlled-RINE parent when available or retrains
only a missing seed locally, then applies the predeclared 256 px matched-clean
balanced-coverage gate and extracts reference-free single-image PRNU-v2 summaries,
trains PRNU-only and RINE+PRNU across seeds 42/43/44, applies the locked 50/50
and per-class gates, then hash-verifies the compact Drive sync, including any
reconstructed parent. The transform cache stays local. It never uses device IDs,
enrolled-camera PCE, or `final_test`. Deliberately downscaled robustness views
remain in the locked evaluation and use explicit zero-valued validity/confidence
masks when they cannot support a 256 px crop. If matched-clean readiness fails,
the notebook records and syncs `blocked_data_readiness` without training. The
earlier 512 px audit remains preserved as evidence of an incompatible protocol;
it is not treated as evidence that the 256 px experiment failed.

The completed Notebook 08 run passes the 256 px label-free device test (AUC
0.8593; top-1 0.6566 versus 0.10 random), extracts all 20,850 development rows,
and keeps `final_test` sealed. PRNU-only reaches a 78.09% mean locked score.
RINE+PRNU is rejected at 33.43% mean versus 99.81% for controlled RINE after
seeds 42 and 43 collapse; PRNU-v2 remains diagnostic-only.
