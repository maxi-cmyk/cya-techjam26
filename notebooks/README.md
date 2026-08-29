# Colab notebooks

Notebooks are thin launchers for the Python modules in `src/`; training logic does not live only in a notebook.

Start with `00_colab_setup.ipynb`, using the official Google Colab VS Code extension:

1. open or create a notebook in this directory;
2. select **Kernel -> Colab -> Auto Connect** and sign in;
3. select a GPU-backed runtime;
4. run **Colab: Mount Google Drive to Server...** from the command palette;
5. clone/pull this repository into `/content/cya-techjam26` on the remote runtime;
6. run `make install-colab` from that remote checkout;
7. copy the active dataset archive or subset from Drive into `/content/hackathon_data`;
8. run `make smoke` before any extraction or training command;
9. keep temporary caches under `/content` and sync checkpoints/metrics to `/content/drive/MyDrive/cya-techjam26/artifacts`.

The Colab virtual machine is disposable. Drive holds source data and durable outputs, while `/content` holds the active repository, extracted dataset, feature cache, and other high-I/O working files.
