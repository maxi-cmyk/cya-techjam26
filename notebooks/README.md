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
