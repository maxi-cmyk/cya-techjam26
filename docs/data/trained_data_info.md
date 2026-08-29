# Hackathon Dataset — README

Track 5: AI-generated image detection (real vs. synthetic classifier)

## Folder structure

```
hackathon_data/
├── raw/
│   └── sid_set/
│       ├── images/          20,000 original images, untouched
│       └── labels.csv        filename, label (int), label_name
│
├── cleaned/
│   └── sid_set/
│       ├── images/          19,998 images (2 exact duplicates removed), bias-corrected
│       ├── labels.csv
│       ├── train/
│       │   ├── images/      15,998 images (8,000 real / 7,998 synthetic)
│       │   └── labels.csv
│       └── val/
│           ├── images/      4,000 images (2,000 real / 2,000 synthetic)
│           └── labels.csv
│
└── wildfake/                 empty — attempted, blocked by ModelScope registration, parked for later
```

**Use `cleaned/sid_set/train/` and `cleaned/sid_set/val/` for training and evaluation.**
Don't mix train and val — val exists specifically to check the model on images it hasn't trained on.

## Source dataset

[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — real photos + full-synthetic images
(FLUX, Kandinsky, SDXL, AbsoluteReality, etc.), pulled via Hugging Face streaming so the full
~200K-image dataset was never fully downloaded.

**Tampered images excluded.** SID_Set has a third class (`label == 2`, tampered/edited real photos),
which we deliberately left out — the competition organizers confirmed test images are strictly
real or fully synthetic, so training on a class we'll never be evaluated on would waste model
capacity and could blur the decision boundary.

## What was done to the data

1. **Sampled** 10,000 real + 10,000 synthetic images from SID_Set (streamed, not bulk downloaded)
2. **Resized** every image to 336×336 (matches CLIP ViT-L/14 input size)
3. **Deduplicated** — found and removed 2 exact-duplicate images (both synthetic)
4. **Checked for corruption** — 0 unreadable/broken files found
5. **Checked for a real/synthetic bias signal** — found one: real images averaged 34.2KB vs.
   synthetic's 26.8KB at identical resolution (real photos compress less efficiently than
   diffusion output). Left unfixed, a classifier could partially learn to key off file size /
   compression artifacts as a shortcut for the real/fake label, rather than genuine visual cues —
   this typically shows up as good accuracy on this dataset but poor generalization to new images
   or under the robustness harness (especially the JPEG compression test).
6. **Fixed the bias** — real images were re-compressed (JPEG quality reduced per-image until each
   hit target size) to bring their size distribution in line with synthetic's. Synthetic images
   were left untouched, since they're the harder class to source and we didn't want to risk
   degrading the generation artifacts the classifier needs to learn to detect.
7. **Split 80/20 into train/val**, stratified by class (so both splits stay balanced), shuffled
   with a fixed random seed (42) so the split is reproducible if anyone needs to regenerate it.

This is a **lightweight approximation of the DDA paper's frequency-alignment idea** (closing a
real/fake compression gap), not the full DDA pipeline (which uses VAE reconstruction + per-image
pixel mixup to generate literal paired counterparts). We checked this was a reasonable trade-off
given hackathon time/compute constraints — full VAE reconstruction was estimated at 70–130+
minutes with real setup risk, for a benefit that hadn't been confirmed necessary via diagnostics.

## Known limitations / what hasn't been checked yet

- Only the file-size/compression bias has been measured and corrected. Other potential bias
  signals (color histogram differences, blur/sharpness differences, pixel intensity patterns)
  haven't been checked yet — no evidence they exist, but no evidence they don't either.
- Only one dataset (SID_Set) is represented. WildFake was intended as a second source for
  generator diversity but downloading was blocked by ModelScope's registration flow; a subset
  (`Real/ffhq.zip`, `Diffusion_based/DDIM.zip`, `Diffusion_based/DDPM.zip`, ~15GB total) was
  identified as a good target if this gets revisited.
- val/ was split from the same SID_Set sample as train/ — it's useful for catching overfitting,
  but a truly independent test (different source images, ideally different generators) would
  give a more honest read on generalization before the actual competition test set.

## Quick load in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

train_dir = "/content/drive/MyDrive/hackathon_data/cleaned/sid_set/train"
val_dir = "/content/drive/MyDrive/hackathon_data/cleaned/sid_set/val"

import pandas as pd
train_labels = pd.read_csv(f"{train_dir}/labels.csv")
val_labels = pd.read_csv(f"{val_dir}/labels.csv")
print(train_labels['label_name'].value_counts())
print(val_labels['label_name'].value_counts())
```

Questions about the dataset → ask Sean.
