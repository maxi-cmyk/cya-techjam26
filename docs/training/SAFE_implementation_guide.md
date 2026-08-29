# SAFE — Implementation Guide for Training

This is for whoever builds the training loop. SAFE is **not** something applied to the dataset
files themselves — it's a set of transforms applied live, every epoch, inside your training
pipeline (via `torchvision.transforms` or equivalent). The dataset handoff (`cleaned/sid_set/`)
is deliberately untouched by SAFE — you apply this on top of it during training.

Paper: Li et al., "Improving Synthetic Image Detection Towards Generalization: An Image
Transformation Perspective" (SAFE), KDD '25.

## Why SAFE exists (one paragraph)

Most detectors overfit to whichever generator they were trained on and fail to generalize to
new generators. SAFE's core finding: this isn't a model-architecture problem, it's a
**training-pipeline bias** problem — specifically, how images get resized and how little
augmentation variety they see. Fix those two things with simple, cheap operations, and a
lightweight classifier generalizes surprisingly well.

## Two changes to make, both applied only in the training data loader

### 1. Replace resize with crop

**Problem:** standard practice resizes images to a fixed size using bilinear downsampling.
This smooths out the subtle local pixel correlations that up-sampling/convolution operations
leave behind during image generation — exactly the signal the detector needs to catch.

**Fix:**
- **Training:** use `RandomCrop`, not `Resize`
- **Inference/validation:** use `CenterCrop`, not `Resize`

```python
from torchvision import transforms

# Training transform
train_transform = transforms.Compose([
    transforms.RandomCrop(224),   # or whatever input size the classifier head expects
    transforms.RandomHorizontalFlip(),
    # ColorJitter, RandomRotation, RandomMask go here — see below
])

# Validation/inference transform
val_transform = transforms.Compose([
    transforms.CenterCrop(224),
])
```

Note: our dataset images are already saved at 336×336. Cropping to a smaller size like 224
(CLIP ViT-B/32's expected input) or 336 (ViT-L/14's) is normal — crop, don't resize, whatever
the final input size ends up being.

### 2. Add three augmentations beyond the usual HorizontalFlip

The paper found that HorizontalFlip alone isn't enough augmentation diversity to generalize
across generator architectures. Add these three, applied together, during training only:

**ColorJitter**
```python
transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
```
Jitters brightness/contrast/saturation by a random factor sampled from `[max(0, 1-α), 1+α]`.
The paper doesn't lock α to one exact value in the main text — 0.5 is a reasonable middle-of-range
starting point; treat it as tunable.

**RandomRotation**
```python
transforms.RandomRotation(degrees=180, fill=0)
```
Rotates by a random angle in `[-180°, +180°]`, filling the exposed corners with zero-value
(black) pixels — that's what `fill=0` does. The paper uses the full ±180° range.

**RandomMask** — this one has no direct `torchvision` equivalent, needs a small custom
transform:
```python
import numpy as np
import random

class RandomMask:
    def __init__(self, patch_size=16, max_mask_ratio=0.75, p=0.5):
        self.patch_size = patch_size
        self.max_mask_ratio = max_mask_ratio
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        r = random.uniform(0, self.max_mask_ratio)
        d = self.patch_size
        n_patches = int((h * w * r) / (d * d))

        for _ in range(n_patches):
            y = random.randint(0, max(0, h - d))
            x = random.randint(0, max(0, w - d))
            img_np[y:y+d, x:x+d] = 0

        from PIL import Image
        return Image.fromarray(img_np)
```
Applies with probability `p` (paper doesn't fix this — 0.5 is reasonable to start with).
Randomly zeroes out `d×d` patches until up to `max_mask_ratio` (paper found even 75% masking
still trains fine) of the image is covered, with no overlapping patches. The point: forces the
detector to rely on many small local regions instead of one obvious global cue.

### Putting it together

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomRotation(degrees=180, fill=0),
    RandomMask(patch_size=16, max_mask_ratio=0.75, p=0.5),
    transforms.ToTensor(),
    # + CLIP's normalization stats, whatever preprocessing CLIP expects
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # same normalization, no augmentation — val/test should be deterministic
])
```

**Important:** augmentations (ColorJitter, RandomRotation, RandomMask) go in `train_transform`
only. `val_transform` should stay deterministic (crop only) so evaluation results are
reproducible and comparable across runs.

## What SAFE does NOT cover (optional, only if there's time)

The paper also proposes a frequency-domain feature (Discrete Wavelet Transform, extracting the
high-frequency HH sub-band) as an additional input signal alongside the raw image. This is a
separate, more involved addition — not required for a first working version, and probably not
worth the implementation time relative to a CLIP-based classifier head, which already has
strong pretrained visual features. Skip this unless there's time to spare after a first
working model.

## Quick checklist before training

- [ ] Crop, not resize, in both train and val transforms
- [ ] ColorJitter + RandomRotation + RandomMask + HorizontalFlip applied together, train only
- [ ] val_transform has no augmentation, only CenterCrop
- [ ] Confirm final input size matches what the CLIP encoder expects (224 for ViT-B/32, 336 for ViT-L/14)
