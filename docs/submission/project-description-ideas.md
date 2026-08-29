Per the track 5 brief, the Written Project Description (via Devpost) must include:

How your solution addresses the problem statement
Development tools used (e.g. VSCode, Colab, Jupyter)
Models or APIs used
Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
Datasets and assets used



# Written Project Description Ideas

- Why we chose frozen CLIP over frequency-artifact detection
- Why we apply a matched JPEG re-encoding pass
- Why we chose RINE over UnivFD
- Assumption: bilinear interpolation is used for both the downsampling and upsampling steps of the resize round trip
- Why downsample-and-restore counts as one compound transform
- Why resize outputs are stored losslessly
- Why the model combines global context with pre-resize local crops
- Why interpolation artifacts are treated as an authentic false-positive risk
- Why matched JPEG normalization is offline rather than an inference step


# 1. How the solution addresses the problem statement
## How Our Solution Addresses the Problem Statement (excerpt)

A core early design decision was choosing frozen CLIP-ViT features as our primary detection backbone over a frequency-artifact-based approach. Frequency-domain detectors are effective on clean, unmodified images, but the signal they rely on — high-frequency spectral artifacts from the generation process — is precisely what transforms like JPEG compression, blurring, and resizing destroy. In our own testing, a frequency-based detector's fake-detection accuracy dropped from **[X]%** on clean images to **[Y]%** under JPEG compression at quality 30, while accuracy on real images stayed high — meaning the model was systematically collapsing toward predicting "real" under degradation, which is the opposite of what a redistribution-robust detector needs to do. By contrast, our frozen-CLIP baseline retained **[Z]%** fake-detection accuracy under the same transform. Since this track weights clean and robust performance equally, we treated robustness under transformation as a first-class design constraint rather than an afterthought, and committed to CLIP as our single production backbone accordingly — using frequency analysis only as a narrow, one-directional fast-track that can flag obvious clean synthetic images but is never allowed to independently certify an image as real.

(swap in x/y/z when done)


# Devpost Written Project Description — Draft

---

## How Our Solution Addresses the Problem Statement

Online platforms need to detect AI-generated images not just on clean uploads, but after the
compression, resizing, and casual editing that real content goes through as it's redistributed.
We built a binary detector — `authentic` vs. `ai_generated` — designed around robustness to that
redistribution pipeline as a first-class constraint, not an afterthought.

**Why frozen CLIP over frequency-artifact detection.** Our first architectural decision was
choosing a frozen CLIP-ViT backbone over a frequency/spectral-artifact detector as the primary
signal. Frequency-based detectors look for spectral fingerprints left behind by GAN/diffusion
upsampling — but JPEG compression, blurring, and resizing (exactly the transforms this track
requires us to survive) specifically destroy high-frequency content. In our own testing, a
frequency-based detector's accuracy on AI-generated images dropped from **[X]%** clean to **[Y]%**
under JPEG quality 30, while accuracy on real images stayed high — meaning the model was
systematically collapsing toward predicting "real" as soon as an image degraded, which is the
opposite of what a redistribution-robust detector needs to do. Our frozen-CLIP baseline retained
**[Z]%** accuracy on AI-generated images under the same transform. Since the track scores clean
and robust performance equally (50/50), we treated degradation robustness as a first-class design
constraint and built the primary pipeline around CLIP accordingly.

**One-directional shortcuts.** Our architecture allows two low-cost paths to shortcut early —
a C2PA provenance check and a frequency-based fast-track — but both are strictly one-directional:
they can only shortcut *toward* an `ai_generated` verdict, never toward `authentic`. Absence of a
C2PA manifest, or a spectral read that looks "clean," is not evidence of authenticity, since both
signals are exactly what routine redistribution (re-encoding, resizing, platform stripping)
erases. This closes a specific failure mode we identified early: a degraded fake could otherwise
silently pass through as "real" simply because the evidence of its synthesis had been destroyed by
the very transforms we're required to survive.

**Catching a shortcut-learning risk in our own data.** During data preparation we found that real
and synthetic images in our training set had different average file sizes at identical resolution
(**34.2KB vs. 26.8KB**) — real photos compress less efficiently than diffusion output. Left
uncorrected, the model could learn to key off compression artifacts as a proxy for the real/fake
label rather than genuine visual evidence, which typically shows up as strong in-dataset accuracy
but poor generalization and a specific vulnerability on the JPEG robustness test. We corrected this
by re-compressing real images down to match the synthetic size distribution before training.

## Development Tools Used

- Google Colab (primary training runtime, GPU-backed)
- VS Code, via the official Colab extension, for development against the remote runtime
- Google Drive for durable artifact/checkpoint storage

## Models or APIs Used

- **Frozen CLIP ViT-L/14** (vision tower only, ~304M params) — primary inference-time backbone,
  linear/MLP head trained on top
- **ConvNeXt-Tiny** — trained once, offline only, solely to produce a comparison baseline
  justifying the CLIP backbone choice; never loaded at inference time
- **C2PA reader/verifier** (`c2pa-python`) — cryptographic provenance manifest check, metadata
  parsing only, no model inference
- All models confirmed under the **<2B parameter** constraint; only one backbone (CLIP) is loaded
  simultaneously at inference time

## Libraries and Frameworks Used

- PyTorch
- Hugging Face Transformers (CLIP loading)
- scikit-learn (calibration, metrics)
- pandas (logging, evaluation tables)
- OpenCV / scikit-image (auxiliary texture-detail feature extraction)

## Datasets and Assets Used

- **[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)** — real photos and fully synthetic
  images (FLUX, Kandinsky, SDXL, AbsoluteReality, and others), streamed via Hugging Face rather
  than bulk-downloaded. We sampled 10,000 real and 10,000 synthetic images, removed 2 exact
  duplicates, and excluded SID_Set's third ("tampered") class entirely, since it falls outside the
  competition's real-vs-synthetic scope.
- WildFake was investigated as a second source for generator diversity but access was blocked by
  ModelScope's registration process; a candidate subset was identified for future use if that gets
  resolved.

---

## Notes for review (remove before submission)

- **[X]**, **[Y]**, **[Z]** — clean/JPEG-30/CLIP-under-JPEG-30 accuracy figures once we have them
  from the frequency-vs-CLIP offline comparison (Table 1 in `design.md`).
- Consider whether to mention PRNU/color/optics/texture auxiliary features at all in the writeup —
  currently framed here as *not* core to keep claims aligned with what's actually shipped; revisit
  if any of them survive the retention ablation and end up in the final fused model.
- Self-training loop intentionally omitted from this draft since it's designed but not yet run —
  add a short paragraph if it gets executed before submission.
- Robustness Evaluation Summary and Error Analysis Note are separate required deliverables, not
  part of this written description — draft those once Table 1 (clean vs. transformed) numbers
  exist.
