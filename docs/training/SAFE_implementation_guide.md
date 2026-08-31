# SAFE training ablation

SAFE is an optional, training-only ablation. It is implemented in
`src/cya_detector/transforms/safe.py`, configured under
`configs/colab.json["training_policy"]["safe"]`, and disabled in the retained
pipeline. It must never be applied to stored dataset files, `selection_val`,
`final_test`, robustness variants, or inference inputs.

The design is based on Li et al., ["Improving Synthetic Image Detection Towards
Generalization: An Image Transformation Perspective"](https://arxiv.org/abs/2408.06741)
(KDD 2025) and its [official implementation](https://github.com/Ouxiang-Li/SAFE).
The paper motivates crop-based preprocessing, ColorJitter and RandomRotation,
and patch-based random masking. The exact settings and deterministic behavior
below are repository choices.

## Relationship to the primary training policy

The retained controlled policy draws either a clean view or exactly one Task 3
transform. SAFE is a separate alternative, not an extra augmentation layer on
top of that sampler. `validate_training_policy()` requires exactly one of the
controlled and SAFE policies to be enabled and rejects SAFE outside
`phase="seed_train"`.

The current configuration keeps:

```text
training_policy.controlled.enabled = true
training_policy.safe.enabled = false
```

Enabling SAFE is therefore a new, separately versioned experiment. It cannot
change the frozen controlled-RINE baseline or authorize another `final_test`
read.

## Repository implementation

For each `seed_train` image, `apply_safe()` performs these operations in order:

1. symmetrically zero-pad images smaller than 336 px;
2. take a deterministic random 336 x 336 crop;
3. apply horizontal flip with probability 0.5;
4. sample brightness, contrast, and saturation factors from `[0.5, 1.5]` and
   apply them in that order;
5. rotate by a sampled angle in `[-180, 180]` using bilinear interpolation,
   without canvas expansion, and fill exposed pixels with black;
6. with probability 0.5, choose a target mask fraction from `[0, 0.75]`, shuffle
   the non-overlapping 16 x 16 grid cells, and mask as many whole cells as fit
   without exceeding the target area.

The input size is 336 because the project is pinned to
`openai/clip-vit-large-patch14-336`; the older 224 px ViT-B examples do not
describe this repository.

Randomness is reproducible. Crop, flip, jitter, rotation, and mask each receive
a named local seed derived from the project seed, epoch, sample ID, and operation
name with SHA-256. SAFE does not consume shared global random state. The result
records padding, crop bounds, sampled factors, rotation settings, mask boxes,
realized mask fraction, and all local seeds.

Validation and inference continue to use the normal deterministic 336 px model
preparation defined in `src/cya_detector/transforms/preprocessing.py`; they do
not use SAFE augmentation.

## Verification

Run:

```bash
make task3-test
```

The SAFE tests verify:

- mutual exclusion between controlled and SAFE training policies;
- rejection outside `seed_train`;
- deterministic results for the same sample, epoch, and project seed;
- correct symmetric padding and crop geometry;
- the fixed operation order and recorded settings;
- unique, non-overlapping grid-mask cells that never exceed the configured
  maximum fraction; and
- no mutation of global random state.

## Before any SAFE experiment

- Version the resolved configuration and output root separately from controlled
  RINE.
- Keep the CLIP backbone frozen unless a different experiment explicitly says
  otherwise.
- Evaluate with the unchanged clean and 14-cell independent robustness
  contract.
- Compare all seeds against controlled RINE using the locked 50/50 and per-class
  gates.
- Reject the candidate if it fails those gates; never tune against or reread
  `final_test`.
