# Model Architecture and Decisions

The retained detector is controlled RINE seed 42 over a frozen CLIP
ViT-L/14-336 vision backbone. It is the only learned model in the shipped
inference path. All auxiliary feature-fusion candidates were evaluated offline
and disabled after failing the locked selection rule.

## Selected Model

| Property | Frozen value |
|---|---|
| Model identifier | `openai/clip-vit-large-patch14-336` |
| Resolved revision | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| Input size | CLIP processor output at 336 x 336 |
| Intermediate layers | 6, 12, 18, 24 |
| Matching policy | Fixed-Q96 matched clean |
| Training seed | 42 |
| Temperature | `T=1` (no post-hoc scaling) |
| Evaluation threshold | `0.5` |
| Runtime auxiliaries | None |

For an RGB image, the frozen CLIP tower returns the CLS representation at each
selected layer. The RINE head normalizes each representation, applies a learned
softmax over four layer logits, forms their weighted sum, and sends that vector
to a single linear classifier:

```text
h6, h12, h18, h24
        |
per-layer normalization
        |
softmax(layer_logits)
        |
weighted sum
        |
linear classifier
        |
sigmoid(logit) = pred
```

CLIP parameters remain frozen during training and inference. The trainable head
contains four layer-importance logits plus the linear classifier weight and bias.
Checkpoint loading verifies the experiment stage, seed, matching policy, selected
layers, state shape, and pinned CLIP revision before serving predictions.

Temperature fitting was attempted after checkpoint selection on 165 perfectly
classified clean validation rows. The optimum ran to the search lower bound
(`T=0.0500038`), which would only sharpen already correct scores. It was rejected;
the shipped value is the unchanged raw sigmoid at `T=1`, so `pred` must not be
described as a separately calibrated probability guarantee.

## Candidate Outcomes

| Model or feature path | Evidence | Status |
|---|---:|---|
| Controlled RINE | 100.00% clean, 99.62% robustness, 99.81% locked mean across three seeds | Retained |
| Controlled RINE seed 42 | 99.85% locked development score | Packaged checkpoint |
| Frequency fusion | 52.15% locked mean versus 99.81% parent | Rejected |
| Lab correlation fusion | 98.95% locked mean and 1.82-point AI-class regression | Rejected |
| PRNU-v2 alone | 78.09% locked mean | Diagnostic only |
| RINE + PRNU-v2 | 33.43% locked mean; two seeds collapsed | Rejected |
| Global + texture patches | Passed clean gate; 93.13% robustness versus 99.80% parent | Rejected |
| Chromatic aberration / radial distortion | No eligible calibrated lens, focal, or edge-rich support | Deferred |
| ConvNeXt-Tiny | Optional comparison model | Not shipped |

The frequency early exit is disabled. Texture crops, PRNU summaries, RGB/Lab
correlations, chromatic-aberration estimates, and radial-distortion estimates are
not computed by the production predictor. Their code and results document the
experiments; they do not describe live inference stages.

## Dataset Contract

Only unambiguous binary examples are eligible:

| Label | Included | Excluded |
|---|---|---|
| `authentic` | Genuinely captured source images | AI-enhanced, composited, face-swapped, or otherwise ambiguous images |
| `ai_generated` | Fully synthetic images with no authentic source | Image-to-image edits, inpainting of authentic images, partial-AI, or mixed-origin content |

Every source receives an immutable `source_original`, a canonical fixed-Q96
`matched_clean` derivative, and optional development robustness derivatives.
Derivatives inherit the source split. Matching settings are independent of the
label, preventing file-encoding history from becoming a shortcut.

Robustness variants are independently generated from `matched_clean`. The
protocol covers JPEG compression, Gaussian blur, resize-and-restore at 0.5x and
0.25x, Gaussian noise, color jitter, and center crop. No transformed output is
fed through another transformation.

## Evaluation Contract

Development architecture selection is a 50/50 score:

`locked score = 0.50 x clean accuracy + 0.50 x robustness mean`

The robustness mean is the unweighted mean of the 14 fixed transform-parameter
cells. Overall and per-class accuracy are retained so a candidate cannot hide
class collapse behind its aggregate score.

The sealed final test was read exactly once after architecture, weights,
temperature, threshold, and feature flags were frozen. It contained 141 direct
fixed-Q96 matched-clean rows, not transformed variants. The result was 99.29%
(140/141): 69/69 AI-generated and 71/72 authentic, with ECE 0.0189. The
development robustness matrix remains selection evidence and must stay separate
from this final-test result.

## Scope and Limitations

- The model has not been validated for AI-edited, mixed-origin, inpainted,
  face-swapped, or composited images.
- C2PA is a separate one-way provenance shortcut, not a learned model feature and
  never proof of authenticity.
- Target-hardware latency, peak memory, and disk/cache requirements still need to
  be measured.
- A missing checkpoint currently causes a clearly warned `0.5` stub fallback;
  stub output is not valid detector evidence.
