# Manuscript Line Spatial Registration

This project trains a deep neural registration model that warps a source text-line
image, `Is`, into the coordinate system of a target image, `It`. The primary model
outputs both a dense displacement field and the newly aligned image, `Ialigned`.

The repository also retains the earlier YOLO + Siamese + Smith-Waterman pipeline as
a sequence-alignment baseline. That baseline identifies corresponding word crops;
it does **not** perform spatial image registration and is therefore not the primary
solution to the project task.

## Registration formulation

The model predicts a backward displacement field `u(x)` in pixel units. A
differentiable spatial transformer generates the output:

```text
Ialigned(x) = Is(x + u(x))
```

The architecture is content-independent and fully convolutional:

```text
Is -> shared feature pyramid -> source patch descriptors --\
                                                          correlation -> coarse flow
It -> shared feature pyramid -> target patch descriptors --/               |
                                                                             v
Is + It + warped features -> residual flow decoder -> dense u -> grid_sample -> Ialigned
```

Vertical feature-map columns act as overlapping image blocks. Global horizontal
attention finds coarse target-to-source patch correspondences, while the decoder
estimates vertical and local elastic corrections. A learnable correlation gain is
initialized to zero, so the network starts from a safe identity transformation.

The training objective combines:

- supervised endpoint error against known synthetic flow;
- an auxiliary coarse horizontal-flow loss;
- ink-weighted Charbonnier and SSIM losses for same-appearance pairs;
- flow smoothness;
- a monotonicity penalty that discourages folded or reordered text.

For cross-font pairs, raw pixel loss is disabled because different glyph styles
cannot be expected to match pixel-for-pixel. Their exact synthetic flow remains
fully supervised.

## Data and generalization

### IAM handwriting

`manuscript_registration/data.py` reads the existing `IAM_Data/forms` images and
line annotations directly from `IAM_Data/xml`; no separate line-image extraction
is required. Splits are made by IAM `writer-id`, so no writer appears in more than
one split.

With the current IAM copy and seed 17:

| Split | Lines | Writers | Word tokens unseen in train |
|---|---:|---:|---:|
| Train | 8,101 | 459 | - |
| Validation | 1,460 | 98 | 13.45% |
| Test | 1,783 | 100 | 13.54% |

Each IAM line is converted on the fly into a training pair using affine and smooth
elastic transformations with exact target-to-source ground truth.

### Cross-font synthetic pairs

The common-word vocabulary is split **before rendering** into 70% train, 15%
validation, and 15% test word identities. Identical text is rendered in two
handwriting-like fonts using shared semantic word cells, followed by a known
spatial warp. The last font in sorted order is excluded from training and used for
held-out font evaluation.

This explicitly tests the concern that input words and handwriting styles may not
have appeared during training.

## Environment

The tested workstation configuration is Windows, Python 3.12, PyTorch 2.11 with
CUDA 12.8, and an NVIDIA RTX 4070 SUPER (12 GB).

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-registration.txt
```

The broader `requirements.txt` also installs dependencies for the legacy YOLO
pipeline.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The test suite verifies flow/grid conventions, identity and translated warps,
flow resizing, model gradients and output shapes, XML line extraction,
writer-disjoint splitting, identity-pair generation, real-pair landmarks, and
cross-font supervision behavior. The current suite contains nine tests.

## Training

The final mixed-data configuration used on the RTX 4070 is:

```powershell
.venv\Scripts\python.exe train_registration.py `
  --output-dir registration_runs/final_combined `
  --epochs 30 `
  --batch-size 32 `
  --height 96 `
  --width 512 `
  --base-channels 32 `
  --max-residual-pixels 48 `
  --synthetic-samples 4000 `
  --num-workers 4
```

Training writes:

- `best.pt`: checkpoint with the lowest writer-disjoint validation EPE;
- `last.pt`: latest checkpoint;
- `history.csv`: all loss components and metrics per epoch;
- `split_manifest.json`: exact reproducible writer split and line metadata;
- identity-baseline metrics alongside every validation result.

To resume an interrupted run:

```powershell
.venv\Scripts\python.exe train_registration.py `
  --output-dir registration_runs/final_combined `
  --epochs 30 `
  --resume registration_runs/final_combined/last.pt
```

Use the same architectural and data arguments as the original run when resuming.

The selected model adds an eight-epoch low-learning-rate fine-tune initialized
from the mixed-data checkpoint. Twenty percent of its IAM samples have exact zero
flow, explicitly teaching the model not to warp an already aligned pair:

```powershell
.venv\Scripts\python.exe train_registration.py `
  --output-dir registration_runs/identity_finetune `
  --epochs 8 --batch-size 32 --learning-rate 0.00005 `
  --height 96 --width 512 --base-channels 32 `
  --max-residual-pixels 48 --synthetic-samples 4000 `
  --identity-probability 0.20 --num-workers 4 `
  --init-checkpoint registration_runs/final_combined/best.pt
```

## Evaluation

Writer-disjoint IAM test set:

```powershell
.venv\Scripts\python.exe evaluate_registration.py `
  models/manuscript-registration-best.pt `
  --batch-size 32
```

Held-out words and held-out font:

```powershell
.venv\Scripts\python.exe evaluate_cross_font.py `
  models/manuscript-registration-best.pt `
  --samples 1000 `
  --batch-size 32
```

The primary metric is endpoint error (EPE) of the predicted flow. Additional
metrics include 1/3/5-pixel accuracy, aligned-image MAE, SSIM, and ink-mask Dice.
For cross-font data, EPE is primary because pixel similarity between different
glyph styles is not semantically meaningful.

Real IAM lines with identical transcriptions from different test writers:

```powershell
.venv\Scripts\python.exe evaluate_real_pairs.py `
  models/manuscript-registration-best.pt
```

Exact source-equals-target behavior:

```powershell
.venv\Scripts\python.exe evaluate_identity.py `
  models/manuscript-registration-best.pt
```

### Final results and model selection

The selected checkpoint is committed as `models/manuscript-registration-best.pt`.
It slightly improves synthetic-warp IAM and cross-font EPE while reducing unwanted
motion on identical input images by 86.19% compared with the original mixed model.

| Test set and metric | Identity baseline | Final model |
|---|---:|---:|
| IAM EPE (pixels, lower is better) | 26.54 | **5.47** |
| IAM pixels within 3 px | - | **29.88%** |
| IAM pixels within 5 px | - | **55.12%** |
| IAM image MAE (lower is better) | 0.1187 | **0.0913** |
| IAM SSIM (higher is better) | 0.5178 | **0.6832** |
| IAM ink Dice | - | **0.4733** |
| Held-out cross-font EPE | 26.23 | **6.56** |
| Exact-pair mean predicted motion | 0.00 | **0.32** |
| Real cross-writer word-landmark error | 26.50 | **10.34** |

Cross-font EPE improved by 75.00% over identity. On ten genuine IAM line pairs
from different unseen writers, the word-landmark error improved by 60.98%.
Cross-font MAE, SSIM, and Dice
are retained in `cross_font_metrics.json` as diagnostics, but they compare
different glyph shapes and are not the primary evidence of geometric accuracy.

Render qualitative test examples:

```powershell
.venv\Scripts\python.exe visualize_registration.py `
  models/manuscript-registration-best.pt `
  --count 8
```

The completed run, metric JSON files, learning history, checkpoints, and rendered
examples are under `registration_runs/identity_finetune`,
`alignment_output/identity_finetune_examples`, and
`alignment_output/real_pairs_identity_best`.

## Inference

```powershell
.venv\Scripts\python.exe align_images.py `
  source_line.png target_line.png models/manuscript-registration-best.pt
```

The command writes normalized source and target images, `aligned.png`, a colored
target/aligned overlay, and a flow visualization to `alignment_output`.

Launch the registration UI:

```powershell
.venv\Scripts\python.exe registration_web_app.py
```

The UI returns the actual aligned source image, not only a word correspondence
plot. Lines wider than the 512-pixel training canvas are automatically divided
into overlapping 512-pixel blocks. Their predicted flows are Hann-blended before
the full source is warped, preventing long sentences from being evaluated far
outside the training width. Before local registration, an ink-bounding-box affine
step removes large global translation and scale differences. The affine and neural
backward fields are composed and applied to the original source in one sampling
operation, which avoids double-interpolation blur.

For interpretability, the UI displays the global affine-prealigned source and its
target overlay separately from the final affine-plus-dense result. This makes the
contribution of the learned local registration visible in each example.

## Legacy sequence-alignment baseline

The original pipeline remains available for comparison:

1. `prepare_yolo_dataset.py` creates word-detection patches.
2. `train_yolo.py` trains a one-class word detector.
3. `train_siamese_triplet.py` embeds word crops.
4. `english_alignment_web_app.py` applies Smith-Waterman to the word sequence.

It is useful as a baseline or future source of high-confidence word anchors, but
its output is a correspondence visualization rather than `Ialigned`.

## Reference

B. Madi, A. Droby, and J. El-Sana, "Textline alignment on the image domain,"
*International Journal on Document Analysis and Recognition*, 25, 415-427 (2022),
doi:10.1007/s10032-022-00408-5.
