# Manuscript Alignment in the Image Space

This project aligns two images of the same handwritten text line. The source and
target may differ in spacing, scale, writing style, and local geometry. Instead of
recognizing the sentence and returning text, the system predicts how each part of
the source image should move and produces a new, spatially aligned image.

The selected checkpoint is included at
`models/manuscript-registration-best.pt`, and the repository contains the
training, evaluation, inference, visualization, and Gradio demo code needed to
reproduce the project.

## Project overview

The project began with a word-based pipeline inspired by Madi et al. [1]:

1. YOLO detected word regions.
2. A Siamese network compared word crops.
3. Smith-Waterman aligned the resulting word sequences.

That approach was useful for finding corresponding words, but it did not satisfy
the main output requirement: it did not transform the source image. I therefore
reframed the task as dense image registration.

The final system directly returns:

- the affine-prealigned source;
- the final aligned source image;
- a target/aligned color overlay;
- the predicted dense displacement field; and
- diagnostic information about displacement and matching confidence.

## Final registration pipeline

Given a source image `Is` and target image `It`, the network predicts a backward
displacement field `u(x)` in pixel units. A differentiable spatial transformer
then computes:

```text
Ialigned(x) = Is(x + u(x))
```

The processing stages are:

```text
source + target
      |
      v
shared three-level feature encoder
      |
      v
patch correlation and coarse horizontal correspondence
      |
      v
skip-connected residual flow decoder
      |
      v
dense horizontal and vertical displacement field
      |
      v
spatial transformer -> aligned source
```

The shared encoder gives both images comparable feature representations. Patch
correlation estimates the large horizontal correspondence between text regions,
while the decoder refines vertical motion and local elastic differences. Before
the neural stage, an ink-bounding-box affine alignment removes large translation
and scale differences. The affine and learned flows are composed and applied in
one final warp to avoid unnecessary interpolation blur.

The model has 1,117,219 trainable parameters. It is deliberately compact enough
for a course project while still modeling dense, non-rigid motion.

## Try the trained model

### Gradio website

Create the environment and install the registration dependencies:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-registration.txt
```

Then launch the interface:

```powershell
.venv\Scripts\python.exe registration_web_app.py
```

Open the local URL printed in the terminal, upload one source line and one target
line, and select **Align source to target**. A CUDA-capable GPU is faster, but the
application also runs on CPU.

### Command-line inference

```powershell
.venv\Scripts\python.exe align_images.py `
  source_line.png target_line.png models/manuscript-registration-best.pt
```

The command writes the normalized inputs, aligned image, color overlay, and flow
visualization to `alignment_output`.

### Input image sizes

Uploaded images do **not** have to be `96 x 512` pixels. That is the training
block size, not an upload requirement. The inference pipeline:

- converts both inputs to normalized grayscale;
- preserves aspect ratio while normalizing the line height;
- places both lines on a shared white canvas;
- applies global affine prealignment; and
- processes long lines in overlapping 512-pixel blocks.

The block predictions are Hann-blended before the full image is warped. This
allows ordinary full-line images to be used without manually resizing them.

For the most reliable result, both images should contain one reasonably cropped
text line with the same words in the same order.

## Data and generalization

### IAM handwriting

`manuscript_registration/data.py` reads IAM form images and XML annotations
directly. The split is performed by writer identity, so a writer cannot appear in
both training and evaluation data.

Using seed 17:

| Split | Lines | Writers | Tokens unseen in training |
|---|---:|---:|---:|
| Train | 8,101 | 459 | - |
| Validation | 1,460 | 98 | 13.45% |
| Test | 1,783 | 100 | 13.54% |

Each training line is transformed on the fly using known affine and smooth
elastic motion. Because the transformation is known, the exact ground-truth flow
is available for supervision.

### Held-out words and font

The synthetic vocabulary is divided into train, validation, and test word
identities before rendering. One handwriting-like font is also excluded from
training. This creates a controlled evaluation in which both the words and visual
style can be unseen.

### Genuine cross-writer pairs

The project also evaluates real IAM lines with identical transcriptions written
by different test writers. Matching XML word centers serve as landmarks, allowing
geometric improvement to be measured without constructing a synthetic target.

## Training details

The model was trained on an NVIDIA RTX 4070 SUPER using:

- image blocks of `96 x 512`;
- batch size 32;
- AdamW with initial learning rate `2e-4`;
- weight decay `1e-4`;
- cosine-annealing learning-rate scheduling;
- bfloat16 mixed precision;
- gradient clipping at 5.0; and
- random seed 17.

Training used two stages:

1. **Main training:** 30 epochs on IAM lines and 4,000 cross-font synthetic
   pairs.
2. **Identity fine-tuning:** 8 additional epochs at learning rate `5e-5`, with
   20% exact identity pairs.

The second stage reduced unnecessary movement when source and target were already
aligned. The complete selected model therefore received 38 epochs of training.

Reproduce the main stage:

```powershell
.venv\Scripts\python.exe train_registration.py `
  --output-dir registration_runs/final_combined `
  --epochs 30 --batch-size 32 `
  --height 96 --width 512 `
  --base-channels 32 --max-residual-pixels 48 `
  --synthetic-samples 4000 --num-workers 4
```

Reproduce the fine-tuning stage:

```powershell
.venv\Scripts\python.exe train_registration.py `
  --output-dir registration_runs/identity_finetune `
  --epochs 8 --batch-size 32 --learning-rate 0.00005 `
  --height 96 --width 512 `
  --base-channels 32 --max-residual-pixels 48 `
  --synthetic-samples 4000 --identity-probability 0.20 `
  --num-workers 4 `
  --init-checkpoint registration_runs/final_combined/best.pt
```

Each run saves the best and final checkpoints, learning history, metrics, and an
exact split manifest.

## Evaluation and results

Endpoint error (EPE) is the primary metric. It measures the average Euclidean
distance between predicted and ground-truth flow vectors. The identity baseline
leaves the source unchanged.

| Evaluation | No alignment | Final model | Change |
|---|---:|---:|---:|
| IAM flow EPE | 26.54 px | **5.47 px** | 79.4% lower |
| IAM image MAE | 0.1187 | **0.0913** | 23.1% lower |
| IAM image SSIM | 0.5178 | **0.6832** | +0.1654 |
| Held-out words/font EPE | 26.23 px | **6.56 px** | 75.0% lower |
| Real cross-writer landmark error | 26.50 px | **10.34 px** | 61.0% lower |

On exact source-equals-target pairs, the final model predicts only 0.32 pixels of
mean motion. This identity-stability test matters because a useful registration
system should improve misaligned pairs without unnecessarily distorting inputs
that are already correct.

Run the evaluations:

```powershell
.venv\Scripts\python.exe evaluate_registration.py `
  models/manuscript-registration-best.pt --batch-size 32

.venv\Scripts\python.exe evaluate_cross_font.py `
  models/manuscript-registration-best.pt --samples 1000 --batch-size 32

.venv\Scripts\python.exe evaluate_real_pairs.py `
  models/manuscript-registration-best.pt

.venv\Scripts\python.exe evaluate_identity.py `
  models/manuscript-registration-best.pt
```

Generate qualitative examples:

```powershell
.venv\Scripts\python.exe visualize_registration.py `
  models/manuscript-registration-best.pt --count 8
```

Saved metrics and examples are available under:

- `registration_runs/identity_finetune`;
- `alignment_output/identity_finetune_examples`; and
- `alignment_output/real_pairs_identity_best`.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The 12 tests cover flow conventions, warping, flow resizing and composition,
model outputs and gradients, tiled inference, affine prealignment, IAM XML
extraction, writer-disjoint splits, identity examples, cross-font supervision,
and real-pair landmarks.

## Repository guide

| Path | Purpose |
|---|---|
| `manuscript_registration/model.py` | Shared encoder, patch correlation, decoder, and spatial transformer |
| `manuscript_registration/data.py` | IAM loading, writer splits, and synthetic training pairs |
| `manuscript_registration/inference.py` | Normalization, affine prealignment, tiling, flow composition, and visualizations |
| `train_registration.py` | Main training and fine-tuning entry point |
| `evaluate_registration.py` | Writer-disjoint IAM evaluation |
| `evaluate_cross_font.py` | Held-out vocabulary and font evaluation |
| `evaluate_real_pairs.py` | Genuine cross-writer landmark evaluation |
| `evaluate_identity.py` | Source-equals-target stability evaluation |
| `registration_web_app.py` | Gradio demonstration interface |
| `models/manuscript-registration-best.pt` | Selected trained checkpoint |
| `tests/test_registration.py` | Registration test suite |

## Limitations

- Both images are expected to contain the same text in the same order.
- Missing or additional words are not explicitly masked.
- Large rotation, strong shear, severe cropping, and very low-resolution strokes
  remain difficult.
- Different handwriting styles cannot overlap perfectly at the pixel level even
  when their word positions are geometrically correct.
- Very long lines may be downscaled when they exceed the configured maximum
  processing width.

These limitations are why flow and word-landmark error are more informative than
pixel similarity for cross-style evaluation.

## Reference

[1] B. Madi, A. Droby, and J. El-Sana, "Textline alignment on the image domain,"
*International Journal on Document Analysis and Recognition (IJDAR)*, vol. 25,
no. 4, pp. 415-427, 2022, doi: 10.1007/s10032-022-00408-5.
