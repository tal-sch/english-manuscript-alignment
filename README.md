# English Manuscript Alignment Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to align two different copies of handwritten English manuscript lines. It uses a custom **YOLOv8** model to detect and slice words, a **Siamese Network (with Triplet Loss)** to compare word similarity regardless of handwriting style, and the **Smith-Waterman Algorithm** to find the optimal local alignment between the two lines.

## How to Run the Pipeline (Step-by-Step)

If you are starting from scratch, here is the exact order in which the scripts should be run:

### 1. Data Preparation & Augmentation
First, you need to prepare the datasets. This involves extracting real handwriting from the IAM Database and generating massive amounts of synthetic cursive handwriting for robustness.

1. **`prepare_yolo_dataset.py`**: Parses the raw IAM Handwriting Database. It reads the XML metadata, crops out all 115,000+ real English words, and automatically generates YOLO-formatted label `.txt` files.
2. **`download_fonts.py`**: Downloads 5 different cursive Google Fonts (e.g., Caveat, Dancing Script).
3. **`generate_synthetic_data.py`**: Uses the downloaded fonts and a list of the 1,000 most common English words to generate 25,000 highly distorted, synthetic, single-word image crops.
4. **`generate_synthetic_pages.py`** (The Virtual Scribe): Generates 500 full-page, fully synthetic manuscript pages, scattering the words across the page and automatically generating YOLO bounding box labels.

### 2. Model Training
Once the data is generated, you train the two independent neural networks.

5. **`train_yolo.py`**: Trains the YOLOv8 object detection model on both the real IAM manuscript patches and the 500 synthetic pages. This creates our ultra-robust word slicer.
6. **`train_siamese_triplet.py`**: Trains the Siamese Feature Extractor for 50 Epochs. It uses a `TripletMarginLoss` to look at an Anchor word, a Positive (same word, different font/writer), and a Negative (different word), learning to mathematically cluster identical words while pushing different words apart.

### 3. Inference & Web App
Once both models are trained and their `.pt` weights are saved, you can run the final application.

7. **`english_alignment_web_app.py`**: This is the main application! Run this to launch the Gradio Web UI. It automatically loads the YOLO weights and the Siamese weights. You can upload two images of manuscript lines (Line A and Line B), tweak the Smith-Waterman hyperparameter penalties, and instantly see the visual alignment map.

### 4. Helper Scripts
- **`create_test_lines.py`**: A quick utility script that automatically crops out two consecutive lines from an IAM test form so you have something to easily drag-and-drop into the Web App for testing.
