Here is a comprehensive, production-ready `README.md` tailored specifically to your "Glove/Ungloved" detection project. It incorporates your custom data cleaning, ensemble validation, and human-in-the-loop labeling tools into a cohesive workflow.

---

# 🧤 Gloved vs. Ungloved Hand Detection (YOLOv8)

This project implements an end-to-end computer vision pipeline for detecting **Gloved** vs. **Ungloved** hands. It moves beyond simple model training by including a robust set of data engineering tools—automated sanitization, AI-assisted label correction (using CLIP and Pose estimation), and a custom Streamlit GUI for manual review.

While the final model achieves **high test accuracy**, the pipeline is designed to balance this precision with reasonable real-time inference speeds for production environments.

---

## ✨ Key Features & Workflow

This repository is structured around a "Quality-First" data pipeline:

1. **Data Sanitization (`auto_clean_yolo.py`)**: Automatically fixes coordinate errors (0-1 normalization) and quarantines corrupt label files.
2. **Human-in-the-Loop Review (`yolo_labeler.py`)**: A custom Streamlit app to visually inspect, fix, or junk images rapidly.
3. **Ensemble Validation (`ensemble_safe_validate_and_fix.py`)**: A powerful script that uses **YOLOv8-Pose** and **CLIP (ViT-L/14)** to cross-validate labels, auto-correcting high-confidence errors and flagging ambiguous ones.
4. **Dataset Splitting (`finalize_split.py`)**: Randomly splits the cleaned "Golden Dataset" into Train (80%), Val (15%), and Test (5%) sets.
5. **Training & Tuning**: Automated hyperparameter tuning for YOLOv8.

---

## 📂 Project Structure

```text
.
├── config.yaml                     # Global configuration (paths, classes, etc.)
├── models/
│   ├── best.pt                     # Your current best detection model
│   └── yolov8x-hand-ensemble.pt    # (Optional) Pose model for validation
├── src/
│   ├── auto_clean_yolo.py          # Script: Fix coords & quarantine corrupt files
│   ├── yolo_labeler.py             # App: Streamlit GUI for manual labeling
│   ├── ensemble_safe_validate.py   # Script: CLIP+Pose+YOLO auto-validation
│   ├── finalize_split.py           # Script: Generate Train/Val/Test splits
   └── tuning_script.py             # Script: Hyperparameter tuning (ray/optuna)
├── data/                           # Final output for training
│   ├── images/
│   └── labels/
└── requirements.txt                # Python dependencies
│
└── runs                            # contain run artifacts including the final trained model to try for reference

```

---

## 🛠️ Installation

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Install Dependencies

This project requires standard YOLO libraries plus `streamlit` for the GUI and `clip` for the ensemble validator.

```bash
# Core requirements
pip install ultralytics torch torchvision torchaudio opencv-python pyyaml tqdm

# For the Labeling GUI
pip install streamlit

# For Ensemble Validation (CLIP)
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git

```

*Note: For GPU acceleration (recommended), ensure you install the CUDA-enabled version of PyTorch.*

---

## 🚀 Pipeline Usage Guide

Follow these steps to process your raw data into a production-ready model.

### Step 1: Data Sanitization

Before viewing data, ensure all YOLO coordinates are valid (normalized 0-1) and remove orphan images.

```bash
python src/auto_clean_yolo.py

```

* **Output:** Clean files move to `interim/`, corrupt files move to `quarantine/`.

### Step 2: Human Visual Review (Streamlit)

Use the custom GUI to rapidly verify the sanitized images. This tool allows you to fix classes, delete blurry images, and save to a "Golden" directory.

```bash
streamlit run src/yolo_labeler.py

```

* **Controls:**
* `Save Changes`: Moves file to Golden dataset.
* `Junk Image`: Moves file to Quarantine.
* `Checkbox`: Uncheck to delete a specific bounding box.



### Step 3: AI-Assisted Auto-Correction

Run the ensemble validator to catch errors the human eye missed. This uses your current `best.pt` model, a Pose model, and CLIP to validate labels.

```bash
# Dry Run (Safe Mode - won't change files)
python src/ensemble_safe_validate_and_fix.py --dry-run

# Auto-Fix Mode (Writes corrections to data_corrected/)
python src/ensemble_safe_validate_and_fix.py --autofix --delete-low

```

* **Logic:**
* **Auto-Fix:** If `Confidence > 0.90` (Model + Pose + CLIP consensus), the label is auto-written.
* **Flag:** If `0.75 < Confidence < 0.90`, the image is moved to a review folder.
* **Delete:** If `Confidence < 0.20`, the image is considered empty/background.



### Step 4: Finalize Splits

Once your data is clean and validated, generate the final folder structure for YOLO training.

```bash
python src/finalize_split.py

```

* **Result:** Populates `data/train`, `data/val`, and `data/test` based on an 80/15/5 ratio.

### Step 5: Training & Tuning

Run the training script (assuming `combined_training.py` or similar).

```bash
yolo detect train data=config.yaml model=yolov8n.pt epochs=100 imgsz=640

```

---

## ⚙️ Configuration (`config.yaml`)

Control the pipeline paths using a central configuration file:

```yaml
paths:
  raw_images: "raw_data/images"
  raw_labels: "raw_data/labels"
  interim_images: "interim_data/images"   # After auto_clean
  interim_labels: "interim_data/labels"
  golden_images: "golden_data/images"     # After human review
  golden_labels: "golden_data/labels"
  quarantine: "quarantine"

classes:
  0: "ungloved"
  1: "gloved"

params:
  swap_classes: false  # Set true if 0 and 1 need to be inverted globally

```

---

## 📊 Performance & Inference Note

* **Accuracy:** The model currently achieves high precision/recall on the test set, largely due to the rigorous cleaning pipeline (CLIP validation + Human review).
* **Inference Speed:** Real-time performance is "reasonable" but depends heavily on the backbone used (Nano vs. Small vs. Medium).
* For **Webcam/Edge** devices: Recommend using `yolov8n.pt` or `yolov8s.pt`.
* For **Server/Batch**: `yolov8m.pt` or `yolov8l.pt` will provide better small-object detection at the cost of speed.



### Real-Time Inference Command

```bash
yolo detect predict model=models/best.pt source=0 show=True conf=0.5

```

## License
MIT License. See LICENSE for details.
