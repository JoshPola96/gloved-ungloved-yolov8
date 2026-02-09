# src/inference_script.py

import os
import glob
from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
# Based on your split script, your test images are likely here:
TEST_DATA_PATH = "data/test/images"
CONF_THRESHOLD = 0.35  # Confidence threshold
IOU_THRESHOLD = 0.45  # NMS threshold


def get_latest_train_run_path(base_path="runs/train"):
    """
    Finds and returns the path to the latest YOLOv8 training run directory.
    """
    # Check if base path exists
    if not os.path.exists(base_path):
        return None

    train_runs = glob.glob(os.path.join(base_path, "*"))  # Match all subfolders

    # Filter for folders that look like 'glove_detection_final_hybrid' or 'train'
    valid_runs = [r for r in train_runs if os.path.isdir(r)]

    if not valid_runs:
        return None

    # Sort by modification time (most recent first) is usually safer/easier
    valid_runs.sort(key=os.path.getmtime, reverse=True)
    latest_run_path = valid_runs[0]

    print(f"📂 Detected latest run: {latest_run_path}")
    return latest_run_path


def run_inference(source_folder):
    """
    Performs object detection inference on a folder of images.
    """
    # 1. Find Model
    latest_train_path = get_latest_train_run_path()
    if not latest_train_path:
        print("❌ Error: Could not find any training runs in 'runs/train'.")
        return

    model_path = os.path.join(latest_train_path, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Error: 'best.pt' not found in {latest_train_path}/weights")
        return

    # 2. Load Model
    print(f"🚀 Loading model from: {model_path}")
    model = YOLO(model_path)

    # 3. Validate Source
    if not os.path.exists(source_folder):
        print(f"❌ Error: Source folder '{source_folder}' does not exist.")
        print("   Did you mean 'data/test/images'?")
        return

    # 4. Hardware Check
    device = 0 if torch.cuda.is_available() else "cpu"
    print(
        f"   Inference Device: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}"
    )

    # 5. Run Inference
    print("   Running predictions...")
    results = model.predict(
        source=source_folder,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        augment=True,  # TTA enabled (Slower but more accurate for testing)
        device=device,  # Force GPU
        project="runs/predict",
        name="inference",
        exist_ok=True,  # Overwrite old results so you don't get inference_results2, 3, etc.
    )

    print("\n✅ Inference complete!")
    print(f"   Processed: {len(results)} images")
    print("   View results: runs/predict/inference_results")


if __name__ == "__main__":
    run_inference(TEST_DATA_PATH)
