import os
import sys
import yaml
import torch
from ultralytics import YOLO

# Force UTF-8 for Windows console safety (prevents emoji crashes)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# --- CONFIGURATION ---
PROJECT_DIR = "runs/train"
RUN_NAME = "train"
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "last.pt")
HYPERPARAMS_FILE = "best_hyperparameters.yaml"


def train_manager():
    # 1. CHECK FOR RESUME (Crash Recovery)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n♻️ Found checkpoint at: {CHECKPOINT_PATH}")
        print("   Resuming training from where it crashed...")
        try:
            model = YOLO(CHECKPOINT_PATH)
            model.train(resume=True)
            print("\n🎉 Resumed Training Complete!")
            return  # Exit function after successful resume
        except Exception as e:
            print(f"⚠️ Resume failed: {e}")
            print("   Falling back to fresh training check...")

    # 2. FRESH TRAINING SETUP
    print("\n🚀 Starting FINAL Training (Fresh Hybrid Config)...")

    # Load the Optimized Params
    if not os.path.exists(HYPERPARAMS_FILE):
        print(f"❌ Error: Could not find '{HYPERPARAMS_FILE}' in project root.")
        return

    print(f"   Loading tuned parameters from: {HYPERPARAMS_FILE}")
    with open(HYPERPARAMS_FILE, "r") as f:
        best_hyp = yaml.safe_load(f)

    # Hardware Check
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   GPU Detected: {gpu_name} (Cache Cleared)")
    else:
        print("⚠️ Warning: No GPU detected.")

    # Initialize Model
    model = YOLO("yolov8m.pt")

    print("   Starting training... (This will take time)")

    try:
        model.train(
            data="data/data.yaml",
            # --- Training Duration ---
            epochs=200,
            patience=20,
            # --- 4GB GPU Safety Lock ---
            imgsz=640,
            batch=8,  # 8 is safe for 4GB VRAM.
            workers=0,  # 0 prevents Windows crashes.
            device=0,
            cache=False,  # Save RAM
            amp=True,  # Mixed Precision
            # --- Advanced Settings (Manual Overrides) ---
            close_mosaic=10,  # Sharper finish
            label_smoothing=0.05,  # Generalization
            copy_paste=0.1,  # Augmentation
            dropout=0.0,  # Let weight decay handle it
            # --- Storage ---
            project=PROJECT_DIR,
            name=RUN_NAME,
            save=True,
            save_period=10,
            plots=True,
            exist_ok=True,  # Allow overwriting if starting fresh
            # --- Tuned Hyperparameters (Priority) ---
            **best_hyp,
        )
        print("\n🎉 Fresh Training Complete!")

    except Exception as e:
        print(f"\n❌ Crash: {e}")
        if "CUDA out of memory" in str(e):
            print("\n💡 TIP: Your GPU ran out of memory.")
            print("   Edit this script: change 'batch=8' to 'batch=4'.")


if __name__ == "__main__":
    train_manager()
