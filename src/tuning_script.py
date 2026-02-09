import os
import sys
import torch  # Import torch to clear cache

# 1. FORCE UTF-8 (Fixes Windows Emoji Crash)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from ultralytics import YOLO

try:
    from ray import tune
except ImportError:
    print("[ERROR] Ray Tune is missing. Run: pip install 'ray[tune]'")
    exit()


def tune_model():
    print("🚀 Starting Robust Ray Tune for Glove Detection...")
    print("   Strategy: ASHA Scheduler | AdamW | Safe Memory Config")

    # 2. CLEAR GPU CACHE BEFORE STARTING
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")

    # 3. RESOLVE ABSOLUTE PATHS
    # This prevents Ray from getting lost in temp directories
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data", "data.yaml")
    run_dir = os.path.join(cwd, "runs", "tune")  # Force logs to stay in project folder

    if not os.path.exists(data_path):
        print(f"[ERROR] Cannot find: {data_path}")
        return

    model = YOLO("yolov8s.pt")

    search_space = {
        # Optimizer
        "lr0": tune.uniform(1e-4, 1e-2),
        "lrf": tune.uniform(0.01, 0.1),
        "momentum": tune.uniform(0.9, 0.98),
        "weight_decay": tune.uniform(1e-4, 1e-3),
        # Loss Gains
        "box": tune.uniform(7.5, 12.0),
        "cls": tune.uniform(0.3, 1.0),
        "dfl": tune.uniform(1.0, 2.0),
        # Augmentation
        "degrees": tune.uniform(-45.0, 45.0),
        "translate": tune.uniform(0.1, 0.2),
        "scale": tune.uniform(0.4, 0.9),
        "hsv_h": tune.uniform(0.0, 0.02),
        "hsv_s": tune.uniform(0.5, 0.9),
        "hsv_v": tune.uniform(0.3, 0.6),
        "mosaic": tune.uniform(0.8, 1.0),
        "mixup": tune.uniform(0.0, 0.15),
        "fliplr": tune.uniform(0.5, 0.5),
        "flipud": tune.uniform(0.0, 0.1),
    }

    try:
        model.tune(
            data=data_path,
            epochs=30,
            iterations=50,
            use_ray=True,
            space=search_space,
            # --- CRITICAL STABILITY SETTINGS ---
            gpu_per_trial=1,
            # 1. LOWER BATCH SIZE: Prevents "GPU memory overflow"
            batch=8,
            # 2. DISABLE MULTIPROCESSING: Fixes "FileNotFound" & Windows crashes
            workers=4,
            # Fixed Settings
            imgsz=640,
            plots=True,
            save=True,
            val=True,
            optimizer="AdamW",
            # Project Paths
            project=run_dir,
            name="tune",
        )
        print("\n[SUCCESS] Tuning Complete! Check 'runs/tune/'")

    except Exception as e:
        print(f"\n[EXCEPTION] Run Failed: {e}")


if __name__ == "__main__":
    tune_model()
