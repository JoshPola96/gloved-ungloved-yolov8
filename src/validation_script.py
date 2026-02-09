# validation_script.py

import os
import glob
import re
from ultralytics import YOLO


def get_latest_train_run_path(base_path="runs/train"):
    """
    Finds and returns the path to the latest YOLOv8 training run directory.
    """
    train_runs = glob.glob(os.path.join(base_path, "train*"))

    if not train_runs:
        return None

    def sort_key(path):
        match = re.search(r"train(\d*)$", path)
        if match:
            return int(match.group(1) or 0)
        return float("inf")

    train_runs.sort(key=sort_key, reverse=True)
    latest_run_path = train_runs[0]

    print(f"Detected the latest training run at: {latest_run_path}")
    return latest_run_path


def validate_model():
    """
    Validates the trained YOLOv8 model on validation dataset.
    """
    latest_train_path = get_latest_train_run_path()
    if not latest_train_path:
        print("Error: Could not find any training runs.")
        return

    model_path = os.path.join(latest_train_path, "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at '{model_path}'.")
        return

    model = YOLO(model_path)
    print(f"Loading and validating model from: {model_path}")

    results = model.val(
        data="data/data.yaml",
        project="runs/val",
        name="validation_results",
        save_json=True,
        plots=True,
        batch=8,
        workers=4,
        imgsz=640,
        conf=0.25,
        iou=0.45,
    )

    print("\nValidation complete! Results saved to 'runs/val/validation_results'")
    print("\n" + "=" * 60)
    print("KEY METRICS:")
    print("=" * 60)
    print(f"mAP@50:      {results.box.map50:.4f} ({results.box.map50 * 100:.2f}%)")
    print(f"mAP@50-95:   {results.box.map:.4f} ({results.box.map * 100:.2f}%)")
    print(f"Precision:   {results.box.mp:.4f} ({results.box.mp * 100:.2f}%)")
    print(f"Recall:      {results.box.mr:.4f} ({results.box.mr * 100:.2f}%)")

    # Per-class metrics if available
    if hasattr(results.box, "maps"):
        print("\nPER-CLASS mAP@50:")
        class_names = ["gloved", "not-gloved"]
        for i, class_map in enumerate(results.box.maps):
            if i < len(class_names):
                print(
                    f"  {class_names[i]:12s}: {class_map:.4f} ({class_map * 100:.2f}%)"
                )

    print("\nNext step: Run inference_script.py or realtime_detection_script.py")


if __name__ == "__main__":
    validate_model()
