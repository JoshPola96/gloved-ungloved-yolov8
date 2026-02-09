# realtime_detection_script.py

import os
import glob
import re
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time


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


class SmoothDetector:
    """
    Temporal smoothing for stable real-time detections.
    Reduces jitter and false positives.
    """

    def __init__(self, window_size=5):
        self.detections_history = deque(maxlen=window_size)

    def smooth(self, current_detections):
        """Apply temporal smoothing using a sliding window."""
        self.detections_history.append(current_detections)

        if len(self.detections_history) < 2:
            return current_detections

        # Count class occurrences across frames
        class_votes = {}
        for frame_dets in self.detections_history:
            for det in frame_dets:
                cls = det["class"]
                class_votes[cls] = class_votes.get(cls, 0) + 1

        # Filter based on temporal consistency
        threshold = len(self.detections_history) * 0.4
        stable_detections = []
        for det in current_detections:
            if class_votes.get(det["class"], 0) >= threshold:
                stable_detections.append(det)

        return stable_detections


def run_real_time_detection():
    """
    Runs optimized real-time glove detection on webcam.
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
    print(f"Loading model from: {model_path}")

    # Initialize smoother
    smoother = SmoothDetector(window_size=5)

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # FPS calculation
    fps_history = deque(maxlen=30)
    prev_time = time.time()

    print("\n" + "=" * 60)
    print("Real-time detection started!")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=0.35, iou=0.45, verbose=False, stream=False)

        # Extract detections
        current_detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                current_detections.append(
                    {
                        "class": int(box.cls[0]),
                        "conf": float(box.conf[0]),
                        "box": box.xyxy[0].cpu().numpy(),
                    }
                )

        # Apply temporal smoothing
        stable_detections = smoother.smooth(current_detections)

        # Visualize
        annotated_frame = frame.copy()
        class_names = model.names

        for det in stable_detections:
            box = det["box"]
            cls = det["class"]
            conf = det["conf"]

            x1, y1, x2, y2 = map(int, box)

            # Color coding: Green=gloved, Red=not-gloved
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)

            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_names[cls]} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)

        # Display info
        info_text = f"FPS: {avg_fps:.1f} | Detections: {len(stable_detections)}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Glove Detection - Press 'q' to quit", annotated_frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snapshot_name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_name, annotated_frame)
            print(f"Snapshot saved: {snapshot_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.")


if __name__ == "__main__":
    run_real_time_detection()
