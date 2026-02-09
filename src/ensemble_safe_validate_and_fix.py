# src/ensemble_safe_validate_and_fix.py

"""
ensemble_safe_validate_and_fix.py

Ensemble label validator & safe auto-corrector.

Primary model: your model (model/best.pt) - must exist.
Second model: pose/hand detector (models/yolov8x-pose.pt) - optional but recommended.
Third check: CLIP ViT-L/14 (auto-downloaded via clip.load).

Default: dry-run (no writes). Use --autofix to write corrected labels into data_corrected/ and perform backups.
"""

import os
import sys
import cv2
import torch
import shutil
import argparse
import csv
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import numpy as np
import clip
import math

# -------------------------
# USER CONFIG / PATHS
# -------------------------
PRIMARY_MODEL_PATH = "runs/train/train/weights"  # <- your model (confirmed)
ENSEMBLE_MODEL = "models/yolov8x-hand-ensemble.pt"
HF_SOURCE = "https://huggingface.co/lewiswatson/yolov8x-tuned-hand-gestures/resolve/main/weights/best.pt"
CLIP_NAME = "ViT-L/14"  # CLIP model (auto-downloaded)

if not os.path.exists(ENSEMBLE_MODEL):
    print("Downloading hand model from HF...")
    temp = YOLO(HF_SOURCE)
    os.makedirs("models", exist_ok=True)
    shutil.copy(temp.ckpt_path, ENSEMBLE_MODEL)
    print("Saved to models/yolov8x-hand-ensemble.pt")

pose_model = YOLO(ENSEMBLE_MODEL)

IMG_DIR = "data/train/images"
LBL_DIR = "data/train/labels"
OUT_LABEL_DIR = "data_corrected/train/labels"
BACKUP_DIR = "backups"
SUSPECTS_DIR = "suspects"
DELETED_DIR = "deleted"
REVIEW_QUEUE_DIR = "review_queue/lowest_conf_300"
LOG_FILE = "label_correction_log.csv"

os.makedirs(OUT_LABEL_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(SUSPECTS_DIR, exist_ok=True)
os.makedirs(DELETED_DIR, exist_ok=True)
os.makedirs(REVIEW_QUEUE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

# -------------------------
# THRESHOLDS & WEIGHTS
# -------------------------
IOU_DUPLICATE_THRESH = 0.5  # IoU threshold to consider duplicate
AUTO_FIX_CONF = 0.90  # final_conf >= this => auto-fix
FLAG_CONF = 0.75  # final_conf in [FLAG_CONF, AUTO_FIX_CONF) => flagged
DELETE_CONF = 0.20  # final_conf < this and no orig label => candidate for deletion (only with --delete-low)
MAX_CROP_SIDE = 2000

# Ensemble weights (sum to 1)
WEIGHT_PRIMARY = 0.6  # your model confidence
WEIGHT_POSE = 0.2  # pose presence boost (0 or 1 for overlap)
WEIGHT_CLIP = 0.2  # CLIP glove conf

# How many images to put in review queue (lowest final_conf suspects)
REVIEW_QUEUE_COUNT = 300


# -------------------------
# Helpers
# -------------------------
def fail(msg):
    print("ERROR:", msg)
    sys.exit(1)


def read_original_labels(lbl_path):
    if not os.path.exists(lbl_path):
        return []
    lines = []
    for ln in open(lbl_path, "r").read().splitlines():
        parts = ln.strip().split()
        if len(parts) >= 5:
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                lines.append((cls, x, y, w, h, ln.strip()))
            except:
                continue
    return lines


def write_yolo_lines(path, yolo_lines):
    with open(path, "w") as f:
        f.write("\n".join(yolo_lines))


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return 0.0 if denom == 0 else inter / denom


def yolo_coord_from_box(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def ensure_model_exists(path, name):
    if not os.path.exists(path):
        print(f"Model '{name}' not found at: {path}")
        return False
    return True


# -------------------------
# Main processing
# -------------------------
def process_dataset(args):
    # check primary model
    if not ensure_model_exists(PRIMARY_MODEL_PATH, "primary (your) model"):
        fail("Please place your best model at: " + PRIMARY_MODEL_PATH)
    # load primary
    print("Loading primary model:", PRIMARY_MODEL_PATH)
    primary_model = YOLO(PRIMARY_MODEL_PATH)

    # load pose model if present; if missing, we'll continue but pose_boost will be 0
    pose_model = None
    if ensure_model_exists(ENSEMBLE_MODEL, "pose model"):
        print("Loading pose model:", ENSEMBLE_MODEL)
        pose_model = YOLO(ENSEMBLE_MODEL)
    else:
        print(
            "Warning: pose model not found. Pose-based boosting will be skipped. Place model at:",
            ENSEMBLE_MODEL,
        )

    # load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP model:", CLIP_NAME, "on", device)
    clip_model, preprocess = clip.load(CLIP_NAME, device=device)
    text_labels = ["a gloved hand", "a bare hand"]
    text_tokens = clip.tokenize(text_labels).to(device)

    # prepare CSV
    header = [
        "image",
        "det_idx",
        "x1",
        "y1",
        "x2",
        "y2",
        "orig_count",
        "primary_cls",
        "primary_conf",
        "pose_overlap",
        "clip_pred",
        "clip_conf",
        "final_conf",
        "action",
        "notes",
    ]
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    images = [
        f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if args.sample:
        images = images[: args.sample]

    per_image_lowconf = []  # collect (image, min_final_conf) for suspects sorting

    stats = {
        "images": 0,
        "detections": 0,
        "auto_fixed": 0,
        "flagged": 0,
        "deleted": 0,
        "kept": 0,
    }

    for img_file in tqdm(images, desc="processing images"):
        stats["images"] += 1
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_file)[0] + ".txt")
        out_lbl_path = os.path.join(
            OUT_LABEL_DIR, os.path.splitext(img_file)[0] + ".txt"
        )

        # backup once
        bkp_img_dir = os.path.join(BACKUP_DIR, "images")
        os.makedirs(bkp_img_dir, exist_ok=True)
        bkp_lbl_dir = os.path.join(BACKUP_DIR, "labels")
        os.makedirs(bkp_lbl_dir, exist_ok=True)
        if not os.path.exists(os.path.join(bkp_img_dir, img_file)):
            shutil.copy2(img_path, os.path.join(bkp_img_dir, img_file))
        if os.path.exists(lbl_path) and not os.path.exists(
            os.path.join(bkp_lbl_dir, os.path.basename(lbl_path))
        ):
            shutil.copy2(
                lbl_path, os.path.join(bkp_lbl_dir, os.path.basename(lbl_path))
            )

        orig_labels = read_original_labels(lbl_path)
        orig_lines = [t[-1] for t in orig_labels]

        img = cv2.imread(img_path)
        if img is None:
            print("Could not read image:", img_path)
            continue
        h, w = img.shape[:2]

        # primary model predictions (assume primary outputs class & confidence)
        try:
            primary_res = primary_model(img)[0]
        except Exception as e:
            print("Primary model inference error on", img_file, e)
            continue

        # For ensemble, build list of boxes from primary model
        primary_boxes = []  # (x1,y1,x2,y2,cls,conf)
        for b in primary_res.boxes:
            try:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                # attempt to get class/conf from b.cls / b.conf; fallbacks if not present
                cls = int(b.cls[0]) if hasattr(b, "cls") and len(b.cls) > 0 else None
                conf = (
                    float(b.conf[0]) if hasattr(b, "conf") and len(b.conf) > 0 else None
                )
            except Exception:
                # fallback: try .xyxy, .conf, .cls attributes (Ultralytics versions vary)
                data = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, data.tolist())
                cls = None
                conf = None
                try:
                    conf = float(b.conf.cpu().numpy()[0])
                except:
                    conf = 1.0
            primary_boxes.append((x1, y1, x2, y2, cls, conf))

        # pose model detections
        pose_boxes = []
        if pose_model is not None:
            try:
                pose_res = pose_model(img)[0]
                for b in pose_res.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        pose_boxes.append((x1, y1, x2, y2))
                    except:
                        coords = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords.tolist())
                        pose_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                print("Pose model error:", e)
                pose_boxes = []

        # if primary found no boxes, fallback: use pose detections (we will attempt to classify those with CLIP)
        boxes_to_process = (
            primary_boxes[:]
            if primary_boxes
            else [(x1, y1, x2, y2, None, None) for (x1, y1, x2, y2) in pose_boxes]
        )

        stats["detections"] += len(boxes_to_process)

        final_yolo_lines = list(
            orig_lines
        )  # start with originals to preserve human labels

        min_final_conf_for_image = 1.0

        det_idx = 0
        for x1, y1, x2, y2, prim_cls, prim_conf in boxes_to_process:
            det_idx += 1
            # clamp & sanity check
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if (
                (x2 - x1) <= 0
                or (y2 - y1) <= 0
                or (x2 - x1) > MAX_CROP_SIDE
                or (y2 - y1) > MAX_CROP_SIDE
            ):
                with open(LOG_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            img_file,
                            det_idx,
                            x1,
                            y1,
                            x2,
                            y2,
                            len(orig_labels),
                            "",
                            "",
                            "",
                            "skipped_invalid_crop",
                            "",
                        ]
                    )
                continue

            # pose overlap score (0 or 1)
            pose_overlap = 0.0
            for pb in pose_boxes:
                if iou((x1, y1, x2, y2), pb) > IOU_DUPLICATE_THRESH:
                    pose_overlap = 1.0
                    break

            # CLIP prediction on crop
            crop = img[y1:y2, x1:x2]
            clip_pred = -1
            clip_conf = 0.0
            try:
                img_t = (
                    preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    logits_per_image, _ = clip_model(img_t, text_tokens)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                clip_pred = int(np.argmax(probs))
                clip_conf = float(probs[clip_pred])
            except Exception:
                clip_pred = -1
                clip_conf = 0.0

            # primary confidence fallback
            prim_conf_num = (
                prim_conf
                if (prim_conf is not None and not math.isnan(prim_conf))
                else 0.5
            )

            # final confidence = weighted sum
            final_conf = (
                (WEIGHT_PRIMARY * prim_conf_num)
                + (WEIGHT_POSE * pose_overlap)
                + (WEIGHT_CLIP * clip_conf)
            )

            # clamp
            final_conf = max(0.0, min(1.0, final_conf))
            min_final_conf_for_image = min(min_final_conf_for_image, final_conf)

            # determine predicted class: prefer primary model class if present else CLIP
            predicted_class = (
                prim_cls
                if prim_cls is not None
                else (clip_pred if clip_pred >= 0 else 0)
            )

            # decide action
            action = ""
            notes = ""

            if final_conf >= AUTO_FIX_CONF:
                # create yolo line and append if not duplicate (IoU check)
                xc, yc, bw, bh = yolo_coord_from_box(x1, y1, x2, y2, w, h)
                new_line = f"{predicted_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                duplicate_found = False
                for existing in final_yolo_lines:
                    parts = existing.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        ex_x, ex_y, ex_w, ex_h = map(float, parts[1:5])
                        ex_x1 = int((ex_x - ex_w / 2.0) * w)
                        ex_y1 = int((ex_y - ex_h / 2.0) * h)
                        ex_x2 = int((ex_x + ex_w / 2.0) * w)
                        ex_y2 = int((ex_y + ex_h / 2.0) * h)
                        if (
                            iou((x1, y1, x2, y2), (ex_x1, ex_y1, ex_x2, ex_y2))
                            > IOU_DUPLICATE_THRESH
                        ):
                            duplicate_found = True
                            break
                    except:
                        continue
                if not duplicate_found:
                    final_yolo_lines.append(new_line)
                    action = "auto_fixed"
                    stats["auto_fixed"] += 1
                else:
                    action = "dup_skipped"
            elif final_conf >= FLAG_CONF:
                # flagged for review (copy to suspects)
                action = "flagged"
                stats["flagged"] += 1
                shutil.copy2(img_path, os.path.join(SUSPECTS_DIR, img_file))
            else:
                # low confidence
                action = "low_conf"
                # optionally delete if user asked AND no original labels
                if (
                    (len(orig_labels) == 0)
                    and args.delete_low_conf
                    and final_conf < DELETE_CONF
                ):
                    try:
                        shutil.move(img_path, os.path.join(DELETED_DIR, img_file))
                        if os.path.exists(lbl_path):
                            shutil.move(
                                lbl_path,
                                os.path.join(DELETED_DIR, os.path.basename(lbl_path)),
                            )
                        action = "deleted_low_conf"
                        stats["deleted"] += 1
                    except Exception:
                        action = "delete_failed"

            # write detection log
            with open(LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        img_file,
                        det_idx,
                        x1,
                        y1,
                        x2,
                        y2,
                        len(orig_labels),
                        prim_cls,
                        prim_conf_num,
                        pose_overlap,
                        clip_pred,
                        clip_conf,
                        f"{final_conf:.4f}",
                        action,
                        notes,
                    ]
                )

        # after all dets, decide what to write (if autofix)
        if not args.dry_run:
            # write final_yolo_lines to out label
            write_yolo_lines(out_lbl_path, final_yolo_lines)

        # collect min confidence per image for ranking suspects
        per_image_lowconf.append((img_file, min_final_conf_for_image))

    # build review queue: choose lowest-confidence flagged images up to REVIEW_QUEUE_COUNT
    # first read log to find flagged images and their final_conf; but we already have per_image_lowconf
    # filter to suspects (images present in SUSPECTS_DIR)
    suspects = [
        (img, conf)
        for (img, conf) in per_image_lowconf
        if os.path.exists(os.path.join(SUSPECTS_DIR, img))
    ]
    suspects = sorted(suspects, key=lambda x: x[1])  # ascending (lowest conf first)
    queue = suspects[:REVIEW_QUEUE_COUNT]
    # copy queue images to REVIEW_QUEUE_DIR
    for img, conf in queue:
        src = os.path.join(SUSPECTS_DIR, img)
        dst = os.path.join(REVIEW_QUEUE_DIR, img)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # summary
    print("\n=== SUMMARY ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("Logs:", LOG_FILE)
    print("Suspects:", SUSPECTS_DIR)
    print("Review queue (lowest confidence):", REVIEW_QUEUE_DIR)
    if args.dry_run:
        print(
            "DRY RUN: Nothing was written. Use --autofix to write corrected labels to data_corrected/..."
        )


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="Process only first N images")
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Do not write changes (default behavior).",
    )
    p.add_argument(
        "--autofix",
        dest="dry_run",
        action="store_false",
        help="Write corrected labels to data_corrected/ (disables dry-run).",
    )
    p.add_argument(
        "--delete-low",
        dest="delete_low_conf",
        action="store_true",
        help="Move very-low-confidence images with no original label to deleted/ (dangerous).",
    )
    p.add_argument(
        "--pose-model",
        type=str,
        default=None,
        help="Optional path to pose/hand model (overrides ENSEMBLE_MODEL).",
    )
    args = p.parse_args()

    # override pose model path if provided
    if args.pose_model:
        ENSEMBLE_MODEL = args.pose_model

    if "dry_run" not in vars(args):
        args.dry_run = True

    process_dataset(args)
