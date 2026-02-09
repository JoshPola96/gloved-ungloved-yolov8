# src/yolo_labeler.py

import streamlit as st
import os
import yaml
import cv2
import glob
import shutil
import numpy as np
from PIL import Image

# --- Config & Setup ---
st.set_page_config(layout="wide", page_title="YOLO Cleaner")


@st.cache_data
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

# Define paths
IMG_DIR = cfg["paths"]["interim_images"]
LBL_DIR = cfg["paths"]["interim_labels"]
GOLDEN_IMG = cfg["paths"]["golden_images"]
GOLDEN_LBL = cfg["paths"]["golden_labels"]

# Ensure golden dirs exist
os.makedirs(GOLDEN_IMG, exist_ok=True)
os.makedirs(GOLDEN_LBL, exist_ok=True)


# Helper: IOU for duplicate detection
def calculate_iou(box1, box2):
    # box = [x, y, w, h]
    x1_min = box1["x"] - box1["w"] / 2
    x1_max = box1["x"] + box1["w"] / 2
    y1_min = box1["y"] - box1["h"] / 2
    y1_max = box1["y"] + box1["h"] / 2

    x2_min = box2["x"] - box2["w"] / 2
    x2_max = box2["x"] + box2["w"] / 2
    y2_min = box2["y"] - box2["h"] / 2
    y2_max = box2["y"] + box2["h"] / 2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0, xi_max - xi_min)
    inter_h = max(0, yi_max - yi_min)
    intersection = inter_w * inter_h

    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# Helper to get list of images
def get_images():
    all_imgs = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMG_DIR, "*.png"))
    )
    return all_imgs


# --- Session State ---
if "file_list" not in st.session_state:
    st.session_state.file_list = get_images()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


# --- Logic ---
def save_and_next(img_path, labels, move_file=True):
    basename = os.path.basename(img_path)
    txt_name = os.path.splitext(basename)[0] + ".txt"

    out_lbl_path = os.path.join(GOLDEN_LBL, txt_name)
    with open(out_lbl_path, "w") as f:
        for l in labels:
            f.write(f"{l['cls']} {l['x']} {l['y']} {l['w']} {l['h']}\n")

    if move_file:
        try:
            shutil.move(img_path, os.path.join(GOLDEN_IMG, basename))
            old_lbl = os.path.join(LBL_DIR, txt_name)
            if os.path.exists(old_lbl):
                os.remove(old_lbl)
        except PermissionError:
            st.error("Windows locked the file. Please try clicking 'Save' again.")
            return

    st.toast(f"Saved {basename}", icon="💾")
    st.session_state.current_idx += 1
    st.rerun()


def delete_and_next(img_path):
    basename = os.path.basename(img_path)
    txt_name = os.path.splitext(basename)[0] + ".txt"
    quarantine = cfg["paths"]["quarantine"]
    os.makedirs(quarantine, exist_ok=True)

    try:
        shutil.move(img_path, os.path.join(quarantine, basename))
        old_lbl = os.path.join(LBL_DIR, txt_name)
        if os.path.exists(old_lbl):
            shutil.move(old_lbl, os.path.join(quarantine, txt_name))
    except PermissionError:
        st.error("Windows locked the file. Try again.")
        return

    st.toast("Moved to Quarantine", icon="🗑️")
    st.session_state.current_idx += 1
    st.rerun()


# --- UI ---
st.title("🥊 Glove/No-Glove Visual Fixer")

if st.session_state.current_idx < len(st.session_state.file_list):
    curr_img_path = st.session_state.file_list[st.session_state.current_idx]

    if not os.path.exists(curr_img_path):
        st.session_state.current_idx += 1
        st.rerun()

    # Load Image Safe
    try:
        with Image.open(curr_img_path) as img_ref:
            image = img_ref.copy().convert("RGB")
    except Exception as e:
        st.error(f"Error reading image: {e}")
        st.stop()

    w_img, h_img = image.size
    txt_path = os.path.join(
        LBL_DIR, os.path.splitext(os.path.basename(curr_img_path))[0] + ".txt"
    )

    current_labels = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    current_labels.append(
                        {
                            "cls": int(parts[0]),
                            "x": float(parts[1]),
                            "y": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        }
                    )

    # --- LAYOUT ---
    col_img, col_ctrl = st.columns([4, 1])

    # --- CONTROLS ---
    with col_ctrl:
        st.write(f"**Image:** {os.path.basename(curr_img_path)}")
        st.progress(
            (st.session_state.current_idx + 1) / len(st.session_state.file_list)
        )

        st.write("### 🔍 Zoom")
        img_width = st.slider("Size", 300, 2000, 800, 50, label_visibility="collapsed")

        st.divider()
        st.write("### 🏷️ Annotations")

        final_labels_to_save = []

        # We iterate and create controls
        for i, lbl in enumerate(current_labels):
            # Check for duplicates (simple IOU check against others)
            is_duplicate = False
            for j, other in enumerate(current_labels):
                if i != j and calculate_iou(lbl, other) > 0.85:  # 85% overlap threshold
                    is_duplicate = True
                    break

            with st.container():
                c1, c2 = st.columns([0.15, 0.85])
                with c1:
                    # The "Keep" Checkbox
                    keep = st.checkbox("", value=True, key=f"chk_{i}")
                with c2:
                    # Label details
                    cls_name = cfg["classes"][lbl["cls"]]
                    warn = "⚠️ **Overlap**" if is_duplicate else ""
                    st.markdown(f"**Box {i}:** {cls_name} {warn}")

                    # Edit Class
                    if keep:
                        new_cls = st.radio(
                            "Class",
                            [0, 1],
                            index=lbl["cls"],
                            format_func=lambda x: cfg["classes"][x],
                            key=f"rad_{i}",
                            horizontal=True,
                            label_visibility="collapsed",
                        )
                        lbl["cls"] = new_cls
                        final_labels_to_save.append(lbl)

                    # Store state for visualization
                    lbl["_keep"] = keep

            st.divider()

        # Actions
        if st.button("✅ Save Changes", type="primary", use_container_width=True):
            save_and_next(curr_img_path, final_labels_to_save)

        if st.button("🗑️ Junk Image", type="secondary", use_container_width=True):
            delete_and_next(curr_img_path)

    # --- IMAGE RENDER ---
    with col_img:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        viz = cfg.get("visualization", {})
        box_th = viz.get("box_thickness", 3)
        font_sc = viz.get("font_scale", 1.0)
        text_th = viz.get("text_thickness", 2)

        for i, lbl in enumerate(current_labels):
            # Check if user kept or removed this box
            is_kept = lbl.get("_keep", True)

            if is_kept:
                # Normal Bright Colors
                color = (0, 255, 0) if lbl["cls"] == 0 else (0, 0, 255)
                thickness = box_th
                text_color = (255, 255, 255)
            else:
                # Removed Style: Grey, Dashed (simulated by thin line), No background text
                color = (128, 128, 128)  # Grey
                thickness = 1
                text_color = (128, 128, 128)

            x1 = int((lbl["x"] - lbl["w"] / 2) * w_img)
            y1 = int((lbl["y"] - lbl["h"] / 2) * h_img)
            x2 = int((lbl["x"] + lbl["w"] / 2) * w_img)
            y2 = int((lbl["y"] + lbl["h"] / 2) * h_img)

            # Draw Box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)

            # Label Text
            label_text = f"{i}: {cfg['classes'][lbl['cls']]}"
            if not is_kept:
                label_text += " (DEL)"

            (w_text, h_text), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_sc, text_th
            )

            # Only draw solid background if kept
            if is_kept:
                cv2.rectangle(
                    img_cv, (x1, y1 - h_text - 10), (x1 + w_text, y1), color, -1
                )

            cv2.putText(
                img_cv,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_sc,
                text_color,
                text_th,
            )

        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), width=img_width)

else:
    st.success("Dataset Complete!")
