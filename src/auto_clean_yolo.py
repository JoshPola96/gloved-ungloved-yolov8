# src/auto_clean_yolo.py

import os
import shutil
import yaml
import glob
from tqdm import tqdm


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def normalize_yolo_coords(line):
    # Ensure coordinates are float and within 0-1
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        c, x, y, w, h = parts
        x, y, w, h = float(x), float(y), float(w), float(h)
        # Clip to 0-1 to fix "bbox_out_of_range" errors
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        return f"{c} {x} {y} {w} {h}"
    except ValueError:
        return None


def run_cleaning():
    cfg = load_config()
    paths = cfg["paths"]

    ensure_dirs([paths["interim_images"], paths["interim_labels"], paths["quarantine"]])

    raw_imgs = glob.glob(os.path.join(paths["raw_images"], "*.*"))
    print(f"🚀 Processing {len(raw_imgs)} images...")

    for img_path in tqdm(raw_imgs):
        basename = os.path.basename(img_path)
        name_only = os.path.splitext(basename)[0]
        label_path = os.path.join(paths["raw_labels"], name_only + ".txt")

        # 1. Check Orphan (Image exists, Label missing)
        if not os.path.exists(label_path):
            shutil.copy(img_path, os.path.join(paths["quarantine"], basename))
            continue  # Skip to next

        # 2. Check Corrupt Label
        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:  # Empty label file
            shutil.copy(img_path, os.path.join(paths["quarantine"], basename))
            shutil.copy(
                label_path, os.path.join(paths["quarantine"], name_only + ".txt")
            )
            continue

        clean_lines = []
        is_corrupt = False

        for line in lines:
            normalized = normalize_yolo_coords(line)
            if normalized:
                # OPTIONAL: logic from your label_swap_script.py
                if cfg["params"]["swap_classes"]:
                    parts = normalized.split()
                    parts[0] = "1" if parts[0] == "0" else "0"
                    normalized = " ".join(parts)
                clean_lines.append(normalized)
            else:
                is_corrupt = True

        if is_corrupt or not clean_lines:
            # Move to quarantine if any line was bad
            shutil.copy(img_path, os.path.join(paths["quarantine"], basename))
            shutil.copy(
                label_path, os.path.join(paths["quarantine"], name_only + ".txt")
            )
        else:
            # 3. Success! Copy to Interim
            shutil.copy(img_path, os.path.join(paths["interim_images"], basename))
            with open(
                os.path.join(paths["interim_labels"], name_only + ".txt"), "w"
            ) as f:
                f.write("\n".join(clean_lines))

    print("✅ Auto-cleaning complete.")
    print(f"   Clean data: {paths['interim_images']}")
    print(f"   Quarantined: {paths['quarantine']}")


if __name__ == "__main__":
    run_cleaning()
