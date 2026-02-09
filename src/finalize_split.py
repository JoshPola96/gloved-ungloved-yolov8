# src/finalize_split.py

import os
import shutil
import random
import yaml
from tqdm import tqdm


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

# Source Paths (Your Golden Dataset)
SRC_IMG = cfg["paths"]["raw_images"]
SRC_LBL = cfg["paths"]["raw_labels"]

# Destination Paths (Final Training Set)
DEST_BASE = "data"
SPLITS = ["train", "val", "test"]
RATIOS = [0.80, 0.15, 0.05]  # 80% Train, 15% Val, 5% Test


def main():
    # 1. Gather all valid pairs
    print(f"🔍 Scanning {SRC_IMG}...")
    images = [f for f in os.listdir(SRC_IMG) if f.endswith((".jpg", ".png"))]
    valid_pairs = []

    for img in images:
        label = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(os.path.join(SRC_LBL, label)):
            valid_pairs.append((img, label))

    print(f"✅ Found {len(valid_pairs)} valid image/label pairs.")

    # 2. Shuffle
    random.shuffle(valid_pairs)

    # 3. Calculate split indices
    total = len(valid_pairs)
    train_end = int(total * RATIOS[0])
    val_end = train_end + int(total * RATIOS[1])

    datasets = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:],
    }

    # 4. Move files
    print("📦 Distributing files...")
    for split, pairs in datasets.items():
        # Create directories
        img_dest = os.path.join(DEST_BASE, "images", split)
        lbl_dest = os.path.join(DEST_BASE, "labels", split)
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for img, lbl in tqdm(pairs, desc=f"Creating {split} set"):
            shutil.copy2(os.path.join(SRC_IMG, img), os.path.join(img_dest, img))
            shutil.copy2(os.path.join(SRC_LBL, lbl), os.path.join(lbl_dest, lbl))

    print(f"\n🎉 Done! Final dataset is in '{DEST_BASE}/'")
    print(f"   Train: {len(datasets['train'])}")
    print(f"   Val:   {len(datasets['val'])}")
    print(f"   Test:  {len(datasets['test'])}")


if __name__ == "__main__":
    main()
