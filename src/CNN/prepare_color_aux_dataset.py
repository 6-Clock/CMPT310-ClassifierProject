"""
Catalog the external "store items by colour" Kaggle dataset into a CSV that
train_cnn.py can merge into training as colour-only auxiliary rows.

Why this exists: our own baseColour classifier is weakest on Brown (test
precision 0.34) and Beige (0.49). This external dataset has 681 Brown
training images vs. our own 402 -- a real boost for the weakest class. It
does NOT have season/usage labels, so these rows can only ever supplement
the colour head (see the has_full_labels masking in train_cnn.py).

Source: https://www.kaggle.com/datasets/imoore/6000-store-items-images-classified-by-color
Download with:
    kaggle datasets download -d imoore/6000-store-items-images-classified-by-color -p data/external/color_ds --unzip

Run from the project root, venv active:
    python src/CNN/prepare_color_aux_dataset.py
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/external/color_ds/train")
OUTPUT_CSV = Path("data/processed/color_aux_v1.csv")

# Maps this dataset's folder names -> our own baseColour taxonomy
# (src/balance_clean_dataset.py's COLOR_MAP). 11 of their 12 classes match
# ours directly. "silver" is intentionally omitted: we have no matching
# class, and force-mapping it to Grey would inject noise (a shiny metallic
# garment doesn't look like matte grey). Our own "Beige" has no counterpart
# here either -- this dataset can't help that gap.
FOLDER_TO_COLOR = {
    "black": "Black",
    "blue": "Blue",
    "brown": "Brown",
    "green": "Green",
    "grey": "Grey",
    "orange": "Orange",
    "pink": "Pink",
    "purple": "Purple",
    "red": "Red",
    "white": "White",
    "yellow": "Yellow",
}


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} not found. Download the dataset first (see the "
            "module docstring for the kaggle CLI command)."
        )

    rows = []
    for folder, colour in FOLDER_TO_COLOR.items():
        folder_path = RAW_DIR / folder
        if not folder_path.exists():
            print(f"[WARNING] Expected folder not found, skipping: {folder_path}")
            continue

        for img_path in sorted(folder_path.glob("*.jpg")):
            rows.append({
                "id": f"colords_{img_path.stem}",
                "raw_image_path": str(img_path),
                "baseColour": colour,
            })

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} auxiliary colour-only rows to {OUTPUT_CSV}")
    print("\nPer-class counts:")
    print(df["baseColour"].value_counts())


if __name__ == "__main__":
    main()
