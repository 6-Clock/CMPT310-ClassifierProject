"""
Fetch + catalog the "fashion-lookmatch" HuggingFace dataset as auxiliary
SEASON and USAGE training data (colour is intentionally ignored -- see below).

Why this exists: our own season data is severely imbalanced (Spring 95 /
Winter 594 vs. Summer 8404 in training), and usage's Formal class is thin.
This dataset's season split is close to even (~115-133 images per season)
and directly targets Formal/Sports.

IMPORTANT CAVEAT: every image in this dataset is AI-generated (Stable
Diffusion XL from a text prompt), not a real photograph. That's a real
domain gap from our own e-commerce product photos. See train_cnn.py's
Phase 4 for how we contain that risk (freeze everything except the
season/usage heads, same principle as Phase 3's colour-only fine-tune).

Source: https://huggingface.co/datasets/orianrivlin/fashion-lookmatch-dataset
(accessed via the datasets-server rows API, no local download step needed
before running this script -- it fetches directly).

Filtering applied:
  - Only "apparel" categories are kept (jacket, hoodie, chinos, t_shirt,
    shirt, jeans, dress, skirt). bag/watch/sneakers/boots are dropped --
    our own dataset is filtered to masterCategory == "Apparel" too
    (see balance_clean_dataset.py), so accessories/footwear are off-domain.
  - season: kept as-is if present, else marked has_season=False (~28% of
    rows have no season entry at all).
  - usage: derived from the free-form `style_tags` list via a priority
    mapping (every apparel row has at least one tag, so usage is always
    derivable -- there just isn't an "Ethnic" tag anywhere in this dataset,
    so that gap stays open regardless):
        "formal" present         -> Formal
        else "sporty" present    -> Sports
        else (smart_casual, vintage, streetwear, minimal, casual, ...)
                                  -> Casual

Run from the project root, venv active:
    python src/CNN/prepare_style_aux_dataset.py
"""

from pathlib import Path
import json
import urllib.request

import pandas as pd

DATASET_NAME = "orianrivlin/fashion-lookmatch-dataset"
ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET_NAME.replace('/', '%2F')}&config=default&split=train"
    "&offset={offset}&length=100"
)
TOTAL_ROWS = 1000
PAGE_SIZE = 100

IMAGE_DIR = Path("data/external/lookmatch_ds")
OUTPUT_CSV = Path("data/processed/style_aux_v1.csv")

APPAREL_CATEGORIES = {
    "jacket", "hoodie", "chinos", "t_shirt", "shirt", "jeans", "dress", "skirt",
}


def fetch_all_rows():
    """Pages through the datasets-server rows API to get all 1000 rows."""
    all_rows = []
    for offset in range(0, TOTAL_ROWS, PAGE_SIZE):
        url = ROWS_API.format(offset=offset)
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
        all_rows.extend(data["rows"])
        print(f"Fetched rows {offset}-{offset + PAGE_SIZE}")
    return all_rows


def map_usage(style_tags):
    """Priority mapping from free-form style tags to our 4-class usage taxonomy."""
    tags = set(style_tags or [])
    if "formal" in tags:
        return "Formal"
    if "sporty" in tags:
        return "Sports"
    # smart_casual, vintage, streetwear, minimal, casual (or anything else) -> Casual
    return "Casual"


def download_image(url, dest_path):
    if dest_path.exists():
        return True
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as error:
        print(f"[WARNING] Failed to download {dest_path.name}: {error}")
        return False


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching dataset metadata...")
    entries = fetch_all_rows()

    apparel_entries = [e for e in entries if e["row"]["category"] in APPAREL_CATEGORIES]
    print(f"Total rows: {len(entries)} | Apparel-only rows: {len(apparel_entries)}")

    rows = []
    for i, entry in enumerate(apparel_entries, start=1):
        row_idx = entry["row_idx"]
        row = entry["row"]

        dest_path = IMAGE_DIR / f"{row_idx}.jpg"
        ok = download_image(row["image"]["src"], dest_path)
        if not ok:
            continue

        season = row["season"]  # already lowercase e.g. "summer"; None if missing
        rows.append({
            "id": f"lookmatch_{row_idx}",
            "raw_image_path": str(dest_path),
            "season": season.capitalize() if season else "",
            "has_season": bool(season),
            "usage": map_usage(row["style_tags"]),
            "has_usage": True,  # every apparel row has at least one style tag
        })

        if i % 100 == 0:
            print(f"Downloaded {i}/{len(apparel_entries)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} auxiliary season/usage rows to {OUTPUT_CSV}")
    print(f"\nRows with a usable season: {df['has_season'].sum()}")
    print("\nSeason distribution (has_season rows only):")
    print(df[df["has_season"]]["season"].value_counts())
    print("\nUsage distribution:")
    print(df["usage"].value_counts())


if __name__ == "__main__":
    main()
