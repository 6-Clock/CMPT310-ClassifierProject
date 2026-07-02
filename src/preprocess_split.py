from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from tqdm import tqdm

# Project paths
# RAW_CSV = Path("data/raw/styles.csv")
# RAW_IMAGE_DIR = Path("data/raw/images")
CLEANED_CSV = Path("data/processed/clean_colour_season_style.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_IMAGE_DIR = PROCESSED_DIR / "images_96"

# Labels our project wants to predict
TARGET_COLS = ["baseColour", "season", "usage"]     # removed "articleType"

# Image preprocessing settings
IMG_SIZE = 96

# Makes the split reproducible
RANDOM_STATE = 42

def load_clean_metadata() -> pd.DataFrame:
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {CLEANED_CSV}")

    df = pd.read_csv(CLEANED_CSV)

    keep_cols = ["id", "image_path"] + TARGET_COLS
    df = df[keep_cols].copy()

    return df.reset_index(drop=True)

# def clean_metadata() -> pd.DataFrame:
#     df = pd.read_csv(RAW_CSV, on_bad_lines="skip")
#     print(f"Loaded rows: {len(df)}")

#     df = df.dropna(subset=TARGET_COLS).copy()
#     print(f"After dropping missing labels: {len(df)}")

#     df["raw_image_path"] = df["id"].astype(str).apply(
#         lambda image_id: str(RAW_IMAGE_DIR / f"{image_id}.jpg")
#     )

#     df["image_exists"] = df["raw_image_path"].apply(lambda path: Path(path).exists())
#     df = df[df["image_exists"]].copy()
#     print(f"After dropping missing images: {len(df)}")

#     keep_cols = ["id", "raw_image_path"] + TARGET_COLS
#     return df[keep_cols].reset_index(drop=True)


def resize_with_padding(input_path: Path, output_path: Path, size: int = IMG_SIZE):
    img = Image.open(input_path).convert("RGB")

    # Resize while preserving aspect ratio, then pad to square.
    padded = ImageOps.pad(
        img,
        size=(size, size),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )

    padded.save(output_path, format="JPEG", quality=95)


def resize_images(df: pd.DataFrame) -> pd.DataFrame:
    PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    processed_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resizing images"):
        input_path = Path(row["image_path"])
        output_path = PROCESSED_IMAGE_DIR / input_path.name

        if not output_path.exists():
            resize_with_padding(input_path, output_path)

        processed_paths.append(str(output_path))

    df = df.copy()
    df["image_path"] = processed_paths

    keep_cols = ["id", "image_path"] + TARGET_COLS
    return df[keep_cols].reset_index(drop=True)


def split_metadata(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return train_df, val_df, test_df


def save_outputs(df, train_df, val_df, test_df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DIR / "metadata_clean.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\nSaved files:")
    print(f"- {PROCESSED_DIR / 'metadata_clean.csv'}")
    print(f"- {PROCESSED_DIR / 'train.csv'}")
    print(f"- {PROCESSED_DIR / 'val.csv'}")
    print(f"- {PROCESSED_DIR / 'test.csv'}")

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)}")

    print(f"\nProcessed images saved in: {PROCESSED_IMAGE_DIR}")


def main():
    # df = clean_metadata()
    df = load_clean_metadata()
    df = resize_images(df)
    train_df, val_df, test_df = split_metadata(df)
    save_outputs(df, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
