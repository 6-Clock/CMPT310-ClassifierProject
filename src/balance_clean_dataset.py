from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image


RAW_CSV = Path("data/raw/styles.csv")
IMAGE_DIR = Path("data/raw/images")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "clean_apparel_v2.csv"

REPORT_DIR = Path("reports")
CONFLICT_REPORT = REPORT_DIR / "v2_conflicting_duplicates.csv"
IMAGE_ERROR_REPORT = REPORT_DIR / "v2_unreadable_images.csv"
MAPPING_REPORT = REPORT_DIR / "v2_color_mapping.csv"


VALID_USAGE = {
    "Casual",
    "Ethnic",
    "Sports",
    "Formal",
}

VALID_SEASONS = {
    "Summer",
    "Fall",
    "Winter",
    "Spring",
}

# Manually confirmed metadata error.
EXCLUDED_IDS = {
    51036,
}


COLOR_MAP = {
    "Black": "Black",

    "Blue": "Blue",
    "Navy Blue": "Blue",
    "Teal": "Blue",
    "Turquoise Blue": "Blue",
    "Sea Green": "Blue",

    "White": "White",
    "Off White": "White",

    "Grey": "Grey",
    "Charcoal": "Grey",
    "Grey Melange": "Grey",
    "Silver": "Grey",

    "Green": "Green",
    "Olive": "Green",
    "Lime Green": "Green",
    "Fluorescent Green": "Green",

    "Red": "Red",
    "Maroon": "Red",
    "Burgundy": "Red",
    "Rust": "Red",

    "Pink": "Pink",
    "Rose": "Pink",
    "Magenta": "Pink",
    "Peach": "Pink",

    "Purple": "Purple",
    "Lavender": "Purple",
    "Mauve": "Purple",

    "Brown": "Brown",
    "Coffee Brown": "Brown",
    "Mushroom Brown": "Brown",

    "Yellow": "Yellow",
    "Mustard": "Yellow",

    "Beige": "Beige",
    "Cream": "Beige",
    "Khaki": "Beige",
    "Skin": "Beige",
    "Nude": "Beige",
    "Tan": "Beige",
    "Taupe": "Beige",
    "Gold": "Beige",

    "Orange": "Orange",
}


def print_distribution(df, title):
    print(f"\n===== {title} =====")

    print("\nUsage distribution:")
    print(df["usage"].value_counts())

    print("\nSeason distribution:")
    print(df["season"].value_counts())

    print("\nColour distribution:")
    print(df["baseColour"].value_counts())


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def check_image(path):
    if not path.exists():
        return False, "Image file does not exist"

    try:
        with Image.open(path) as image:
            image.convert("RGB").load()

        return True, ""

    except Exception as error:
        return False, str(error)


def load_and_filter_metadata():
    df = pd.read_csv(RAW_CSV, on_bad_lines="skip")

    print(f"Raw rows: {len(df):,}")

    needed_cols = [
        "id",
        "masterCategory",
        "baseColour",
        "season",
        "usage",
        "productDisplayName",
    ]

    df = df[needed_cols].copy()

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "id",
            "masterCategory",
            "baseColour",
            "season",
            "usage",
        ]
    ).copy()

    df["id"] = df["id"].astype(int)

    print(f"Rows with required metadata: {len(df):,}")

    # Keep only clothing products based on project scope.
    df = df[
        df["masterCategory"] == "Apparel"
    ].copy()

    print(f"Apparel rows: {len(df):,}")

    # Keep the four project usage and season categories.
    df = df[
        df["usage"].isin(VALID_USAGE)
        & df["season"].isin(VALID_SEASONS)
    ].copy()

    print(
        "Rows after usage and season filtering: "
        f"{len(df):,}"
    )

    # Remove manually confirmed metadata errors.
    df = df[
        ~df["id"].isin(EXCLUDED_IDS)
    ].copy()

    # Preserve the original colour before consolidating shades.
    df["originalColour"] = df["baseColour"]

    df["baseColour"] = (
        df["originalColour"].map(COLOR_MAP)
    )

    # Multi and any unsupported colours are excluded.
    df = df.dropna(
        subset=["baseColour"]
    ).copy()

    print(f"Rows after colour mapping: {len(df):,}")

    df["image_path"] = df["id"].apply(
        lambda image_id: (
            IMAGE_DIR / f"{image_id}.jpg"
        )
    )

    return df


def validate_and_hash_images(df):
    valid_rows = []
    image_errors = []

    for number, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        image_path = row.image_path
        readable, error = check_image(image_path)

        if not readable:
            image_errors.append(
                {
                    "id": row.id,
                    "image_path": str(image_path),
                    "originalColour": row.originalColour,
                    "mappedColour": row.baseColour,
                    "error": error,
                }
            )
            continue

        row_data = row._asdict()
        row_data["sha256"] = sha256_file(image_path)
        valid_rows.append(row_data)

        if number % 2000 == 0:
            print(
                f"Validated {number:,}/{len(df):,} "
                "candidate rows"
            )

    valid_df = pd.DataFrame(valid_rows)
    error_df = pd.DataFrame(image_errors)

    return valid_df, error_df


def remove_exact_duplicates(df):
    clean_parts = []
    conflicting_parts = []

    for _, group in df.groupby(
        "sha256",
        sort=False,
    ):
        label_variants = group[
            [
                "baseColour",
                "season",
                "usage",
            ]
        ].drop_duplicates()

        # Identical images with contradictory labels are unreliable.
        if len(label_variants) > 1:
            conflicting_parts.append(group)
            continue

        # Keep one representative of identical, agreeing rows.
        representative = (
            group.sort_values("id").iloc[[0]]
        )

        clean_parts.append(representative)

    clean_df = (
        pd.concat(
            clean_parts,
            ignore_index=True,
        )
        .sort_values("id")
        .reset_index(drop=True)
    )

    if conflicting_parts:
        conflicting_df = (
            pd.concat(
                conflicting_parts,
                ignore_index=True,
            )
            .sort_values(
                ["sha256", "id"]
            )
            .reset_index(drop=True)
        )
    else:
        conflicting_df = pd.DataFrame()

    return clean_df, conflicting_df


def validate_final_dataset(df):
    target_cols = [
        "baseColour",
        "season",
        "usage",
    ]

    assert df["id"].duplicated().sum() == 0
    assert df["sha256"].duplicated().sum() == 0
    assert df[target_cols].isna().sum().sum() == 0

    assert set(df["baseColour"]).issubset(
        set(COLOR_MAP.values())
    )

    assert set(df["season"]).issubset(
        VALID_SEASONS
    )

    assert set(df["usage"]).issubset(
        VALID_USAGE
    )


def save_outputs(
    clean_df,
    image_errors,
    conflicting_df,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "id",
        "image_path",
        "sha256",
        "baseColour",
        "originalColour",
        "season",
        "usage",
        "productDisplayName",
    ]

    clean_df = clean_df[
        output_columns
    ].copy()

    clean_df["image_path"] = (
        clean_df["image_path"].astype(str)
    )

    clean_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    mapping_df = pd.DataFrame(
        sorted(COLOR_MAP.items()),
        columns=[
            "originalColour",
            "mappedColour",
        ],
    )

    mapping_df.to_csv(
        MAPPING_REPORT,
        index=False,
    )

    if image_errors.empty:
        if IMAGE_ERROR_REPORT.exists():
            IMAGE_ERROR_REPORT.unlink()
    else:
        image_errors.to_csv(
            IMAGE_ERROR_REPORT,
            index=False,
        )

    if conflicting_df.empty:
        if CONFLICT_REPORT.exists():
            CONFLICT_REPORT.unlink()
    else:
        conflicting_df = conflicting_df.copy()

        conflicting_df["image_path"] = (
            conflicting_df["image_path"].astype(str)
        )

        conflicting_df.to_csv(
            CONFLICT_REPORT,
            index=False,
        )


def main():
    df = load_and_filter_metadata()

    valid_df, image_errors = (
        validate_and_hash_images(df)
    )

    print(
        f"\nReadable candidate images: "
        f"{len(valid_df):,}"
    )

    print(
        f"Missing or unreadable images: "
        f"{len(image_errors):,}"
    )

    clean_df, conflicting_df = (
        remove_exact_duplicates(valid_df)
    )

    validate_final_dataset(clean_df)

    save_outputs(
        clean_df,
        image_errors,
        conflicting_df,
    )

    print("\n" + "=" * 70)
    print("CLEAN APPAREL V2 RESULTS")
    print("=" * 70)

    print(
        "Readable rows before deduplication: "
        f"{len(valid_df):,}"
    )

    print(
        "Unique hashes before conflicts:     "
        f"{valid_df['sha256'].nunique():,}"
    )

    conflicting_group_count = (
        conflicting_df["sha256"].nunique()
        if not conflicting_df.empty
        else 0
    )

    print(
        "Conflicting duplicate groups:       "
        f"{conflicting_group_count:,}"
    )

    print(
        f"Final clean rows:                    "
        f"{len(clean_df):,}"
    )

    print_distribution(
        clean_df,
        "Final clean dataset",
    )

    print("\nValidation:")
    print(
        "Duplicate IDs:    "
        f"{clean_df['id'].duplicated().sum():,}"
    )

    print(
        "Duplicate hashes: "
        f"{clean_df['sha256'].duplicated().sum():,}"
    )

    print(
        f"\nSaved cleaned dataset: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Saved colour mapping:  "
        f"{MAPPING_REPORT}"
    )

    if not conflicting_df.empty:
        print(
            f"Saved conflicts:       "
            f"{CONFLICT_REPORT}"
        )

    if not image_errors.empty:
        print(
            f"Saved image errors:    "
            f"{IMAGE_ERROR_REPORT}"
        )


if __name__ == "__main__":
    main()