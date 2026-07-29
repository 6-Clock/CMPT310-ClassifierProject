from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

INPUT_CSV = Path("data/processed/clean_apparel_v2.csv")

PROCESSED_DIR = Path("data/processed")
PROCESSED_IMAGE_DIR = PROCESSED_DIR / "images_96_v2"

METADATA_OUTPUT = PROCESSED_DIR / "metadata_v2.csv"
TRAIN_OUTPUT = PROCESSED_DIR / "train_v2.csv"
VAL_OUTPUT = PROCESSED_DIR / "val_v2.csv"
TEST_OUTPUT = PROCESSED_DIR / "test_v2.csv"

REPORT_DIR = Path("reports")
DISTRIBUTION_REPORT = REPORT_DIR / "v2_split_distributions.csv"


# -------------------------------------------------------------------
# Preprocessing and split settings
# -------------------------------------------------------------------

TARGET_COLUMNS = [
    "baseColour",
    "season",
    "usage",
]

IMG_SIZE = 96

TEST_AND_VALIDATION_SIZE = 0.30
TEST_SHARE_OF_TEMP = 0.50

RANDOM_STATE = 42

# A common label combination must have at least this many rows
# to be used directly as a stratification group.
#
# Smaller combinations are pooled for splitting only.
# No dataset rows are removed by this process.
MIN_STRATUM_SIZE = 50


def load_clean_dataset():
    """Load and validate the cleaned V2 master dataset."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned dataset was not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "id",
        "image_path",
        "sha256",
        "baseColour",
        "originalColour",
        "season",
        "usage",
        "productDisplayName",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{missing_columns}"
        )

    df = df[required_columns].copy()

    df["id"] = pd.to_numeric(
        df["id"],
        errors="raise",
    ).astype(int)

    if df["id"].duplicated().any():
        duplicate_count = df["id"].duplicated().sum()

        raise ValueError(
            f"Input dataset contains {duplicate_count} duplicate IDs."
        )

    if df["sha256"].duplicated().any():
        duplicate_count = df["sha256"].duplicated().sum()

        raise ValueError(
            "Input dataset contains "
            f"{duplicate_count} duplicate image hashes."
        )

    missing_target_values = (
        df[TARGET_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    if missing_target_values:
        raise ValueError(
            "Input dataset contains missing target values."
        )

    # Preserve the path to the original full-resolution image.
    df["raw_image_path"] = df["image_path"].astype(str)

    missing_images = df[
        ~df["raw_image_path"].apply(
            lambda path: Path(path).exists()
        )
    ]

    if not missing_images.empty:
        raise FileNotFoundError(
            f"{len(missing_images)} raw image files are missing."
        )

    print("=" * 70)
    print("CLEAN DATASET LOADED")
    print("=" * 70)
    print(f"Rows:             {len(df):,}")
    print(f"Unique IDs:       {df['id'].nunique():,}")
    print(f"Unique hashes:    {df['sha256'].nunique():,}")
    print(f"Colour classes:   {df['baseColour'].nunique():,}")
    print(f"Season classes:   {df['season'].nunique():,}")
    print(f"Usage classes:    {df['usage'].nunique():,}")

    return df


def resize_with_padding(
    input_path,
    output_path,
    size=IMG_SIZE,
):
    """Resize an image while preserving its aspect ratio."""

    with Image.open(input_path) as image:
        image = image.convert("RGB")

        processed_image = ImageOps.pad(
            image,
            size=(size, size),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )

        processed_image.save(
            output_path,
            format="JPEG",
            quality=95,
        )


def create_processed_images(df):
    """Create the 96x96 image copies used by the KNN model."""

    PROCESSED_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_paths = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Creating V2 96x96 images",
    ):
        input_path = Path(row.raw_image_path)

        output_path = (
            PROCESSED_IMAGE_DIR
            / f"{row.id}.jpg"
        )

        if not output_path.exists():
            resize_with_padding(
                input_path=input_path,
                output_path=output_path,
            )

        processed_paths.append(str(output_path))

    processed_df = df.copy()

    # image_path is the KNN-ready 96x96 image.
    # raw_image_path remains available for CNN training.
    processed_df["image_path"] = processed_paths

    return processed_df


def create_stratification_key(df):
    """
    Build a stratification key using colour, season and usage.

    Common combinations are stratified directly. Rare combinations
    are pooled only for the purpose of creating stable splits.
    No rows are deleted.
    """

    full_key = (
        df["baseColour"].astype(str)
        + "|"
        + df["season"].astype(str)
        + "|"
        + df["usage"].astype(str)
    )

    full_key_counts = full_key.value_counts()

    # Keep common colour-season-usage combinations.
    # Pool uncommon colours within their season-usage group.
    pooled_key = full_key.where(
        full_key.map(full_key_counts)
        >= MIN_STRATUM_SIZE,
        (
            df["season"].astype(str)
            + "|"
            + df["usage"].astype(str)
            + "|OTHER_COLOUR"
        ),
    )

    pooled_key_counts = pooled_key.value_counts()

    # A few season-usage pools may still be very small.
    # Combine those into one final rare-combination group.
    final_key = pooled_key.where(
        pooled_key.map(pooled_key_counts)
        >= MIN_STRATUM_SIZE,
        "RARE_COMBINATIONS",
    )

    final_counts = final_key.value_counts()

    if final_counts.min() < 4:
        raise ValueError(
            "At least one stratification group is too small "
            "for a 70/15/15 split."
        )

    print("\n" + "=" * 70)
    print("STRATIFICATION SUMMARY")
    print("=" * 70)
    print(
        "Original colour-season-usage combinations: "
        f"{full_key.nunique():,}"
    )
    print(
        "Final stratification groups:               "
        f"{final_key.nunique():,}"
    )
    print(
        "Smallest final stratification group:       "
        f"{final_counts.min():,}"
    )

    if "RARE_COMBINATIONS" in final_counts:
        print(
            "Rows pooled as rare combinations:         "
            f"{final_counts['RARE_COMBINATIONS']:,}"
        )

    return final_key


def split_dataset(df):
    """Create reproducible 70/15/15 splits."""

    split_df = df.copy()

    split_df["_stratify_key"] = (
        create_stratification_key(split_df)
    )

    train_df, temp_df = train_test_split(
        split_df,
        test_size=TEST_AND_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=split_df["_stratify_key"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_SHARE_OF_TEMP,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["_stratify_key"],
    )

    train_df = train_df.drop(
        columns=["_stratify_key"]
    ).copy()

    val_df = val_df.drop(
        columns=["_stratify_key"]
    ).copy()

    test_df = test_df.drop(
        columns=["_stratify_key"]
    ).copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df = (
        train_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df


def validate_splits(
    full_df,
    train_df,
    val_df,
    test_df,
):
    """Check split sizes, classes and data leakage."""

    total_split_rows = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    if total_split_rows != len(full_df):
        raise ValueError(
            "Split sizes do not add up to the master dataset."
        )

    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    if train_ids & val_ids:
        raise ValueError(
            "ID leakage exists between train and validation."
        )

    if train_ids & test_ids:
        raise ValueError(
            "ID leakage exists between train and test."
        )

    if val_ids & test_ids:
        raise ValueError(
            "ID leakage exists between validation and test."
        )

    train_hashes = set(train_df["sha256"])
    val_hashes = set(val_df["sha256"])
    test_hashes = set(test_df["sha256"])

    if train_hashes & val_hashes:
        raise ValueError(
            "Image leakage exists between train and validation."
        )

    if train_hashes & test_hashes:
        raise ValueError(
            "Image leakage exists between train and test."
        )

    if val_hashes & test_hashes:
        raise ValueError(
            "Image leakage exists between validation and test."
        )

    for column in TARGET_COLUMNS:
        expected_classes = set(full_df[column])

        for split_name, split_data in [
            ("train", train_df),
            ("validation", val_df),
            ("test", test_df),
        ]:
            split_classes = set(split_data[column])

            missing_classes = (
                expected_classes - split_classes
            )

            if missing_classes:
                raise ValueError(
                    f"{split_name} is missing classes in "
                    f"{column}: {sorted(missing_classes)}"
                )

    print("\n" + "=" * 70)
    print("SPLIT VALIDATION")
    print("=" * 70)
    print(f"Master rows:       {len(full_df):,}")
    print(f"Training rows:     {len(train_df):,}")
    print(f"Validation rows:   {len(val_df):,}")
    print(f"Testing rows:      {len(test_df):,}")
    print("Cross-split ID overlap:    0")
    print("Cross-split hash overlap:  0")
    print("Missing target classes:    0")


def build_distribution_report(split_datasets):
    """Create one report containing all split distributions."""

    report_rows = []

    for split_name, split_df in split_datasets.items():
        for target_column in TARGET_COLUMNS:
            counts = (
                split_df[target_column]
                .value_counts()
                .sort_index()
            )

            percentages = (
                split_df[target_column]
                .value_counts(normalize=True)
                .mul(100)
                .sort_index()
            )

            for class_name in counts.index:
                report_rows.append(
                    {
                        "split": split_name,
                        "target": target_column,
                        "class": class_name,
                        "count": int(
                            counts.loc[class_name]
                        ),
                        "percentage": float(
                            percentages.loc[class_name]
                        ),
                    }
                )

    return pd.DataFrame(report_rows)


def print_split_distributions(
    train_df,
    val_df,
    test_df,
):
    """Print target distributions for each split."""

    for split_name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        print("\n" + "=" * 70)
        print(f"{split_name} DISTRIBUTIONS")
        print("=" * 70)

        for target_column in TARGET_COLUMNS:
            print(f"\n{target_column}:")
            print(
                split_df[target_column]
                .value_counts()
                .to_string()
            )


def save_outputs(
    train_df,
    val_df,
    test_df,
):
    """Save the split CSVs and reports."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )

    metadata_df = (
        metadata_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    output_columns = [
        "id",
        "image_path",
        "raw_image_path",
        "sha256",
        "baseColour",
        "originalColour",
        "season",
        "usage",
        "productDisplayName",
        "split",
    ]

    metadata_df[output_columns].to_csv(
        METADATA_OUTPUT,
        index=False,
    )

    train_df[output_columns].to_csv(
        TRAIN_OUTPUT,
        index=False,
    )

    val_df[output_columns].to_csv(
        VAL_OUTPUT,
        index=False,
    )

    test_df[output_columns].to_csv(
        TEST_OUTPUT,
        index=False,
    )

    distribution_report = build_distribution_report(
        {
            "full": metadata_df,
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
    )

    distribution_report.to_csv(
        DISTRIBUTION_REPORT,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"Metadata:       {METADATA_OUTPUT}")
    print(f"Training split: {TRAIN_OUTPUT}")
    print(f"Validation:     {VAL_OUTPUT}")
    print(f"Testing split:  {TEST_OUTPUT}")
    print(f"96x96 images:   {PROCESSED_IMAGE_DIR}")
    print(f"Report:         {DISTRIBUTION_REPORT}")


def main():
    clean_df = load_clean_dataset()

    processed_df = create_processed_images(
        clean_df
    )

    train_df, val_df, test_df = split_dataset(
        processed_df
    )

    validate_splits(
        full_df=processed_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    print_split_distributions(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()