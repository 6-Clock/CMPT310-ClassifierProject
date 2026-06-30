from pathlib import Path
import pandas as pd

RAW_CSV = Path("data/raw/styles.csv")
IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "clean_colour_season_style.csv"

SELECTED_COLORS = [
    "Black",
    "White",
    "Blue",
    "Brown",
    "Grey",
    "Red",
    "Green",
    "Pink",
    "Navy Blue",
    "Purple",
]

SELECTED_USAGE = [
    "Casual",
    "Sports",
    "Ethnic",
    "Formal",
]

SELECTED_SEASONS = [
    "Summer",
    "Fall",
    "Winter",
    "Spring",
]

SAMPLES_PER_USAGE = 1500
MIN_SAMPLES_PER_SEASON = 100

RANDOM_STATE = 42


def print_distribution(df, title):
    print(f"\n===== {title} =====")

    print("\nUsage distribution:")
    print(df["usage"].value_counts())

    print("\nSeason distribution:")
    print(df["season"].value_counts())

    print("\nColour distribution:")
    print(df["baseColour"].value_counts())

    print("\nUsage x Season:")
    print(pd.crosstab(df["usage"], df["season"]))


def main():
    df = pd.read_csv(RAW_CSV, on_bad_lines="skip")

    needed_cols = [
        "id",
        "masterCategory",
        "baseColour",
        "season",
        "usage",
        "productDisplayName",
    ]

    df = df[needed_cols].copy()

    # Remove missing labels
    df = df.dropna(
        subset=[
            "id",
            "masterCategory",
            "baseColour",
            "season",
            "usage",
        ]
    )

    # Keep only Apparel, based on project scope
    df = df[df["masterCategory"] == "Apparel"].copy()

    # Keep reliable labels only
    df = df[
        df["baseColour"].isin(SELECTED_COLORS)
        & df["season"].isin(SELECTED_SEASONS)
        & df["usage"].isin(SELECTED_USAGE)
    ].copy()

    # Add image path
    df["image_path"] = df["id"].apply(lambda x: IMAGE_DIR / f"{int(x)}.jpg")

    # Remove rows where image file does not exist
    df = df[df["image_path"].apply(lambda p: p.exists())].copy()

    # Convert path to string for CSV
    df["image_path"] = df["image_path"].astype(str)

    print_distribution(df, "Before balancing")

    # Balance mainly by usage, not full colour-season-usage combinations.
    # Full combinations can become too sparse.
    balanced_parts = []

    for usage, usage_group in df.groupby("usage"):
        n = min(SAMPLES_PER_USAGE, len(usage_group))

        sampled = usage_group.sample(
            n=n,
            random_state=RANDOM_STATE,
        )

        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts).reset_index(drop=True)

    # Light season cleanup:
    # If a season has extremely low samples after usage balancing,
    # add more from the filtered dataset when possible.
    season_parts = [balanced_df]

    for season in SELECTED_SEASONS:
        current_count = len(balanced_df[balanced_df["season"] == season])

        if current_count >= MIN_SAMPLES_PER_SEASON:
            continue

        needed = MIN_SAMPLES_PER_SEASON - current_count

        candidates = df[
            (df["season"] == season)
            & (~df["id"].isin(balanced_df["id"]))
        ]

        if len(candidates) == 0:
            print(f"[INFO] No extra candidates available for season={season}")
            continue

        extra = candidates.sample(
            n=min(needed, len(candidates)),
            random_state=RANDOM_STATE,
        )

        if len(extra) < needed:
            print(f"[INFO] Could only add {len(extra)}/{needed} extra samples for season={season}")

        season_parts.append(extra)

    balanced_df = (
        pd.concat(season_parts)
        .drop_duplicates(subset=["id"])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print_distribution(balanced_df, "After balancing")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved cleaned dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(balanced_df)}")

if __name__ == "__main__":
    main()