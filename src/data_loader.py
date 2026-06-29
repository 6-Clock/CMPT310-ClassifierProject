import os
import pandas as pd


LABEL_COLS = ["baseColour", "season", "articleType", "usage"]


def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def check_available_labels(df: pd.DataFrame) -> list[str]:
    """
    Print which of our target label columns exist and show unique values.
    """
    available = []
    for col in LABEL_COLS:
        if col in df.columns:
            n_unique = df[col].nunique()
            n_missing = df[col].isna().sum()
            print(f"  [{col}] found — {n_unique} unique values, {n_missing} missing")
            available.append(col)
        else:
            print(f"  [{col}] NOT found in dataset — will be skipped")
    return available


def filter_and_balance(
    df: pd.DataFrame, label_cols: list[str], min_samples: int = 100
) -> pd.DataFrame:
    """
    Drop NaN rows, remove rare classes iteratively until stable, then
    downsample once on the label with the most classes to cap imbalance.
    """
    present = [c for c in label_cols if c in df.columns]

    # Drop rows missing any label
    filtered = df.dropna(subset=present).copy()
    print(f"  Dropped NaN rows: {len(df)} → {len(filtered)}")

    # Iteratively remove rare classes until no more are found
    changed = True
    while changed:
        changed = False
        for col in present:
            counts = filtered[col].value_counts()
            valid_classes = counts[counts >= min_samples].index
            before = len(filtered)
            filtered = filtered[filtered[col].isin(valid_classes)]
            if len(filtered) != before:
                changed = True

    for col in present:
        n = filtered[col].nunique()
        print(f"  [{col}] {n} classes, {len(filtered)} rows after rare-class removal")

    # Downsample once on the label with the most classes to limit imbalance
    primary = max(present, key=lambda c: filtered[c].nunique())
    min_count = filtered[primary].value_counts().min()
    sampled_idx = (
        filtered.groupby(primary, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), min_count), random_state=42))
        .index
    )
    filtered = filtered.loc[sampled_idx]
    print(f"  Downsampled on [{primary}]: {min_count} samples per class → {len(filtered)} rows total")

    return filtered.reset_index(drop=True)



if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/styles.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}. Download the dataset first.")
        raise SystemExit(1)

    df = load_metadata(csv_path)
    print("\nChecking available labels:")
    available = check_available_labels(df)

    print("\nFiltering and balancing:")
    df_balanced = filter_and_balance(df, available, min_samples=100)

    print("\nDone.")
