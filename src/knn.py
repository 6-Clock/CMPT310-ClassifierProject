from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =========================
# File paths
# =========================

CLEAN_CSV = Path("data/processed/clean_colour_season_style.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "knn_baseline.joblib"


# =========================
# KNN settings
# =========================

IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_NEIGHBORS = 5

LABEL_COLS = ["baseColour", "season", "usage"]

# For adding articleType
# LABEL_COLS = ["baseColour", "season", "articleType", "usage"]


def load_and_flatten_images(df):
    """
    Loads image files from image_path, resizes them, and flattens each image
    into a one-dimensional feature vector for KNN.
    """

    X = []
    valid_indices = []

    for idx, row in df.iterrows():
        image_path = Path(row["image_path"])

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_flat = img_array.flatten()

            X.append(img_flat)
            valid_indices.append(idx)

        except (FileNotFoundError, UnidentifiedImageError, OSError):
            print(f"[WARNING] Skipping unreadable image: {image_path}")

    X = np.array(X, dtype=np.float32)
    valid_df = df.loc[valid_indices].reset_index(drop=True)

    return X, valid_df


def encode_labels(df):
    """
    Encodes text labels into integers because scikit-learn models need
    numeric target values.
    """

    y_encoded_parts = []
    label_encoders = {}

    for col in LABEL_COLS:
        encoder = LabelEncoder()
        encoded_col = encoder.fit_transform(df[col])

        y_encoded_parts.append(encoded_col)
        label_encoders[col] = encoder

    y = np.column_stack(y_encoded_parts)

    return y, label_encoders


def evaluate_model(y_test, y_pred, label_encoders):
    """
    Prints exact-match accuracy and separate accuracy/F1 scores for each label.
    """

    print("\n==============================")
    print("KNN Baseline Results")
    print("==============================")

    exact_match_accuracy = np.mean(np.all(y_test == y_pred, axis=1))
    print(f"\nExact-match accuracy: {exact_match_accuracy:.4f}")

    for i, label in enumerate(LABEL_COLS):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        f1 = f1_score(y_test[:, i], y_pred[:, i], average="weighted")

        print(f"\n--- {label} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1-score: {f1:.4f}")

        class_names = label_encoders[label].classes_

        print("\nClassification report:")
        print(
            classification_report(
                y_test[:, i],
                y_pred[:, i],
                target_names=class_names,
                zero_division=0
            )
        )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEAN_CSV)

    print(f"Initial rows: {len(df)}")

    # Make sure required columns exist
    required_cols = ["image_path"] + LABEL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Remove rows with missing labels or image paths
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    print(f"Rows after dropping missing values: {len(df)}")

    print("Loading and flattening images...")
    X, df = load_and_flatten_images(df)

    print(f"Final image feature matrix shape: {X.shape}")
    print(f"Final label dataframe shape: {df.shape}")

    print("Encoding labels...")
    y, label_encoders = encode_labels(df)

    # Stratify by usage because your data was mainly balanced by usage
    stratify_col = df["usage"]

    print("Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_col
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")

    print("Training KNN baseline...")

    knn = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights="distance",
        metric="minkowski",
        n_jobs=-1
    )

    model = MultiOutputClassifier(knn)
    model.fit(X_train, y_train)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred, label_encoders)

    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(
        {
            "model": model,
            "label_encoders": label_encoders,
            "label_columns": LABEL_COLS,
            "image_size": IMAGE_SIZE
        },
        MODEL_PATH
    )

    print("Done.")


if __name__ == "__main__":
    main()