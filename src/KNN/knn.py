from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =========================
# File paths
# =========================

# Train/evaluate on the SAME split the CNN uses (src/preprocess_split.py),
# instead of KNN's own separate 80/20 split. That old approach let KNN's own
# training set overlap with the CNN's test set by ~80% (both were carved out
# of the same pool independently), making the KNN-vs-CNN comparison in
# evaluate_cnn.py invalid. Using the identical split makes it a fair, honest
# comparison with zero overlap.
TRAIN_CSV = Path("data/processed/train_v2.csv")
TEST_CSV = Path("data/processed/test_v2.csv")
MODEL_DIR = Path("models/KNN")
MODEL_PATH = MODEL_DIR / "knn_baseline.joblib"


# =========================
# KNN settings
# =========================

IMAGE_SIZE = (96, 96)
N_NEIGHBORS = 5

LABEL_COLS = ["baseColour", "season", "usage"]

# For adding articleType
# LABEL_COLS = ["baseColour", "season", "articleType", "usage"]


def load_and_flatten_images(df):
    """
    Loads image files from image_path, resizes them, and flattens each image
    into a one-dimensional feature vector for KNN.

    image_path already points at the pre-resized 96x96 images (images_96_v2/,
    aspect-ratio-preserving padding from preprocess_split.py), so .resize()
    here is a no-op in practice -- kept as a safety net.
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


def fit_label_encoders(df):
    """
    Fits one LabelEncoder per label column on the TRAINING data only, so we
    never let the test set's label distribution leak into encoding.
    """
    label_encoders = {}
    for col in LABEL_COLS:
        encoder = LabelEncoder()
        encoder.fit(df[col])
        label_encoders[col] = encoder
    return label_encoders


def encode_labels(df, label_encoders):
    """Encodes text labels into integers using already-fitted encoders."""
    y_encoded_parts = [label_encoders[col].transform(df[col]) for col in LABEL_COLS]
    return np.column_stack(y_encoded_parts)


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


def load_split(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["image_path"] + LABEL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

    return df.dropna(subset=required_cols).reset_index(drop=True)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train/test splits...")
    train_df = load_split(TRAIN_CSV)
    test_df = load_split(TEST_CSV)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    print("Loading and flattening training images...")
    X_train, train_df = load_and_flatten_images(train_df)

    print("Loading and flattening test images...")
    X_test, test_df = load_and_flatten_images(test_df)

    print(f"Train feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")

    print("Encoding labels...")
    label_encoders = fit_label_encoders(train_df)
    y_train = encode_labels(train_df, label_encoders)
    y_test = encode_labels(test_df, label_encoders)

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
