"""
Evaluate the trained multi-task CNN on the held-out TEST set (Milestone 2).

Produces everything the report needs:
  - accuracy + macro & micro F1 per label
  - per-class precision/recall (classification report) per label
  - one confusion matrix per label  -> saved as PNG in reports/
  - training curves (loss + accuracy)  -> saved as PNG in reports/

Run from the project root, venv active, AFTER train_cnn.py:
    python src/CNN/evaluate_cnn.py
"""

from pathlib import Path
import sys
import json

# Put the project root on the import path so `from src.CNN...` works no matter
# what folder you launch the script from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Reuse the exact same pipeline + constants the model was trained with, so
# test images are preprocessed identically. (No duplicated logic = no drift.)
from src.CNN.train_cnn import (
    make_dataset,
    LABEL_COLS,
    HEAD_NAMES,
    MODEL_PATH,
    ENCODERS_PATH,
    HISTORY_PATH,
)

TEST_CSV = Path("data/processed/test.csv")
REPORTS_DIR = Path("reports")

# The saved KNN baseline (src/KNN/knn.py), used for the comparison table.
KNN_MODEL_PATH = Path("models/KNN/knn_baseline.joblib")


# =========================================================================
# 1. Run the model on the test set and collect predictions
# =========================================================================

def get_predictions(model, test_df, encoders):
    """
    Returns two dicts keyed by column name:
        y_true[col] = integer ground-truth labels
        y_pred[col] = integer predicted labels (argmax of the softmax head)
    """
    test_ds = make_dataset(test_df, encoders, class_weights=None, training=False)

    # model.predict returns a dict: {"color": probs, "season": probs, "usage": probs}
    # where each `probs` is (num_images, num_classes). argmax picks the most
    # likely class per image.
    preds = model.predict(test_ds)

    y_true, y_pred = {}, {}
    for col in LABEL_COLS:
        head = HEAD_NAMES[col]
        # preds[head] has shape (num_images, num_classes) — one probability
        # distribution per image. argmax along axis=1 picks the class index
        # with the highest probability for each image.
        y_pred[col] = np.argmax(preds[head], axis=1)
        y_true[col] = encoders[col].transform(test_df[col])
    return y_true, y_pred


# =========================================================================
# 2. Metrics per label
# =========================================================================

def report_metrics(y_true, y_pred, encoders):
    """
    Prints the full per-label report AND returns a compact dict:
        {col: {"accuracy": ..., "f1_macro": ..., "f1_micro": ...}}
    The returned dict is reused by the KNN-vs-CNN comparison table so we don't
    recompute the same numbers twice.
    """
    metrics = {}
    for col in LABEL_COLS:
        names = encoders[col].classes_
        acc = accuracy_score(y_true[col], y_pred[col])

        # Macro F1 = unweighted average over classes -> rare classes count as
        # much as common ones (this is what exposes weak Spring/Winter).
        # Micro F1 = pools all predictions -> dominated by common classes.
        f1_macro = f1_score(y_true[col], y_pred[col], average="macro")
        f1_micro = f1_score(y_true[col], y_pred[col], average="micro")

        metrics[col] = {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}

        print(f"\n================ {col} ================")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1 macro : {f1_macro:.4f}")
        print(f"F1 micro : {f1_micro:.4f}")
        print("\nPer-class precision / recall / F1:")
        print(classification_report(
            y_true[col], y_pred[col], target_names=names, zero_division=0
        ))

    return metrics


# =========================================================================
# 3. Confusion matrix per label
# =========================================================================

def plot_confusion_matrices(y_true, y_pred, encoders):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for col in LABEL_COLS:
        names = encoders[col].classes_
        cm = confusion_matrix(y_true[col], y_pred[col])

        disp = ConfusionMatrixDisplay(cm, display_labels=names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"Confusion matrix — {col}")
        fig.tight_layout()

        out = REPORTS_DIR / f"confusion_{col}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


# =========================================================================
# 4. Training curves (from the saved history.json)
# =========================================================================

def plot_training_curves():
    if not HISTORY_PATH.exists():
        print(f"No history file at {HISTORY_PATH}, skipping curves.")
        return

    with open(HISTORY_PATH) as f:
        h = json.load(f)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(h.get("loss", []), label="train")
    ax1.plot(h.get("val_loss", []), label="val")
    ax1.set_title("Total loss")
    ax1.set_xlabel("epoch")
    ax1.legend()

    # Keras names each head's accuracy metric "<head>_accuracy" (and prefixes
    # "val_" for the validation version) since we passed
    # metrics={"color": "accuracy", ...} in compile_model. We plot the color
    # head here; swap the key to "season_accuracy"/"usage_accuracy" to see
    # the others.
    ax2.plot(h.get("color_accuracy", []), label="train")
    ax2.plot(h.get("val_color_accuracy", []), label="val")
    ax2.set_title("Color head accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()

    fig.tight_layout()
    out = REPORTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# =========================================================================
# 5. KNN baseline vs CNN comparison table
# =========================================================================

def evaluate_knn_baseline(test_df):
    """
    Loads the saved KNN model (src/KNN/knn.py) and scores it on the SAME test
    rows the CNN was just evaluated on, so the comparison table is apples-to-
    apples on identical images.

    CAVEAT: the KNN model was originally trained on its own 80/20 split of
    `clean_colour_season_style.csv`, not on the CNN's train/val/test split.
    Some of these "test" images may have been in the KNN's own training set,
    which can make the KNN score look slightly better than true generalization
    performance. Worth a one-line mention in the report.
    """
    if not KNN_MODEL_PATH.exists():
        print(f"No KNN model found at {KNN_MODEL_PATH}, skipping comparison. "
              f"Run src/KNN/knn.py first.")
        return None

    bundle = joblib.load(KNN_MODEL_PATH)
    knn_model = bundle["model"]
    knn_encoders = bundle["label_encoders"]
    knn_cols = bundle["label_columns"]          # ["baseColour", "season", "usage"]
    image_size = bundle["image_size"]           # (96, 96)

    # test.csv's image_path already points at the pre-resized 96x96 JPEGs, so
    # this is the exact same flatten-to-vector step knn.py used at train time.
    X = []
    for path in test_df["image_path"]:
        img = Image.open(path).convert("RGB").resize(image_size)
        X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
    X = np.array(X, dtype=np.float32)

    y_pred = knn_model.predict(X)   # shape (n_samples, len(knn_cols))

    metrics = {}
    for i, col in enumerate(knn_cols):
        y_true = knn_encoders[col].transform(test_df[col])
        y_pred_col = y_pred[:, i]

        metrics[col] = {
            "accuracy": accuracy_score(y_true, y_pred_col),
            "f1_macro": f1_score(y_true, y_pred_col, average="macro"),
            "f1_micro": f1_score(y_true, y_pred_col, average="micro"),
        }
    return metrics


def print_comparison_table(cnn_metrics, knn_metrics):
    """Side-by-side table showing the payoff of fine-tuning over the baseline."""
    if knn_metrics is None:
        return

    print("\n================ KNN baseline vs CNN (fine-tuned) ================")
    header = f"{'label':<12}{'KNN acc':>10}{'CNN acc':>10}{'KNN F1-macro':>14}{'CNN F1-macro':>14}"
    print(header)
    print("-" * len(header))

    for col in LABEL_COLS:
        knn = knn_metrics[col]
        cnn = cnn_metrics[col]
        print(
            f"{col:<12}"
            f"{knn['accuracy']:>10.4f}"
            f"{cnn['accuracy']:>10.4f}"
            f"{knn['f1_macro']:>14.4f}"
            f"{cnn['f1_macro']:>14.4f}"
        )


def main():
    print("Loading model + encoders...")
    model = keras.models.load_model(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    test_df = pd.read_csv(TEST_CSV)
    print(f"Test rows: {len(test_df)}")

    y_true, y_pred = get_predictions(model, test_df, encoders)

    cnn_metrics = report_metrics(y_true, y_pred, encoders)
    plot_confusion_matrices(y_true, y_pred, encoders)
    plot_training_curves()

    knn_metrics = evaluate_knn_baseline(test_df)
    print_comparison_table(cnn_metrics, knn_metrics)

    print("\nDone. Figures are in the reports/ folder.")


if __name__ == "__main__":
    main()
