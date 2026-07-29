"""
Train a multi-task MobileNetV2 that predicts colour, season, and usage
from a single clothing image (Milestone 2).

Big picture (transfer learning):
  - MobileNetV2 was already trained on ImageNet (1.4M photos). It already knows
    generic visual features (edges, textures, shapes). We reuse that "knowledge"
    instead of learning from scratch on our ~4k images.
  - We bolt THREE small classifier "heads" on top of the shared backbone, one
    per label. This is called multi-task learning: one image encoder, several
    predictions.
  - We train in TWO phases:
      Phase 1 (feature extraction): freeze the backbone, train only the heads.
      Phase 2 (fine-tuning): unfreeze the top layers of the backbone and train
        everything at a very low learning rate so we adapt the features to
        clothing without destroying them.

Run from the project root, venv active:
    python src/CNN/train_cnn.py
"""

"""
Description:

"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# =========================================================================
# Paths and constants
# =========================================================================

TRAIN_CSV = Path("data/processed/train.csv")
VAL_CSV = Path("data/processed/val.csv")

# We load the FULL-RESOLUTION images (not the 96x96 copies) because MobileNetV2
# expects 224x224 and upscaling tiny images loses detail. The split CSVs have an
# `id` column, so we rebuild the raw path from it.
RAW_IMAGE_DIR = Path("data/raw/images")

MODEL_DIR = Path("models/CNN")
MODEL_PATH = MODEL_DIR / "mobilenet_multitask.keras"
ENCODERS_PATH = MODEL_DIR / "label_encoders.joblib"
HISTORY_PATH = MODEL_DIR / "history.json"

# The three labels we predict. Left = column in the CSV, right = the name we
# give that output "head" in the model.
LABEL_COLS = ["baseColour", "season", "usage"]
HEAD_NAMES = {"baseColour": "color", "season": "season", "usage": "usage"}

# ----- Hyperparameters -------------------------------------------------------
IMG_SIZE = 224          # MobileNetV2's expected input size
BATCH_SIZE = 32         # Number of images the model process at a time
DROPOUT = 0.3           # regularization: randomly drops units to reduce overfitting

EPOCHS_PHASE1 = 8       # feature-extraction epochs (backbone frozen)
EPOCHS_PHASE2 = 10      # fine-tuning epochs (top of backbone unfrozen)

LR_PHASE1 = 1e-3        # normal learning rate: only the fresh heads are training
LR_PHASE2 = 1e-5        # TINY learning rate: we are nudging pretrained weights, not rewriting them

UNFREEZE_LAST_N = 40    #Unfreeze the N last layers (Stage 2)
#UNFREEZE_LAST_N the lower the value -> lower risk of overfitting, more layer will be trained based on our data

RANDOM_STATE = 42 #Random Seed, we set it just make the random operations reproducible


# =========================================================================
# 1. Labels: fit one LabelEncoder per column
# =========================================================================

def fit_label_encoders(df):
    """
    Turn text labels ("Blue", "Summer", ...) into integers (0, 1, 2, ...),
    because the network outputs numbers, not words. We keep the encoders so we
    can translate predictions back to text later (in evaluate + the demo).
    """
    encoders = {}
    for col in LABEL_COLS:
        enc = LabelEncoder()
        enc.fit(df[col])
        encoders[col] = enc
    return encoders


# =========================================================================
# 2. Class weights (to fight imbalance, e.g. Spring/Winter are rare)
# =========================================================================

def compute_all_class_weights(df, encoders):
    """
    Returns, per label, a dict {class_index: weight}.
    Rare classes get a HIGHER weight so the model is penalized more for getting them wrong and can't just
    ignore them.
    """
    class_weights = {}
    for col in LABEL_COLS:
        y = encoders[col].transform(df[col])
        classes = np.arange(len(encoders[col].classes_))

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y,
        )

        class_weights[col] = {int(c): float(w) for c, w in zip(classes, weights)}
    return class_weights


# =========================================================================
# 3. Data pipeline (tf.data) Making more varied of data
# =========================================================================

# Light augmentation, applied to TRAINING images only. It shows the model
# slightly varied versions of each image (flipped, rotated, zoomed) so it
# generalizes instead of memorizing. (We do NOT augment val/test.)

# Augmentation like rotate and zoom only applied when training
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.2),
    ],
    name="data_augmentation",
)


def _load_image(path):
    """Read a JPEG file from disk and return a float32 [0, 255] tensor of shape
    (IMG_SIZE, IMG_SIZE, 3). We do NOT scale to [0,1] here because MobileNetV2's
    preprocess_input expects raw 0-255 values and maps them to [-1, 1] itself."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img


def make_dataset(df, encoders, class_weights=None, training=False):
    """
    Build a tf.data pipeline that yields batches of:
        (image, {"color": y, "season": y, "usage": y})           if no weights
        (image, {...labels...}, {...per-sample weights...})       if class_weights given

    The per-output weights dict is how we sneak class weighting into a
    multi-output model (see compute_all_class_weights).
    """
    # Rebuild raw image paths from the id column.
    paths = [str(RAW_IMAGE_DIR / f"{int(i)}.jpg") for i in df["id"].values]

    # Integer-encode each label column into a dict keyed by head name.
    # Eg. baseColour = ["Blue", "Black", "Blue", "Red"]  ----- encoders.transform ----> "color" : array([0, 1, 0, 2], dtype = int32)
    # For each attribute color, season, usage
    labels = {
        HEAD_NAMES[col]: encoders[col].transform(df[col]).astype("int32")
        for col in LABEL_COLS
    }
    # Class weigth used to blance the dataset example Spring has 76 and Winter has 2067 images
    # gradient descent will happily learn "always guess Summer" because that minimizes loss on average.
    # Weighting up the rare classes during training forces the model to actually pay attention to them while it's adjusting its weights.


    # When we are valiadating we do not use the weights, so the weight does not affect our predictions
    # This will simulate a more natural prediction
    # Eg. choosing a random clothing photo with no additional information --> predict his attributes
    if class_weights is not None:
        # For each sample, look up the weight of its true class, per head.
        weights = {
            HEAD_NAMES[col]: np.array(
                [class_weights[col][int(c)] for c in labels[HEAD_NAMES[col]]],
                dtype="float32",
            )
            for col in LABEL_COLS
        }
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_STATE)

    def _map(path, y, w=None):
        img = _load_image(path)
        if training:
            # Augmentation like rotate and zoom only applied when training
            img = data_augmentation(img)
        img = preprocess_input(img)          # MobileNetV2-specific scaling to [-1, 1]
        if w is None:
            return img, y
        return img, y, w

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================================================================
# 4. Model: shared backbone + three heads
# =========================================================================

def build_model(encoders):
    """
    Returns (model, base_model). `base_model` is handed back so we can unfreeze
    part of it in phase 2.
    """
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,        # drop ImageNet's 1000-class classifier; we add our own
        weights="imagenet",       # <- the pretrained knowledge we're reusing ImageNet
    )
    base_model.trainable = False  # Phase 1: freeze the whole backbone

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)) # shape = (255, 255, 3)
    x = base_model(inputs, training=False)     # training=False keeps BatchNorm stats frozen
    x = layers.GlobalAveragePooling2D()(x)     # (H, W, C) feature map -> one vector per image
    x = layers.Dropout(DROPOUT)(x)

    # One softmax head per label. `n_classes` differs per label
    # (colour=10, season=4, usage=4). The `name=` MUST match the keys we used
    # for the labels dict in make_dataset ("color"/"season"/"usage").
    n_color = len(encoders["baseColour"].classes_)
    color_out = layers.Dense(n_color, activation="softmax", name="color")(x)

    n_season = len(encoders["season"].classes_)
    season_out = layers.Dense(n_season, activation="softmax", name="season")(x)

    n_usage = len(encoders["usage"].classes_)
    usage_out = layers.Dense(n_usage, activation="softmax", name="usage")(x)

    model = keras.Model(inputs=inputs, outputs={
        "color": color_out,
        "season": season_out,
        "usage": usage_out,
    })
    return model, base_model


def compile_model(model, learning_rate):
    """
    Each head is a normal single-label classifier, so each uses
    sparse categorical cross-entropy (sparse = our labels are integers, not
    one-hot). Keras sums the three losses into the total training loss.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "color": "sparse_categorical_crossentropy",
            "season": "sparse_categorical_crossentropy",
            "usage": "sparse_categorical_crossentropy",
        },
        metrics={
            "color": "accuracy",
            "season": "accuracy",
            "usage": "accuracy",
        },
    )


# =========================================================================
# 5. Training driver
# =========================================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading split CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    # Fit encoders on TRAIN ONLY so we never peek at val/test label sets.
    encoders = fit_label_encoders(train_df)
    class_weights = compute_all_class_weights(train_df, encoders)

    train_ds = make_dataset(train_df, encoders, class_weights, training=True)
    val_ds = make_dataset(val_df, encoders, class_weights=None, training=False)

    model, base_model = build_model(encoders)
    model.summary()

    # Callbacks: stop early if val loss stops improving, and drop the learning
    # rate when progress stalls. `restore_best_weights` keeps the best model.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    # ---- Phase 1: feature extraction (backbone frozen) ----
    print("\n=== Phase 1: training the heads (backbone frozen) ===")
    compile_model(model, LR_PHASE1)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks,
    )

    # ---- Phase 2: fine-tuning (unfreeze the top of the backbone) ----
    print("\n=== Phase 2: fine-tuning the top of MobileNetV2 ===")
    base_model.trainable = True
    # Freeze everything EXCEPT the last UNFREEZE_LAST_N layers. The early layers
    # detect generic features we want to keep; the late layers are the ones
    # worth specializing to clothing.

    #Unfreeze the last (UNFREEZE_LAST_N) Layers
    for layer in base_model.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

    # IMPORTANT: recompile after changing `trainable`, using the tiny LR.
    compile_model(model, LR_PHASE2)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks,
    )

    # ---- Save everything the evaluate script + demo will need ----
    print(f"\nSaving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    # Stitch the two phases' histories together so we can plot one curve.
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history.get(key, [])
    with open(HISTORY_PATH, "w") as f:
        json.dump(combined, f)

    print("Done. Next: python src/CNN/evaluate_cnn.py")


if __name__ == "__main__":
    main()
