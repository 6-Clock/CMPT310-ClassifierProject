"""
Train a multi-task ResNet50 that predicts colour, season, and usage
from a single clothing image (Milestone 2).

Big picture (transfer learning):
  - ResNet50 was already trained on ImageNet (1.4M photos). It already knows
    generic visual features (edges, textures, shapes). We reuse that "knowledge"
    instead of learning from scratch on our ~14k images.
  - ResNet50 is a much DEEPER network than MobileNetV2 (175 layers vs 154),
    built from "residual blocks" that let gradients skip past layers during
    training. That extra depth/capacity is why we're switching to it after
    MobileNetV2 underperformed on colour and season.
  - We bolt THREE small classifier "heads" on top of the shared backbone, one
    per label. This is called multi-task learning: one image encoder, several
    predictions.
  - We train in up to FOUR phases:
      Phase 1 (feature extraction): freeze the backbone, train only the heads.
      Phase 2 (fine-tuning): unfreeze the top layers of the backbone and train
        everything at a very low learning rate so we adapt the features to
        clothing without destroying them.
      Phase 3 (optional colour-only polish): if an auxiliary colour-only
        dataset is present (see prepare_color_aux_dataset.py), fine-tune ONLY
        the colour head on it, with the backbone and the season/usage heads
        completely frozen. We tried training on this data jointly during
        Phase 1/2 first -- it regressed season by 0.036 macro F1, because
        gradients from the auxiliary rows' colour loss still flow through the
        SHARED backbone even when season/usage loss is masked to 0 for those
        rows. Freezing everything except the colour head in a separate phase
        makes that impossible: there is no path left for the auxiliary data
        to reach season/usage at all.
      Phase 4 (optional season+usage-only polish): mirrors Phase 3, but for
        an auxiliary season/usage-labelled dataset (see
        prepare_style_aux_dataset.py -- AI-generated images, used only to
        nudge season/usage; that dataset's images are NOT real photographs,
        so we isolate their influence to just these two heads the same way).
        Backbone and colour head frozen; only season/usage are trainable.

Run from the project root, venv active:
    python src/CNN/train_cnn.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# =========================================================================
# Paths and constants
# =========================================================================

# V2 dataset: deduplicated by image hash, colours consolidated (e.g. Navy
# Blue -> Blue), no artificial per-usage downsampling. See
# src/balance_clean_dataset.py and src/preprocess_split.py.
TRAIN_CSV = Path("data/processed/train_v2.csv")
VAL_CSV = Path("data/processed/val_v2.csv")

# Optional colour-only auxiliary data (src/CNN/prepare_color_aux_dataset.py).
# Has no season/usage ground truth, so those rows only ever train the colour
# head -- see load_color_aux_data() and the has_{col} masking in make_dataset().
COLOR_AUX_CSV = Path("data/processed/color_aux_v1.csv")

# Optional season+usage-only auxiliary data (AI-generated images, see
# prepare_style_aux_dataset.py). Has no colour ground truth -- these rows
# only ever train the season/usage heads. See load_style_aux_data().
STYLE_AUX_CSV = Path("data/processed/style_aux_v1.csv")

MODEL_DIR = Path("models/CNN")
MODEL_PATH = MODEL_DIR / "resnet50_multitask.keras"
ENCODERS_PATH = MODEL_DIR / "label_encoders.joblib"
HISTORY_PATH = MODEL_DIR / "history.json"

# The three labels we predict. Left = column in the CSV, right = the name we
# give that output "head" in the model.
LABEL_COLS = ["baseColour", "season", "usage"]
HEAD_NAMES = {"baseColour": "color", "season": "season", "usage": "usage"}

# ----- Hyperparameters -------------------------------------------------------
IMG_SIZE = 224          # ResNet50's expected input size
BATCH_SIZE = 32         # Number of images the model process at a time
DROPOUT = 0.3           # regularization: randomly drops units to reduce overfitting

EPOCHS_PHASE1 = 8       # feature-extraction epochs (backbone frozen)
EPOCHS_PHASE2 = 10      # fine-tuning epochs (top of backbone unfrozen)
EPOCHS_PHASE3 = 5       # colour-only polish epochs (backbone + other heads frozen)
EPOCHS_PHASE4 = 5       # season+usage-only polish epochs (backbone + colour head frozen)

LR_PHASE1 = 1e-3        # normal learning rate: only the fresh heads are training
LR_PHASE2 = 1e-5        # TINY learning rate: we are nudging pretrained weights, not rewriting them
LR_PHASE3 = 1e-4        # only one shallow Dense layer is trainable, so this can be higher than phase 2
LR_PHASE4 = 1e-4        # same reasoning as phase 3: only two shallow Dense layers are trainable

# ResNet50 has more layers than MobileNetV2 (175 vs 154), so we scale up the
# unfreeze count to keep roughly the same fraction of the backbone trainable.
UNFREEZE_LAST_N = 50    # Unfreeze the N last layers (Stage 2)
# UNFREEZE_LAST_N the lower the value -> lower risk of overfitting, more layer will be trained based on our data

RANDOM_STATE = 42  # Random seed, so the random operations are reproducible

# Prevent extremely rare classes from dominating the total loss.
# Was 4.0: that let rare classes (Winter, Brown, Beige) become "cheap
# insurance" bets whenever the model was unsure, tanking their precision
# while starving recall on large classes (Blue) that got down-weighted
# in response. Lowered to reduce that overcorrection.
MAX_CLASS_WEIGHT = 2.5

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
# 1.5 Optional auxiliary data (colour-only, and season/usage-only)
# =========================================================================

def load_color_aux_data():
    """
    Loads the external colour-labelled dataset (see prepare_color_aux_dataset.py)
    if it's been generated, and shapes it to match train_df's columns so it can
    be concatenated directly.

    These rows have no real season/usage ground truth. We fill them with a
    placeholder that's an EXISTING valid class ("Fall" / "Casual") rather than
    a new/missing value, purely so label-encoding and the loss function have
    something in-range to work with. The placeholder value itself never
    matters, because has_season=has_usage=False zeroes out those heads' loss
    contribution entirely in make_dataset().
    """
    if not COLOR_AUX_CSV.exists():
        return None

    aux_df = pd.read_csv(COLOR_AUX_CSV)
    aux_df["season"] = "Fall"
    aux_df["usage"] = "Casual"
    aux_df["has_baseColour"] = True
    aux_df["has_season"] = False
    aux_df["has_usage"] = False
    return aux_df[[
        "id", "raw_image_path", "baseColour", "has_baseColour",
        "season", "has_season", "usage", "has_usage",
    ]]


def load_style_aux_data():
    """
    Loads the external season/usage-labelled dataset (see
    prepare_style_aux_dataset.py) if it's been generated.

    Every row here has a usage label (derived from style tags), but only
    ~73% have a real season value -- the rest get "Fall" as a placeholder,
    masked out via has_season=False (same placeholder mechanism as above).
    baseColour is entirely placeholder ("Black") since this dataset was
    never meant to train colour at all -- has_baseColour=False for every row.
    """
    if not STYLE_AUX_CSV.exists():
        return None

    aux_df = pd.read_csv(STYLE_AUX_CSV)
    aux_df["baseColour"] = "Black"
    aux_df["has_baseColour"] = False
    aux_df["has_season"] = aux_df["has_season"].astype(bool)
    aux_df["season"] = aux_df["season"].replace("", "Fall")
    aux_df.loc[aux_df["season"].isna(), "season"] = "Fall"
    aux_df["has_usage"] = aux_df["has_usage"].astype(bool)
    return aux_df[[
        "id", "raw_image_path", "baseColour", "has_baseColour",
        "season", "has_season", "usage", "has_usage",
    ]]


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

        # Raw balanced weights can get extreme for very rare classes (e.g.
        # season's Spring/Winter, or usage's Formal in the V2 data).
        # Cap extreme values so rare classes still matter without dominating training.
        weights = np.minimum(weights, MAX_CLASS_WEIGHT)

        class_weights[col] = {
            int(c): float(w)
            for c, w in zip(classes, weights)
        }
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
    (IMG_SIZE, IMG_SIZE, 3). We do NOT scale it ourselves here because
    ResNet50's preprocess_input expects raw 0-255 values and applies its own
    transform (RGB -> BGR, then subtract the ImageNet per-channel mean)."""
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
    # The V2 split CSVs already include a raw_image_path column (the
    # full-resolution image), so we don't need to reconstruct it from `id`.
    # Normalize to forward slashes: whoever generated the CSV on Windows gets
    # backslash-separated paths (pathlib.Path's native form there), which
    # Linux treats as a literal character, not a directory separator, and
    # forward slashes work fine on both OSes either way.
    paths = df["raw_image_path"].astype(str).str.replace("\\", "/", regex=False).values

    # Integer-encode each label column into a dict keyed by head name.
    # Eg. baseColour = ["Blue", "Black", "Blue", "Red"]  ----- encoders.transform ----> "color" : array([0, 1, 0, 2], dtype = int32)
    # For each attribute color, season, usage
    labels = {
        HEAD_NAMES[col]: encoders[col].transform(df[col]).astype("int32")
        for col in LABEL_COLS
    }
    # Class weigth used to blance the dataset example Spring vs Summer are very imbalanced
    # gradient descent will happily learn "always guess Summer" because that minimizes loss on average.
    # Weighting up the rare classes during training forces the model to actually pay attention to them while it's adjusting its weights.


    # When we are valiadating we do not use the weights, so the weight does not affect our predictions
    # This will simulate a more natural prediction
    # Eg. choosing a random clothing photo with no additional information --> predict his attributes
    if class_weights is not None:
        # Rows without real ground truth for a given label (auxiliary data
        # that only has SOME of the three labels) get a per-column mask of 0
        # for whichever heads they lack, so that head's loss contributes
        # nothing for that row -- masked multi-task learning via the same
        # per-sample-weight mechanism already used for class balancing.
        # Real clothing rows have has_baseColour/has_season/has_usage=True
        # (mask=1) implicitly (df.get(...) defaults to all-True when the
        # column doesn't exist), so this is a no-op for normal training data.
        # For each sample, look up the weight of its true class, per head,
        # then zero it out for rows that don't actually have that label.
        weights = {}
        for col in LABEL_COLS:
            head = HEAD_NAMES[col]
            w = np.array(
                [class_weights[col][int(c)] for c in labels[head]],
                dtype="float32",
            )
            has_col = df.get(
                f"has_{col}", pd.Series(True, index=df.index)
            ).values.astype("float32")
            weights[head] = w * has_col

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
        img = preprocess_input(img)          # ResNet50-specific scaling (BGR + mean subtraction)
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
    base_model = ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,        # drop ImageNet's 1000-class classifier; we add our own
        weights="imagenet",       # <- the pretrained knowledge we're reusing
    )
    base_model.trainable = False  # Phase 1: freeze the whole backbone

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)     # training=False keeps BatchNorm stats frozen
    x = layers.GlobalAveragePooling2D()(x)     # (H, W, C) feature map -> one vector per image
    x = layers.Dropout(DROPOUT)(x)

    # One softmax head per label. `n_classes` differs per label
    # (colour=12, season=4, usage=4). The `name=` MUST match the keys we used
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
    train_df["has_baseColour"] = True
    train_df["has_season"] = True
    train_df["has_usage"] = True
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    # Fit encoders on the real clothing rows only. Neither auxiliary dataset
    # (if present) introduces new classes -- every colour/season/usage value
    # they use already exists in our own taxonomy -- so this is safe either
    # way, but fitting on real data only keeps this step unaffected by
    # whether the auxiliary CSVs happen to exist.
    encoders = fit_label_encoders(train_df)

    color_aux_df = load_color_aux_data()
    if color_aux_df is not None:
        print(f"Auxiliary colour-only rows: {len(color_aux_df)} (from {COLOR_AUX_CSV})")
        train_df_color_merged = pd.concat([train_df, color_aux_df], ignore_index=True)
    else:
        train_df_color_merged = None

    style_aux_df = load_style_aux_data()
    if style_aux_df is not None:
        print(f"Auxiliary season/usage-only rows: {len(style_aux_df)} (from {STYLE_AUX_CSV})")
        train_df_style_merged = pd.concat([train_df, style_aux_df], ignore_index=True)
    else:
        train_df_style_merged = None

    # Phase 1 + 2 train on the REAL data only -- this reproduces our best
    # known result for season/usage exactly, since those phases never see
    # either auxiliary dataset. Auxiliary fine-tuning happens in Phase 3
    # (colour) and Phase 4 (season/usage).
    real_only_weights = compute_all_class_weights(train_df, encoders)
    train_ds = make_dataset(train_df, encoders, real_only_weights, training=True)
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
    print("\n=== Phase 2: fine-tuning the top of ResNet50 ===")
    base_model.trainable = True
    # Freeze everything EXCEPT the last UNFREEZE_LAST_N layers. The early layers
    # detect generic features we want to keep; the late layers are the ones
    # worth specializing to clothing.

    # Unfreeze the last (UNFREEZE_LAST_N) layers
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

    # ---- Phase 3: colour-only polish on the auxiliary data (if any) ----
    history3 = None
    if train_df_color_merged is not None:
        print("\n=== Phase 3: fine-tuning ONLY the colour head on the merged colour data ===")

        # Colour weights come from the MERGED data (auxiliary rows should
        # count toward colour balance). Season/usage weights are irrelevant
        # here since those heads are frozen, but compute_all_class_weights
        # needs a value for every column, so we just reuse the real-only ones.
        merged_color_weights = compute_all_class_weights(train_df_color_merged, encoders)
        phase3_weights = {
            "baseColour": merged_color_weights["baseColour"],
            "season": real_only_weights["season"],
            "usage": real_only_weights["usage"],
        }
        train_ds_color_merged = make_dataset(train_df_color_merged, encoders, phase3_weights, training=True)

        # Freeze EVERYTHING except the colour head. With the backbone frozen,
        # there is no path left for the auxiliary rows' gradients to reach
        # season/usage -- not "unlikely to", structurally cannot.
        base_model.trainable = False
        for layer in model.layers:
            if layer.name in ("season", "usage"):
                layer.trainable = False
            elif layer.name == "color":
                layer.trainable = True

        compile_model(model, LR_PHASE3)
        history3 = model.fit(
            train_ds_color_merged,
            validation_data=val_ds,
            epochs=EPOCHS_PHASE3,
            callbacks=callbacks,
        )
    else:
        print("\nNo auxiliary colour data found -- skipping Phase 3.")

    # ---- Phase 4: season+usage-only polish on the auxiliary data (if any) ----
    history4 = None
    if train_df_style_merged is not None:
        print("\n=== Phase 4: fine-tuning ONLY season+usage on the merged style data ===")

        # Season/usage weights come from the MERGED data. Colour weight is
        # irrelevant here since that head is frozen, but reuse the real-only
        # value for the same reason as Phase 3.
        merged_style_weights = compute_all_class_weights(train_df_style_merged, encoders)
        phase4_weights = {
            "baseColour": real_only_weights["baseColour"],
            "season": merged_style_weights["season"],
            "usage": merged_style_weights["usage"],
        }
        train_ds_style_merged = make_dataset(train_df_style_merged, encoders, phase4_weights, training=True)

        # Freeze EVERYTHING except season/usage. Mirrors Phase 3: with the
        # backbone AND the colour head frozen, the auxiliary rows' gradients
        # (including the ~27% with no real season, masked via has_season=0)
        # have no path to reach colour at all.
        base_model.trainable = False
        for layer in model.layers:
            if layer.name == "color":
                layer.trainable = False
            elif layer.name in ("season", "usage"):
                layer.trainable = True

        compile_model(model, LR_PHASE4)
        history4 = model.fit(
            train_ds_style_merged,
            validation_data=val_ds,
            epochs=EPOCHS_PHASE4,
            callbacks=callbacks,
        )
    else:
        print("\nNo auxiliary season/usage data found -- skipping Phase 4.")

    # ---- Save everything the evaluate script + demo will need ----
    print(f"\nSaving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    # Stitch each phase's history together so we can plot one curve.
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history.get(key, [])
        if history3 is not None:
            combined[key] = combined[key] + history3.history.get(key, [])
        if history4 is not None:
            combined[key] = combined[key] + history4.history.get(key, [])
    with open(HISTORY_PATH, "w") as f:
        json.dump(combined, f)

    print("Done. Next: python src/CNN/evaluate_cnn.py")


if __name__ == "__main__":
    main()
