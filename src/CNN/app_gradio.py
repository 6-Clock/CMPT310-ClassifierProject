"""
Gradio web demo (Milestone 2).

Upload a clothing image -> the trained CNN predicts colour, season, and usage,
each with a confidence score. Low-confidence predictions are flagged, matching
our e-commerce story ("send unclear cases to a human").

Run from the project root, venv active, AFTER train_cnn.py:
    python src/CNN/app_gradio.py
Then open the local URL it prints (e.g. http://127.0.0.1:7860).
"""

from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

import joblib
import gradio as gr

MODEL_PATH = Path("models/CNN/resnet50_multitask.keras")
ENCODERS_PATH = Path("models/CNN/label_encoders.joblib")

IMG_SIZE = 224
# Below this top probability we flag the prediction as "uncertain".
CONFIDENCE_THRESHOLD = 0.5   # TODO: tune this once you see real predictions

# Load once at startup (not per request) so the demo stays fast.
model = keras.models.load_model(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Map model output head -> the encoder that decodes its integer classes.
HEAD_TO_COL = {"color": "baseColour", "season": "season", "usage": "usage"}


def preprocess(pil_image):
    """PIL image (any size) -> a (1, 224, 224, 3) batch ready for the model,
    using the SAME preprocessing as training."""
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32")      # 0-255, preprocess_input scales it
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)         # add the batch dimension


def predict(pil_image):
    """Returns three gr.Label-friendly dicts {class_name: probability}, one per
    head, plus a text flag for review, so the UI shows a ranked bar of
    confidences per attribute and calls out predictions worth a human look."""
    if pil_image is None:
        return {}, {}, {}, ""

    batch = preprocess(pil_image)
    preds = model.predict(batch)   # dict: {"color": (1,12), "season": (1,4), "usage": (1,4)}

    outputs = {}
    low_confidence_heads = []

    for head, col in HEAD_TO_COL.items():
        probs = preds[head][0]                 # (num_classes,)
        class_names = encoders[col].classes_

        # gr.Label wants {class_name: probability}, so it can draw a ranked
        # bar of confidences. `float(p)` converts numpy floats to plain
        # Python floats, since Gradio can't serialize numpy types to JSON.
        outputs[head] = {name: float(p) for name, p in zip(class_names, probs)}

        top_confidence = float(np.max(probs))
        if top_confidence < CONFIDENCE_THRESHOLD:
            low_confidence_heads.append(head)

    # Mirrors the proposal's e-commerce story: low-confidence predictions get
    # flagged for a human to double-check instead of being auto-applied.
    if low_confidence_heads:
        flag_text = (
            f"⚠ Low confidence on: {', '.join(low_confidence_heads)} "
            f"(below {CONFIDENCE_THRESHOLD:.0%}) — recommend human review."
        )
    else:
        flag_text = "All predictions above the confidence threshold."

    return outputs["color"], outputs["season"], outputs["usage"], flag_text


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a clothing image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Colour"),
        gr.Label(num_top_classes=3, label="Season"),
        gr.Label(num_top_classes=3, label="Usage"),
        gr.Textbox(label="Review flag"),
    ],
    title="Clothing Attribute Classifier",
    description="Predicts colour, season, and usage from a single clothing photo.",
)


if __name__ == "__main__":
    demo.launch()
