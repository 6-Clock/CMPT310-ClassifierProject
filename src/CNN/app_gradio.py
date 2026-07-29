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

#pinterest style page
TAG_TOP_K = 2


def format_tag(label: str) -> str:
    cleaned = "-".join(str(label).strip().lower().split())
    return f"#{cleaned}"


def suggest_pin_tags(pil_image):
    if pil_image is None:
        return (
            gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                allow_custom_value=True,
            ),
            "Upload a clothing image to generate tags.",
        )

    preds = model.predict(preprocess(pil_image), verbose=0)

    choices = []
    selected = []
    confidence_lines = []

    for head, col in HEAD_TO_COL.items():
        probs = np.asarray(preds[head][0])
        class_names = encoders[col].classes_

        top_indices = np.argsort(probs)[::-1][:TAG_TOP_K]

        for rank, class_index in enumerate(top_indices):
            tag = format_tag(class_names[class_index])

            if tag not in choices:
                choices.append(tag)

            # Automatically select the strongest prediction
            if rank == 0:
                selected.append(tag)

        best_index = top_indices[0]
        best_probability = float(probs[best_index])
        best_label = class_names[best_index]

        confidence_lines.append(
            f"**{head.title()}:** {best_label} "
            f"({best_probability:.1%})"
        )

    updated_tags = gr.Dropdown(
        choices=choices,
        value=selected,
        multiselect=True,
        allow_custom_value=True,
        interactive=True,
    )

    return updated_tags, " · ".join(confidence_lines)

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

with demo.route("Create Listing", "/create-pin"):
    gr.Markdown("# Create Listing")

    with gr.Row():
        with gr.Column(scale=5):
            pin_image = gr.Image(
                type="pil",
                sources=["upload"],
                label="Choose a file or drag and drop it here",
                height=520,
            )

        with gr.Column(scale=7):
            pin_title = gr.Textbox(
                label="Title",
                placeholder="Title",
            )

            pin_description = gr.Textbox(
                label="Description",
                placeholder="Describe your clothing item",
                lines=4,
            )

            pin_link = gr.Textbox(
                label="Price",
                placeholder="0.00",
            )

            pin_board = gr.Dropdown(
                choices=[
                    "Fashion Ideas",
                    "Outfit Inspiration",
                    "Seasonal Looks",
                ],
                allow_custom_value=True,
                label="Board",
            )

            pin_tags = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                allow_custom_value=True,
                label="Suggested tags",
            )

            tag_confidence = gr.Markdown(
                "Upload an image to generate CNN tag suggestions."
            )

            save_button = gr.Button("Post Listing", variant="primary")

    pin_image.change(
        fn=suggest_pin_tags,
        inputs=pin_image,
        outputs=[pin_tags, tag_confidence],
    )


if __name__ == "__main__":
    demo.launch()
