"""
Gradio web demo (Milestone 2).

Upload a clothing image -> the trained CNN predicts colour, season, and usage,
each with a confidence score. Low-confidence predictions are flagged, matching
our e-commerce story ("send unclear cases to a human").

Run from the project root, venv active, AFTER train_cnn.py:
    python src/CNN/app_gradio.py
Then open the local URL it prints (e.g. http://127.0.0.1:7860).
"""

from __future__ import annotations

import base64
import html
import io
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

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

#Demo post storage. local, will disappear on restart.
POSTS: list[dict[str, object]] = []
POSTS_LOCK = threading.Lock()


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

def image_to_data_url(pil_image: Image.Image) -> str:
    """Convert an uploaded PIL image to an embeddable JPEG data URL."""
    image = pil_image.convert("RGB").copy()
    image.thumbnail((1200, 1600))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def safe_external_link(link: str) -> str | None:
    """Allow only normal HTTP(S) links in rendered post cards."""
    candidate = (link or "").strip()
    if not candidate:
        return None

    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def save_pin(
    pil_image: Image.Image | None,
    title: str,
    description: str,
    link: str,
    board: str | None,
    tags: list[str] | None,
    ai_modified: bool,
) -> str:
    """Validate and add a post to the in-memory Pinterest-style feed."""
    clean_title = (title or "").strip()
    clean_description = (description or "").strip()
    clean_board = (board or "Uncategorized").strip() or "Uncategorized"
    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]

    if pil_image is None:
        return "⚠ Please upload an image before saving the Pin."
    if not clean_title:
        return "⚠ Please add a title before saving the Pin."

    post = {
        "image": image_to_data_url(pil_image),
        "title": clean_title,
        "description": clean_description,
        "link": safe_external_link(link),
        "board": clean_board,
        "tags": clean_tags,
        "ai_modified": bool(ai_modified),
        "created_at": datetime.now().strftime("%b %d, %Y at %I:%M %p"),
    }

    with POSTS_LOCK:
        POSTS.insert(0, post)
        total_posts = len(POSTS)

    return (
        "### ✅ Pin saved\n"
        f"**{clean_title}** was added to the **{clean_board}** board.  \n"
        f"There {'is' if total_posts == 1 else 'are'} now **{total_posts}** "
        f"post{'s' if total_posts != 1 else ''} in the feed. Open the **Posts** page to view it."
    )


def post_matches(post: dict[str, object], search_text: str, board: str) -> bool:
    """Return True when a post matches the selected feed filters."""
    if board != "All boards" and str(post["board"]) != board:
        return False

    query = (search_text or "").strip().lower()
    if not query:
        return True

    searchable_text = " ".join(
        [
            str(post.get("title", "")),
            str(post.get("description", "")),
            str(post.get("board", "")),
            " ".join(str(tag) for tag in post.get("tags", [])),
        ]
    ).lower()
    return query in searchable_text


def render_post_cards(posts: list[dict[str, object]]) -> str:
    """Render the saved posts as a Pinterest-style masonry grid."""
    if not posts:
        return """
        <div class="empty-feed">
            <div class="empty-feed-icon">📌</div>
            <h2>No posts found</h2>
            <p>Create a Pin, or change the current search and board filters.</p>
        </div>
        """

    cards: list[str] = []
    for post in posts:
        title = html.escape(str(post.get("title", "Untitled")))
        description = html.escape(str(post.get("description", "")))
        board = html.escape(str(post.get("board", "Uncategorized")))
        created_at = html.escape(str(post.get("created_at", "")))
        image_url = html.escape(str(post.get("image", "")), quote=True)
        link = post.get("link")
        ai_modified = bool(post.get("ai_modified", False))

        tag_html = "".join(
            f'<span class="post-tag">{html.escape(str(tag))}</span>'
            for tag in post.get("tags", [])
        )

        description_html = (
            f'<p class="post-description">{description}</p>' if description else ""
        )
        ai_badge = '<span class="ai-badge">AI-modified</span>' if ai_modified else ""
        link_html = (
            f'<a class="post-link" href="{html.escape(str(link), quote=True)}" '
            'target="_blank" rel="noopener noreferrer">Visit link ↗</a>'
            if link
            else ""
        )

        cards.append(
            f"""
            <article class="post-card">
                <img class="post-image" src="{image_url}" alt="{title}">
                <div class="post-content">
                    <div class="post-board-row">
                        <span class="post-board">{board}</span>
                        {ai_badge}
                    </div>
                    <h3>{title}</h3>
                    {description_html}
                    <div class="post-tags">{tag_html}</div>
                    <div class="post-footer">
                        <span>{created_at}</span>
                        {link_html}
                    </div>
                </div>
            </article>
            """
        )

    return f'<div class="posts-grid">{"".join(cards)}</div>'


def refresh_posts(search_text: str = "", selected_board: str = "All boards"):
    """Refresh the board filter, post count, and feed HTML."""
    with POSTS_LOCK:
        post_snapshot = list(POSTS)

    board_choices = ["All boards"] + sorted(
        {str(post.get("board", "Uncategorized")) for post in post_snapshot}
    )
    if selected_board not in board_choices:
        selected_board = "All boards"

    filtered_posts = [
        post
        for post in post_snapshot
        if post_matches(post, search_text, selected_board)
    ]

    if len(filtered_posts) == len(post_snapshot):
        count_text = f"### {len(post_snapshot)} post{'s' if len(post_snapshot) != 1 else ''}"
    else:
        count_text = (
            f"### Showing {len(filtered_posts)} of {len(post_snapshot)} "
            f"post{'s' if len(post_snapshot) != 1 else ''}"
        )

    return (
        gr.Dropdown(choices=board_choices, value=selected_board),
        count_text,
        render_post_cards(filtered_posts),
    )

APP_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
}

.pin-heading {
    border-bottom: 1px solid #ddddda;
    padding-bottom: 12px;
    margin-bottom: 18px;
}

.pin-upload {
    border-radius: 28px !important;
    overflow: hidden !important;
}

.pin-upload .image-container,
.pin-upload [data-testid="image"] {
    border-radius: 28px !important;
    background: #efefec !important;
}

.pin-field input,
.pin-field textarea,
.pin-field .wrap {
    border-radius: 18px !important;
}

.tag-picker .wrap {
    border-radius: 18px !important;
}

#save-pin,
#save-pin button {
    border-radius: 999px !important;
    background: #e60023 !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
}

.feed-toolbar {
    align-items: end !important;
    margin-bottom: 8px;
}

.posts-grid {
    column-count: 4;
    column-gap: 18px;
    padding: 6px 2px 24px;
}

.post-card {
    display: inline-block;
    width: 100%;
    break-inside: avoid;
    margin: 0 0 18px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 22px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}

.post-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 20px rgba(0, 0, 0, 0.11);
}

.post-image {
    display: block;
    width: 100%;
    height: auto;
    object-fit: cover;
}

.post-content {
    padding: 14px 15px 16px;
}

.post-content h3 {
    font-size: 1rem;
    line-height: 1.3;
    margin: 8px 0 6px;
}

.post-description {
    margin: 0 0 10px;
    font-size: 0.9rem;
    line-height: 1.45;
    color: var(--body-text-color-subdued);
}

.post-board-row,
.post-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.post-board {
    font-size: 0.78rem;
    font-weight: 700;
}

.ai-badge {
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 0.7rem;
    background: var(--background-fill-secondary);
}

.post-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 10px;
}

.post-tag {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    background: var(--background-fill-secondary);
}

.post-footer {
    border-top: 1px solid var(--border-color-primary);
    margin-top: 12px;
    padding-top: 10px;
    font-size: 0.72rem;
    color: var(--body-text-color-subdued);
}

.post-link {
    font-weight: 700;
    text-decoration: none;
}

.empty-feed {
    text-align: center;
    padding: 70px 20px;
    border: 1px dashed var(--border-color-primary);
    border-radius: 24px;
    background: var(--background-fill-secondary);
}

.empty-feed-icon {
    font-size: 2.4rem;
}

@media (max-width: 1100px) {
    .posts-grid { column-count: 3; }
}

@media (max-width: 780px) {
    .posts-grid { column-count: 2; }
}

@media (max-width: 520px) {
    .posts-grid { column-count: 1; }
}
"""

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

            save_button = gr.Button(
                "Save Pin",
                variant="primary",
                elem_id="save-pin",
            )
            save_status = gr.Markdown()

        pin_image.change(
        fn=suggest_pin_tags,
        inputs=pin_image,
        outputs=[pin_tags, tag_confidence],
        )

        save_button.click(
            fn=save_pin,
            inputs=[
                pin_image,
                pin_title,
                pin_description,
                pin_link,
                pin_board,
                pin_tags
            ],
            outputs=save_status,
        )


with demo.route("Posts", "/posts") as posts_page:
    gr.Markdown("# Posts", elem_classes=["pin-heading"])

    with gr.Row(elem_classes=["feed-toolbar"]):
        post_search = gr.Textbox(
            label="Search posts",
            placeholder="Search titles, descriptions, boards, or tags",
            scale=5,
        )
        board_filter = gr.Dropdown(
            choices=["All boards"],
            value="All boards",
            label="Board",
            scale=3,
        )
        refresh_button = gr.Button("Refresh posts", scale=1)

    post_count = gr.Markdown("### 0 posts")
    posts_feed = gr.HTML(render_post_cards([]))

    refresh_outputs = [board_filter, post_count, posts_feed]

    refresh_button.click(
        fn=refresh_posts,
        inputs=[post_search, board_filter],
        outputs=refresh_outputs,
    )
    post_search.change(
        fn=refresh_posts,
        inputs=[post_search, board_filter],
        outputs=refresh_outputs,
    )
    board_filter.change(
        fn=refresh_posts,
        inputs=[post_search, board_filter],
        outputs=refresh_outputs,
    )
    posts_page.load(
        fn=refresh_posts,
        inputs=[post_search, board_filter],
        outputs=refresh_outputs,
    )


if __name__ == "__main__":
    demo.launch()
