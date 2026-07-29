from pathlib import Path
import json

import matplotlib.pyplot as plt


HISTORY_PATH = Path("models/CNN/history.json")
REPORTS_DIR = Path("reports")

# You need history.json for this code.
# When you run the python files you ll get this file.
def plot_training_results():
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        history = json.load(file)

    print("Available keys:")
    for key in history.keys():
        print(key)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    heads = [
        ("color", "Color"),
        ("season", "Season"),
        ("usage", "Usage"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for row, (key, title) in enumerate(heads):
        # Accuracy
        axes[row, 0].plot(
            history.get(f"{key}_accuracy", []),
            label="Train Accuracy",
        )
        axes[row, 0].plot(
            history.get(f"val_{key}_accuracy", []),
            label="Validation Accuracy",
        )
        axes[row, 0].set_title(f"{title} Accuracy")
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("Accuracy")
        axes[row, 0].legend()
        axes[row, 0].grid(alpha=0.3)

        # Loss
        axes[row, 1].plot(
            history.get(f"{key}_loss", []),
            label="Train Loss",
        )
        axes[row, 1].plot(
            history.get(f"val_{key}_loss", []),
            label="Validation Loss",
        )
        axes[row, 1].set_title(f"{title} Loss")
        axes[row, 1].set_xlabel("Epoch")
        axes[row, 1].set_ylabel("Loss")
        axes[row, 1].legend()
        axes[row, 1].grid(alpha=0.3)

    fig.suptitle(
        "Training and Validation Performance by Classification Head",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = REPORTS_DIR / "plots_image.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_training_results()