"""
Run the trained KNN baseline on a single image.

Usage (from project root, venv active):
    python src/KNN/predict_knn.py testingimgs/your_image.jpg
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

MODEL_PATH = Path("models/KNN/knn_baseline.joblib")


def predict(image_path: Path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run src/KNN/knn.py first.")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    label_encoders = bundle["label_encoders"]
    label_columns = bundle["label_columns"]
    image_size = bundle["image_size"]

    img = Image.open(image_path).convert("RGB").resize(image_size)
    X = (np.array(img, dtype=np.float32) / 255.0).flatten().reshape(1, -1)

    pred = model.predict(X)[0]

    print(f"\nPredictions for {image_path.name}:")
    for i, col in enumerate(label_columns):
        label = label_encoders[col].inverse_transform([pred[i]])[0]
        print(f"  {col}: {label}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/KNN/predict_knn.py <path_to_image>")
        raise SystemExit(1)

    predict(Path(sys.argv[1]))
