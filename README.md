# CMPT 310 — Multi-Label Clothing Classification

Multi-label image classifier that predicts four clothing attributes (color, season, type, usage) using Transfer Learning (MobileNetV2 / ResNet50).

---

## Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow / Keras (MobileNetV2, ResNet50)
- **ML / Data:** scikit-learn, NumPy, pandas
- **Image processing:** Pillow
- **Visualization:** Matplotlib
- **Demo (Milestone 2):** Gradio
- **Dataset:** Kaggle CLI (`kaggle`)
- **Notebooks:** Jupyter

---

## Project Structure

```
CMPT310 Project/
├── data/
│   ├── raw/           ← downloaded Kaggle dataset goes here
│   └── processed/     ← cleaned CSV splits and 96x96 processed images (auto-generated)
├── src/
│   └── data_loader.py      ← load CSV, check labels, filter & balance
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
└── PROJECT.MD
```

---

## Setup

### 1. Create a Virtual Environment

Open a PowerShell terminal in the project folder, then run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To leave the virtual enviroment
```powershell
deactivate
```

### 2. Set Up Kaggle Credentials

1. Go to [kaggle.com](https://www.kaggle.com) → your profile → **Account** → scroll to **API** → click **Create New Token**
2. This downloads a file called `kaggle.json`
3. Move it to: `C:\Users\<your-username>\.kaggle\kaggle.json`
If directory does not exist create it!

### 3. Download the Dataset

With the venv active, run:

```powershell
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/raw --unzip
```

This places `styles.csv` and an `images/` folder inside `data/raw/`.

**Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — ~44,000 product images at 80×60 px with rich metadata.

| Column in `styles.csv` | Label used |
|------------------------|------------|
| `baseColour`           | color      |
| `season`               | season     |
| `articleType`          | type       |
| `usage`                | usage      |

### 4. Preprocess and Split the Dataset

After dowloading the Kaggle dataset, clean the dataset and balance it:

```bash
python src/balance_clean_dataset.py
```

Run this command from the project root:

```bash
python src/preprocess_split.py
```

This script prepares the dataset for model training by:

- Loading `data/raw/styles.csv`
- Removing rows with missing labels for `baseColour`, `season`, `articleType`, or `usage`
- Removing rows where the matching image file is missing
- Resizing images to `96x96` RGB using aspect-ratio-preserving padding
- Saving processed images into `data/processed/images_96/`
- Creating train/validation/test CSV files

Generated files:

```text
data/processed/metadata_clean.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/images_96/
```

Split ratio:

```text
70% train
15% validation
15% test
```

For model training, use:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Each CSV contains:

```text
id
image_path
baseColour
season
articleType
usage
```

The `image_path` column points to the processed `96x96` image.

Note: The `data/` folder is ignored by GitHub, so each team member needs to download the Kaggle dataset and run the preprocessing script locally.

### 5. Train the KNN Baseline

With the venv active and the dataset downloaded, run these two scripts in order from the project root:

```powershell
python src/balance_clean_dataset.py
python src/knn.py
```

`balance_clean_dataset.py`:
- Loads `data/raw/styles.csv`, keeps only `Apparel` items with reliable `baseColour`, `season`, and `usage` labels
- Balances the dataset by `usage` (up to 1500 samples per class) and tops up underrepresented seasons
- Saves the result to `data/processed/clean_colour_season_style.csv`

`knn.py`:
- Loads `data/processed/clean_colour_season_style.csv` (must be run after `balance_clean_dataset.py`)
- Resizes each image to 96×96, flattens it, and encodes the `baseColour`, `season`, and `usage` labels
- Splits into train/test (80/20, stratified by `usage`), trains a `MultiOutputClassifier` wrapping `KNeighborsClassifier(k=5, weights="distance")`
- Prints exact-match accuracy plus per-label accuracy, weighted F1, and a classification report
- Saves the trained model to `models/knn_baseline.joblib`

#### About `knn_baseline.joblib`

A dictionary saved with `joblib.dump`, containing everything needed to reuse the trained model without retraining:

- `model` — the fitted `MultiOutputClassifier` (KNN) for `baseColour`, `season`, and `usage`
- `label_encoders` — a `LabelEncoder` per label, used to map predictions back to text
- `label_columns` — `["baseColour", "season", "usage"]`
- `image_size` — `(96, 96)`, the size images must be resized to before prediction

### 5.1 Use the train model

**Example: load the model and predict on a new image**
**Use the included script:** drop a test image into a `testingimgs/` folder in the project root, then run:

```powershell
python src/predict_knn.py testingimgs/your_image.jpg
```

This prints the predicted `baseColour`, `season`, and `usage` for that image.

### 6. Explore the Data

Launch Jupyter and open the exploration notebook:

```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

The notebook will:
- Confirm which of the four labels exist in the dataset
- Plot class distributions per label
- Show a sample grid of 9 clothing images

---

## Verify the Install

```powershell
python -c "import tensorflow, sklearn, PIL; print('All imports OK')"
```

---

## Milestone Summary

| Milestone | Goal |
|-----------|------|
| **1** | Data download, label inspection, preprocessing pipeline, KNN baseline skeleton |
| **2** | Fine-tune MobileNetV2/ResNet50, full evaluation, Gradio demo |
git 