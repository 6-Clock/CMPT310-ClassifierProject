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
│   └── processed/     ← resized .npy image arrays (auto-generated)
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

### 4. Explore the Data

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
