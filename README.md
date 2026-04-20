# Telecom Consumer Complaint Intelligence
### Public Sentiment Analysis System

An end-to-end ML pipeline that fuses **FCC regulatory complaint data** with **YouTube public sentiment** to classify complaint severity and consumer pain levels across telecom issue categories.

---

## Project Structure

```
telecom-ml-project/
├── data/
│   ├── raw/                  # Place original source files here (never edited)
│   │   ├── fcc_full_3million.csv
│   │   ├── comments_part_1.json
│   │   └── comments_part_2.json
│   └── processed/            # Auto-generated cleaned artefacts
├── notebooks/
│   └── analysis.ipynb        # EDA and final visualisations (run after pipeline)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Raw I/O and initial cleaning
│   ├── preprocessing.py      # Feature engineering, TF-IDF, NLP
│   ├── model.py              # Training, evaluation, persistence
│   └── utils.py              # Config loading, logging, summary builders
├── models/                   # Saved .pkl artefacts (auto-generated)
├── config.yaml               # All hyperparameters and file paths
├── requirements.txt
├── main.py                   # Pipeline entry point
└── README.md
```

---

## Quick Start

### 1 — Clone & create a virtual environment

```bash
git clone <your-repo-url>
cd telecom-ml-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add your raw data

Place the following files in `data/raw/`:

| File | Description |
|---|---|
| `fcc_full_3million.csv` | Raw FCC consumer complaint export |
| `comments_part_1.json` | YouTube comments batch 1 |
| `comments_part_2.json` | YouTube comments batch 2 |

### 4 — Configure paths and hyperparameters

Open `config.yaml` and verify the `paths` block matches your file locations. All model hyperparameters are documented there too.

### 5 — Run the full pipeline

```bash
python main.py
```

Run only a specific branch if needed:

```bash
python main.py --stage fcc        # FCC load, features, and model training only
python main.py --stage comments   # YouTube NLP and clustering only
python main.py --log-level DEBUG  # Verbose output
```

### 6 — Explore results in the notebook

```bash
jupyter lab notebooks/analysis.ipynb
```

---

## Models Trained

| Model | Task | Algorithm | Target |
|---|---|---|---|
| RF Regressor | Severity prediction | Random Forest | `severity_score` (1–5) |
| XGB Classifier | Pain level classification | XGBoost | `pain_level` (Low / Medium / High) |
| KMeans | Topic discovery | KMeans | `topic_cluster` (5 clusters) |

Saved artefacts land in `models/` as `.pkl` files and can be loaded with:

```python
from src.model import load_artifact
model = load_artifact("models/xgb_pain_level.pkl")
```

---

## Configuration Reference

All tunable values live in `config.yaml`:

- **`paths`** — input/output file locations
- **`data`** — columns to drop, required fields, datetime columns
- **`features`** — TF-IDF settings, label-encode targets, structural feature lists
- **`severity_map`** — issue → numeric severity mapping
- **`nlp`** — sentiment thresholds, keyword dictionaries
- **`models`** — hyperparameters for RF, XGB, and KMeans
- **`training`** — test split size, random seeds, pain score construction weights

---

## Development Notes

- The `pain_level` target is **synthetically constructed** from weighted text features plus Gaussian noise, then quantile-binned. This avoids label leakage while producing a realistic classification task.
- The high R² on severity regression (~0.95) is expected: severity was derived directly from issue categories, so TF-IDF on issue text recovers it almost perfectly.
- NLTK resources are auto-downloaded on first run via `ensure_nltk_resources()` in `preprocessing.py`.

---

## License

MIT
