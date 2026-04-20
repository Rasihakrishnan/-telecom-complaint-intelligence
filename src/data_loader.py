"""
data_loader.py
──────────────
Handles all raw data I/O: loading the FCC complaint CSV and the
YouTube comment JSON files, performing initial structural cleaning,
and persisting the cleaned artefacts to data/processed/.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def load_raw_fcc(path: str) -> pd.DataFrame:
    """Load the raw FCC complaint CSV from *path*."""
    logger.info("Loading raw FCC data from: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df
def clean_fcc(
    df: pd.DataFrame,
    columns_to_drop: list,
    required_columns: list,
    datetime_columns: list,
) -> pd.DataFrame:
   
    logger.info("Cleaning FCC data …")

    existing_drops = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    logger.debug("Dropped columns: %s", existing_drops)
    # 2 – drop rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=[c for c in required_columns if c in df.columns])
    logger.info("Dropped %d rows with NaN in required columns", before - len(df))
    # 3 – parse datetimes
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.debug("Parsed '%s' as datetime", col)

    logger.info("FCC data shape after cleaning: %s", df.shape)
    return df
def save_cleaned_fcc(df: pd.DataFrame, output_path: str) -> None:
    """Write the cleaned FCC DataFrame to a CSV at *output_path*."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned FCC data → %s", output_path)
def load_cleaned_fcc(path: str) -> pd.DataFrame:
    """Load the already-cleaned FCC CSV (skips raw-cleaning steps)."""
    logger.info("Loading cleaned FCC data from: %s", path)
    df = pd.read_csv(path, parse_dates=["ticket_created", "date_created"])
    logger.info("Shape: %s", df.shape)
    return df
def load_youtube_comments(
    path_part1: str,
    path_part2: str,
) -> pd.DataFrame:
    """
    Load and concatenate the two YouTube comment JSON files.

    Returns a single DataFrame with a *comment_text* column.
    """
    logger.info("Loading YouTube comments …")

    def _read_json(p: str) -> pd.DataFrame:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.json_normalize(data)
    comments1 = _read_json(path_part1)
    comments2 = _read_json(path_part2)
    comments = pd.concat([comments1, comments2], ignore_index=True)
    comments["comment_text"] = comments["comment"]
    logger.info("Loaded %d comments", len(comments))
    return comments


def save_comments(df: pd.DataFrame, output_path: str) -> None:
    """Persist the processed comments DataFrame to a CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved comments → %s", output_path)
# ──────────────────────────────────────────────────────────────
#  Convenience wrapper used by main.py
# ──────────────────────────────────────────────────────────────
def run_fcc_pipeline(cfg: dict) -> pd.DataFrame:
    """
    Full FCC load-and-clean pipeline driven by *cfg* (the parsed config).

    Returns the cleaned DataFrame and saves it to disk.
    """
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    df = load_raw_fcc(paths["raw_fcc_data"])
    df = clean_fcc(
        df,
        columns_to_drop=data_cfg["columns_to_drop_raw"],
        required_columns=data_cfg["required_columns"],
        datetime_columns=data_cfg["datetime_columns"],
    )
    save_cleaned_fcc(df, paths["cleaned_fcc_data"])
    return df
def run_comments_pipeline(cfg: dict) -> pd.DataFrame:
    """Load and return raw YouTube comments (NLP happens in preprocessing.py)."""
    paths = cfg["paths"]
    return load_youtube_comments(
        path_part1=paths["comments_part1"],
        path_part2=paths["comments_part2"],
    )
