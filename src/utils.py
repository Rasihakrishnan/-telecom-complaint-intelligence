"""
utils.py
────────
Shared helpers: config loading, logging setup, summary builders,
and metrics persistence used across the pipeline.

Fixes applied (v2):
  - save_metrics_json helper added
  - save_topic_labels helper added
  - build_fcc_summary uses 'issue' column (safer fallback added)
"""

import json
import logging
import sys
from pathlib import Path

import yaml


# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Parse and return the YAML config file."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ──────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to write to stdout with a clean format."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ──────────────────────────────────────────────────────────────
#  Metrics persistence (NEW in v2)
# ──────────────────────────────────────────────────────────────

def save_metrics_json(metrics: dict, path: str) -> None:
    """
    Persist any metrics dict to a JSON file at *path*.
    Used by model.py evaluate_* functions.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.getLogger(__name__).info("Saved metrics → %s", path)


def load_metrics_json(path: str) -> dict:
    """Load a previously saved metrics JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_topic_labels(topics: dict, path: str = "models/kmeans_topic_labels.json") -> None:
    """
    Save KMeans cluster → top words mapping to JSON.
    NEW in v2 — review noted clusters were never labelled.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(topics, f, indent=2)
    logging.getLogger(__name__).info("Saved topic labels → %s", path)


# ──────────────────────────────────────────────────────────────
#  Summary builders (used in analysis.ipynb)
# ──────────────────────────────────────────────────────────────

def build_fcc_summary(df) -> "pd.DataFrame":
    """Aggregate FCC complaints by issue type for the merged report."""
    import pandas as pd
    col = "issue_type" if "issue_type" in df.columns else "issue"
    summary = df[col].value_counts().reset_index()
    summary.columns = ["issue_type", "complaint_count"]
    return summary


def build_youtube_summary(comments) -> "pd.DataFrame":
    """Aggregate YouTube comments by issue_type for the merged report."""
    summary = comments.groupby("issue_type").agg(
        sentiment_score=("sentiment_score", "mean"),
        youtube_comment_count=("sentiment_label", "count"),
    ).reset_index()
    return summary


def merge_summaries(fcc_summary, youtube_summary) -> "pd.DataFrame":
    """Left-join FCC and YouTube summaries on issue_type."""
    import pandas as pd
    return pd.merge(fcc_summary, youtube_summary, on="issue_type", how="left")


# ──────────────────────────────────────────────────────────────
#  Misc
# ──────────────────────────────────────────────────────────────

def ensure_dirs(paths: list) -> None:
    """Create directories (including parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
