"""
main.py
───────
Orchestrates the full Telecom Complaint Intelligence pipeline.

Fixes applied (v2):
  - KMeans topic labels extracted and saved after clustering
  - Feature importance saved for RF model
  - ensure_dirs called at startup so all folders exist before writing

Usage
-----
    python main.py                        # full pipeline
    python main.py --config my_cfg.yaml   # custom config
    python main.py --stage fcc            # FCC stages only (1,3,5,6)
    python main.py --stage comments       # Comments stages only (2,4,7)
    python main.py --log-level DEBUG      # verbose output
"""

import argparse
import logging

import pandas as pd
from scipy import sparse  # Added for structural feature conversion
from scipy.sparse import hstack, vstack  # Added vstack here

# ... rest of your imports ...

from src.data_loader import run_fcc_pipeline, run_comments_pipeline, save_comments
from src.model import (
    run_severity_model,
    run_pain_level_model,
    train_kmeans,
    get_kmeans_topic_labels,
    save_artifact,
)
from src.preprocessing import (
    preprocess_fcc,
    preprocess_comments,
    build_tfidf_hybrid,
)
from src.utils import (
    build_fcc_summary,
    build_youtube_summary,
    merge_summaries,
    load_config,
    setup_logging,
    ensure_dirs,
    save_topic_labels,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Stage runners
# ──────────────────────────────────────────────────────────────

def run_fcc_stages(cfg: dict) -> pd.DataFrame:
    """Stages 1, 3, 5, 6 — FCC load → preprocess → train models."""
    paths     = cfg["paths"]
    feat_cfg  = cfg["features"]
    train_cfg = cfg["training"]

    # Stage 1 – Load & clean
    logger.info("═══ Stage 1: Load & clean FCC data ═══")
    df = run_fcc_pipeline(cfg)

    # Stage 3 – Feature engineering
    logger.info("═══ Stage 3: FCC feature engineering ═══")
    df = preprocess_fcc(df, cfg)

    # ── Severity model feature matrix ──
    logger.info("Building feature matrix for severity regression …")
    split_idx    = int(len(df) * (1 - train_cfg["test_size"]))
    X_text       = df["issue"].fillna("")
    X_struct_sev = df[feat_cfg["structural_features_severity"]]

    # Build TF-IDF (This returns Train/Test split matrices)
    X_train_sev, X_test_sev, tfidf_sev = build_tfidf_hybrid(
        X_train_text=X_text.iloc[:split_idx],
        X_test_text =X_text.iloc[split_idx:],
        X_train_struct=X_struct_sev.iloc[:split_idx],
        X_test_struct =X_struct_sev.iloc[split_idx:],
        max_features=feat_cfg["tfidf_max_features_severity"],
    )

    # FIX: Use vstack to combine Train and Test rows (Vertical)
    # This creates the full feature set for the entire 'df'
    X_combined_sev = vstack([X_train_sev, X_test_sev])
    
    logger.info(f"Severity Matrix Shape: {X_combined_sev.shape}")
    save_artifact(tfidf_sev, paths["saved_tfidf"])

    # Stage 5 – Severity model
    logger.info("═══ Stage 5: Train severity regression model ═══")
    # Passing the combined matrix and the full target column
    rf_model, rf_metrics, _ = run_severity_model(X_combined_sev, df["severity_score"], cfg)

    # Save feature importance
    try:
        from src.model import get_feature_importance
        tfidf_names  = list(tfidf_sev.get_feature_names_out())
        struct_names = feat_cfg["structural_features_severity"]
        all_names    = tfidf_names + struct_names
        fi_df = get_feature_importance(rf_model, all_names, top_n=15)
        fi_df.to_csv("models/rf_feature_importance.csv", index=False)
        logger.info("Saved feature importance → models/rf_feature_importance.csv")
    except Exception as e:
        logger.warning("Could not save feature importance: %s", e)

    # ── Pain level model feature matrix ──
    logger.info("Building feature matrix for pain-level classifier …")
    X_struct_pain = df[feat_cfg["structural_features_pain"]]

    X_train_pain, X_test_pain, tfidf_pain = build_tfidf_hybrid(
        X_train_text=X_text.iloc[:split_idx],
        X_test_text =X_text.iloc[split_idx:],
        X_train_struct=X_struct_pain.iloc[:split_idx],
        X_test_struct =X_struct_pain.iloc[split_idx:],
        max_features=feat_cfg["tfidf_max_features_pain"],
        ngram_range=tuple(feat_cfg["tfidf_ngram_range"]),
        stop_words=feat_cfg["tfidf_stop_words"],
    )

    # FIX: Again, use vstack for the pain-level feature matrix
    X_combined_pain = vstack([X_train_pain, X_test_pain])
    logger.info(f"Pain Matrix Shape: {X_combined_pain.shape}")

    # Stage 6 – Pain level classifier
    logger.info("═══ Stage 6: Train pain-level XGBoost classifier ═══")
    run_pain_level_model(X_combined_pain, df["pain_level"], cfg)

    return df

def run_comments_stages(cfg: dict) -> pd.DataFrame:
    """Stages 2, 4, 7 — YouTube load → NLP preprocess → KMeans clustering."""
    paths    = cfg["paths"]
    feat_cfg = cfg["features"]
    km_cfg   = cfg["models"]["kmeans_topics"]

    # Stage 2 – Load comments
    logger.info("═══ Stage 2: Load YouTube comments ═══")
    comments = run_comments_pipeline(cfg)

    # Stage 4 – NLP preprocessing
    logger.info("═══ Stage 4: NLP preprocessing for comments ═══")
    comments = preprocess_comments(comments, cfg)

    # Stage 7 – KMeans clustering
    logger.info("═══ Stage 7: KMeans topic clustering ═══")
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_comments = TfidfVectorizer(
        max_features=feat_cfg["tfidf_max_features_severity"]
    )
    tfidf_matrix = tfidf_comments.fit_transform(
        comments["comment_text"].fillna("")
    )

    km = train_kmeans(tfidf_matrix, **km_cfg)
    comments["topic_cluster"] = km.labels_

    # v2 — extract and save topic labels so clusters are interpretable
    topics = get_kmeans_topic_labels(km, tfidf_comments, top_n=10)
    save_topic_labels(topics, path="models/kmeans_topic_labels.json")

    save_artifact(km, "models/kmeans_topics.pkl")
    save_comments(comments, paths["comments_output"])
    return comments


def run_summary_stage(df: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Stage 8 — merge FCC and YouTube summaries."""
    logger.info("═══ Stage 8: Build cross-source summary ═══")
    fcc_sum = build_fcc_summary(df)
    yt_sum  = build_youtube_summary(comments)
    final   = merge_summaries(fcc_sum, yt_sum)
    logger.info("Final summary shape: %s", final.shape)
    logger.info("\n%s", final.to_string(index=False))
    return final


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Telecom Complaint Intelligence Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "fcc", "comments"],
        default="all",
        help="Which pipeline branch to run (default: all)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level: DEBUG | INFO | WARNING (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg  = load_config(args.config)

    # Ensure all output directories exist before any stage runs
    ensure_dirs(["data/raw", "data/processed", "models"])

    df = comments = None

    if args.stage in ("all", "fcc"):
        df = run_fcc_stages(cfg)

    if args.stage in ("all", "comments"):
        comments = run_comments_stages(cfg)

    if args.stage == "all" and df is not None and comments is not None:
        run_summary_stage(df, comments)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
