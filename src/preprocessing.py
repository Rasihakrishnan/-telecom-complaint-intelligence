"""
preprocessing.py
────────────────
All feature engineering, text vectorisation, and scaling logic.
Operates on the cleaned DataFrames produced by data_loader.py.

Fixes applied (v2):
  - label_encode_columns: each column gets its OWN LabelEncoder instance (bug fix)
  - add_class_balance_check: logs class distribution so imbalance is visible
  - add_youtube_eda_features: word count, comment length for comment EDA
"""

import logging
import numpy as np
import pandas as pd
import nltk
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from textblob import TextBlob

logger = logging.getLogger(__name__)

# ── NLTK resources (downloaded once) ──────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers/punkt",            "punkt"),
    ("tokenizers/punkt_tab",        "punkt_tab"),
    ("corpora/stopwords",           "stopwords"),
    ("corpora/wordnet",             "wordnet"),
    ("corpora/omw-1.4",             "omw-1.4"),
    ("sentiment/vader_lexicon.zip", "vader_lexicon"),
]


def ensure_nltk_resources() -> None:
    for resource_path, resource_id in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource_id)
            nltk.download(resource_id, quiet=True)


# ──────────────────────────────────────────────────────────────
#  FCC Complaint Features
# ──────────────────────────────────────────────────────────────

def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = "date_created",
) -> pd.DataFrame:
    """Extract year / month / day / weekday from a datetime column."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"]    = df[date_col].dt.year
    df["month"]   = df[date_col].dt.month
    df["day"]     = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    logger.debug("Added temporal features from '%s'", date_col)
    return df


def add_severity_score(
    df: pd.DataFrame,
    severity_map: dict,
) -> pd.DataFrame:
    """Map the *issue* column to a numeric severity score (1–5)."""
    df["severity_score"] = df["issue"].map(severity_map)
    missing = df["severity_score"].isna().sum()
    if missing:
        logger.warning("%d rows have an unmapped severity; filling with 1", missing)
        df["severity_score"] = df["severity_score"].fillna(1)
    return df


def label_encode_columns(
    df: pd.DataFrame,
    columns: list,
) -> tuple:
    """
    Label-encode each column in *columns*.

    FIX v2: Each column gets its OWN LabelEncoder instance.
    Previously a single `le` was reused, making inverse-transform
    impossible after the first column.

    Returns (df, encoders_dict) where encoders_dict maps
    column_name → fitted LabelEncoder.
    """
    encoders: dict = {}
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found; skipping label encoding", col)
            continue
        df[col] = df[col].astype(str)
        le = LabelEncoder()                      # ← fresh instance per column
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le
        logger.debug(
            "Label-encoded '%s' → '%s_enc'  (%d unique values)",
            col, col, len(le.classes_),
        )
    return df, encoders


def add_text_features(
    df: pd.DataFrame,
    urgent_pattern: str,
    text_col: str = "issue",
) -> pd.DataFrame:
    """Add word_count, text_length, TextBlob sentiment, and urgent-word flag."""
    df[text_col] = df[text_col].astype(str).str.lower()
    df["text_length"]      = df[text_col].apply(len)
    df["word_count"]       = df[text_col].apply(lambda x: len(x.split()))
    df["sentiment"]        = df[text_col].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["has_urgent_words"] = df[text_col].str.contains(
        urgent_pattern, case=False, na=False
    ).astype(int)
    return df


def build_pain_score(
    df: pd.DataFrame,
    weights: dict,
    noise_std: float,
    quantiles: int,
    labels: list,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Construct the synthetic *pain_level* classification target.

    Score = weighted combination of method_enc, log(word_count),
    log(text_length) + Gaussian noise, then quantile-binned.
    """
    np.random.seed(random_seed)
    df["pain_score"] = (
        df["method_enc"] * weights["method_enc"]
        + np.log1p(df["word_count"])  * weights["word_count"]
        + np.log1p(df["text_length"]) * weights["text_length"]
    )
    df["pain_score"] += np.random.normal(0, noise_std, len(df))
    df["pain_level"]  = pd.qcut(df["pain_score"], q=quantiles, labels=labels)
    df.drop(columns=["pain_score"], inplace=True)
    return df


def check_class_balance(y: pd.Series, label: str = "target") -> None:
    """
    Log the class distribution so imbalance is always visible.
    NEW in v2 — addresses the review finding that balance was never checked.
    """
    counts = y.value_counts()
    pct    = (counts / len(y) * 100).round(1)
    logger.info("Class balance for '%s':", label)
    for cls, cnt in counts.items():
        logger.info("  %-12s %7d  (%s%%)", cls, cnt, pct[cls])


# ──────────────────────────────────────────────────────────────
#  Vectorisation helpers
# ──────────────────────────────────────────────────────────────

def build_tfidf_hybrid(
    X_train_text:   pd.Series,
    X_test_text:    pd.Series,
    X_train_struct: pd.DataFrame,
    X_test_struct:  pd.DataFrame,
    max_features:   int,
    ngram_range:    tuple = (1, 2),
    stop_words:     str   = "english",
) -> tuple:
    """
    Fit a TF-IDF vectoriser on *X_train_text* and horizontally stack
    with the structural feature arrays.

    Returns (X_train_combined, X_test_combined, fitted_tfidf).
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text.fillna(""))
    X_test_tfidf  = tfidf.transform(X_test_text.fillna(""))

    X_train_combined = hstack([X_train_tfidf, X_train_struct.values])
    X_test_combined  = hstack([X_test_tfidf,  X_test_struct.values])
    return X_train_combined, X_test_combined, tfidf


def scale_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """Fit StandardScaler on train, transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────────────────────
#  YouTube Comment NLP
# ──────────────────────────────────────────────────────────────

def tokenize_and_clean(
    comments: pd.DataFrame,
    text_col: str = "comment_text",
) -> pd.DataFrame:
    """Tokenise, remove stop-words, and lemmatise comments."""
    from nltk.tokenize import word_tokenize
    from nltk.corpus   import stopwords
    from nltk.stem     import WordNetLemmatizer

    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    comments["tokens"] = comments[text_col].apply(
        lambda x: word_tokenize(str(x))
    )
    comments["tokens"] = comments["tokens"].apply(
        lambda toks: [w for w in toks if w.lower() not in stop_words]
    )
    comments["tokens"] = comments["tokens"].apply(
        lambda toks: [lemmatizer.lemmatize(w) for w in toks]
    )
    return comments


def add_youtube_eda_features(
    comments: pd.DataFrame,
    text_col: str = "comment_text",
) -> pd.DataFrame:
    """
    Add comment_length and comment_word_count for EDA.
    NEW in v2 — review noted YouTube EDA was missing.
    """
    comments["comment_length"]     = comments[text_col].astype(str).apply(len)
    comments["comment_word_count"] = comments[text_col].astype(str).apply(
        lambda x: len(x.split())
    )
    logger.debug("Added YouTube EDA features: comment_length, comment_word_count")
    return comments


def add_vader_sentiment(
    comments: pd.DataFrame,
    text_col: str = "comment_text",
) -> pd.DataFrame:
    """Compute VADER compound scores."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    ensure_nltk_resources()
    sia = SentimentIntensityAnalyzer()

    comments["sentiment_score"] = comments[text_col].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    return comments


def assign_sentiment_labels(
    comments: pd.DataFrame,
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
) -> pd.DataFrame:
    def _label(score: float) -> str:
        if score >= positive_threshold:
            return "Positive"
        elif score <= negative_threshold:
            return "Negative"
        return "Neutral"

    comments["sentiment_label"] = comments["sentiment_score"].apply(_label)
    return comments


def detect_issue_type(
    comments: pd.DataFrame,
    issue_keywords: dict,
    text_col: str = "comment_text",
) -> pd.DataFrame:
    """Rule-based issue type detection from keyword dictionaries."""
    def _detect(text: str) -> str:
        text = text.lower()
        for issue, words in issue_keywords.items():
            if any(w in text for w in words):
                return issue
        return "Other"

    comments["issue_type"] = comments[text_col].apply(_detect)
    return comments


def add_keyword_frequencies(
    comments: pd.DataFrame,
    keywords: list,
    text_col: str = "comment_text",
) -> pd.DataFrame:
    """Count occurrences of each keyword as a separate feature column."""
    for word in keywords:
        comments[f"{word}_freq"] = (
            comments[text_col].str.lower().str.count(word)
        )
    return comments


def remap_issue_types(
    comments: pd.DataFrame,
    issue_map: dict,
) -> pd.DataFrame:
    """Align YouTube issue types to FCC issue categories via *issue_map*."""
    comments["issue_type"] = comments["issue_type"].map(issue_map)
    return comments


# ──────────────────────────────────────────────────────────────
#  Convenience wrappers used by main.py
# ──────────────────────────────────────────────────────────────

def preprocess_fcc(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Full FCC feature-engineering pipeline."""
    feat_cfg   = cfg["features"]
    sev_map    = cfg["severity_map"]
    train_cfg  = cfg["training"]
    urgent_pat = feat_cfg["urgent_words_pattern"]

    df = add_temporal_features(df, date_col="date_created")
    df = add_severity_score(df, sev_map)
    df, _ = label_encode_columns(df, feat_cfg["label_encode_columns"])
    df = add_text_features(df, urgent_pat)
    df = build_pain_score(
        df,
        weights     = train_cfg["pain_score_weights"],
        noise_std   = train_cfg["pain_score_noise_std"],
        quantiles   = train_cfg["pain_level_quantiles"],
        labels      = train_cfg["pain_level_labels"],
        random_seed = train_cfg["random_state"],
    )

    # v2 — always log class balance after target is built
    check_class_balance(df["pain_level"], label="pain_level")
    check_class_balance(df["severity_score"].astype(str), label="severity_score")

    return df


def preprocess_comments(comments: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Full NLP pipeline for YouTube comments."""
    feat_cfg = cfg["features"]
    nlp_cfg  = cfg["nlp"]

    comments = tokenize_and_clean(comments)
    comments = add_youtube_eda_features(comments)          # v2 addition
    comments = add_vader_sentiment(comments)
    comments = assign_sentiment_labels(
        comments,
        positive_threshold=nlp_cfg["sentiment_positive_threshold"],
        negative_threshold=nlp_cfg["sentiment_negative_threshold"],
    )
    comments = detect_issue_type(comments, nlp_cfg["issue_keywords"])
    comments = add_keyword_frequencies(comments, feat_cfg["keyword_frequencies"])
    comments = remap_issue_types(comments, cfg["issue_map"])

    # v2 — log sentiment balance
    check_class_balance(comments["sentiment_label"], label="sentiment_label")

    return comments
