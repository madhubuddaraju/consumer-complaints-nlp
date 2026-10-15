"""
Baseline product classifier: TF-IDF word n-grams + class-weighted logistic regression.

Deliberately simple and transparent: every prediction can be explained by the words that drove it, and
the whole model exports to a few megabytes of JSON so it runs in the browser on the dashboard.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from .config import SEED

log = logging.getLogger(__name__)
MIN_CLASS = 100  # products with fewer training examples are folded into "Other"


def make_pipeline(c: float = 4.0, max_features: int = 30_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=max_features,
                                  sublinear_tf=True, strip_accents=None, token_pattern=r"(?u)\b[a-z][a-z]+\b",
                                  stop_words="english")),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)),
    ])


def labels(df: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    counts = df.loc[train_mask, "product"].value_counts()
    keep = set(counts[counts >= MIN_CLASS].index)
    return df["product"].where(df["product"].isin(keep), "Other")


def evaluate(y_true, y_pred, classes: list[str]) -> dict:
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"],
                          "support": int(rep[c]["support"])} for c in classes},
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def tune_and_fit(train: pd.DataFrame, valid: pd.DataFrame, y_tr, y_va, grid=(1.0, 4.0, 10.0)) -> tuple[Pipeline, dict]:
    best, best_f1, results = None, -1.0, {}
    for c in grid:
        pipe = make_pipeline(c).fit(train.text, y_tr)
        f1 = f1_score(y_va, pipe.predict(valid.text), average="macro")
        results[c] = round(float(f1), 4)
        log.info("logreg C=%-5g valid macro-F1 %.4f", c, f1)
        if f1 > best_f1:
            best, best_f1, best_c = pipe, f1, c
    return best, {"C": best_c, "grid": results}


def top_terms(pipe: Pipeline, n: int = 12) -> dict[str, list[str]]:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_
    return {cls: vocab[np.argsort(-coef[i])[:n]].tolist() for i, cls in enumerate(pipe.named_steps["clf"].classes_)}
