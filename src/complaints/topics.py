"""
Theme discovery and emerging-issue detection.

Non-negative matrix factorisation on TF-IDF finds recurring themes that cut across the CFPB's own product
labels (for example "identity theft" appears under credit reporting, banking and cards). Each month's share
of complaints per theme is then compared with its trailing baseline to flag themes that are growing fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .config import SEED

# placeholders, contraction fragments and legal boilerplate that dominate template letters but carry no theme
DOMAIN_STOP = {"datetoken", "moneytoken", "ve", "ll", "don", "didn", "doesn", "isn", "wasn", "won", "haven", "couldn",
               "section", "sections", "usc", "states", "stated", "consumer", "accordance", "pursuant", "hereby", "regards",
               "sincerely", "thank", "please", "would", "also", "said", "told", "called", "time", "day", "days"}
STOP = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP)


def fit_topics(texts: pd.Series, k: int = 16) -> tuple[TfidfVectorizer, NMF, np.ndarray]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.3, max_features=20_000, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b", stop_words=STOP)
    x = vec.fit_transform(texts)
    nmf = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=400)
    w = nmf.fit_transform(x)
    return vec, nmf, w


def describe(vec: TfidfVectorizer, nmf: NMF, n: int = 8) -> list[list[str]]:
    terms = np.array(vec.get_feature_names_out())
    return [terms[np.argsort(-row)[:n]].tolist() for row in nmf.components_]


def monthly_shares(df: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """Share of each month's complaints whose dominant theme is k (only confident assignments count)."""
    dom = w.argmax(axis=1)
    conf = w.max(axis=1) > np.quantile(w.max(axis=1), 0.2)
    t = pd.DataFrame({"month": df.month.to_numpy(), "topic": np.where(conf, dom, -1)})
    counts = t[t.topic >= 0].groupby(["month", "topic"]).size().unstack(fill_value=0)
    return counts.div(t.groupby("month").size(), axis=0).sort_index()


def emerging(shares: pd.DataFrame, recent: int = 3, baseline: int = 9) -> pd.DataFrame:
    """Ratio and z-score of the last `recent` months' share against the preceding `baseline` months."""
    rec, base = shares.iloc[-recent:], shares.iloc[-(recent + baseline):-recent]
    mu, sd = base.mean(), base.std().replace(0, np.nan)
    out = pd.DataFrame({"recent_share": rec.mean(), "baseline_share": mu, "ratio": rec.mean() / mu,
                        "z": (rec.mean() - mu) / sd})
    return out.sort_values("z", ascending=False)
