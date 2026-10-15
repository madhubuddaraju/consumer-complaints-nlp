"""
Template-letter detection.

A large share of credit-reporting complaints are copies of a few form letters (often circulated by
credit-repair services) with small edits. Exact copies are removed during cleaning; this module finds
near-copies: complaints whose word-trigram TF-IDF vector has cosine similarity >= THRESHOLD with some other
complaint. Computed in blocks with sparse matrix products, so 50k+ documents take well under a minute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

THRESHOLD = 0.6


def nearest_similarity(texts: pd.Series, block: int = 1000) -> np.ndarray:
    # trigrams shared by more than 2% of complaints ("on my credit") say nothing about copying and make
    # the similarity matrix dense, so they are dropped
    vec = TfidfVectorizer(ngram_range=(3, 3), min_df=2, max_df=0.02, sublinear_tf=True, max_features=300_000,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b")
    x = vec.fit_transform(texts)
    xt = x.T.tocsr()
    best = np.zeros(x.shape[0])
    for start in range(0, x.shape[0], block):
        s = (x[start:start + block] @ xt).tocsr()
        n = s.shape[0]
        self_sim = sp.csr_matrix((s.diagonal(k=start), (np.arange(n), np.arange(start, start + n))), shape=s.shape)
        s = s - self_sim  # ignore similarity with itself
        best[start:start + n] = s.max(axis=1).toarray().ravel()
    return best


def summarise(df: pd.DataFrame, sim: np.ndarray) -> dict:
    t = df.assign(templated=sim >= THRESHOLD)
    by_product = t.groupby("product").templated.agg(["mean", "size"]).sort_values("mean", ascending=False)
    by_month = t.groupby("month").templated.mean()
    return {
        "threshold": THRESHOLD,
        "share": float(t.templated.mean()),
        "by_product": [{"product": p, "share": float(r["mean"]), "complaints": int(r["size"])} for p, r in by_product.iterrows()],
        "by_month": {"months": by_month.index.tolist(), "share": by_month.round(4).tolist()},
    }
