"""
Export the TF-IDF + logistic regression model to compact JSON so the dashboard can classify complaints in the
browser, with no server. `dashboard/classifier.js` re-implements scikit-learn's tokeniser, stop-word removal,
n-grams, sublinear TF, IDF weighting and L2 normalisation; a test checks the two agree.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline


def export(pipe: Pipeline, decimals: int = 3) -> dict:
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    vocab = tfidf.get_feature_names_out().tolist()
    coef = np.round(clf.coef_, decimals)
    # store coefficients column-wise and sparsely: term index -> [class, weight] pairs with non-zero weight
    terms = []
    for j in range(coef.shape[1]):
        nz = np.nonzero(coef[:, j])[0]
        terms.append([[int(i), float(coef[i, j])] for i in nz])
    return {
        "classes": clf.classes_.tolist(),
        "intercept": np.round(clf.intercept_, 4).tolist(),
        "vocab": vocab,
        "idf": np.round(tfidf.idf_, 4).tolist(),
        "coef": terms,
        "stop": sorted(ENGLISH_STOP_WORDS),
        "ngram": list(tfidf.ngram_range),
    }
