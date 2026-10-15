"""
End-to-end analysis.

    python -m complaints          # clean -> split -> baseline classifier -> themes -> urgency -> reports + web model
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import classify, clean, templates, topics, urgency, web_model
from .config import ARTIFACTS, META, REPORTS

log = logging.getLogger("complaints")


def run() -> dict:
    t0 = time.time()
    df, stats = clean.load()
    df["split"] = clean.temporal_split(df)
    y = classify.labels(df, df.split == "train")
    classes = sorted(y.unique())
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    log.info("train %d (%s to %s) / valid %d / test %d (%s to %s)", len(tr), tr.month.min(), tr.month.max(),
             len(va), len(te), te.month.min(), te.month.max())

    # 1) classifier: tune on validation, refit on train+valid, evaluate once on the latest months
    _, tuning = classify.tune_and_fit(tr, va, y[tr.index], y[va.index])
    trva = pd.concat([tr, va])
    pipe = classify.make_pipeline(tuning["C"]).fit(trva.text, y[trva.index])
    pred = pipe.predict(te.text)
    proba = pipe.predict_proba(te.text)
    base = classify.evaluate(y[te.index], pred, classes)
    majority = classify.evaluate(y[te.index], np.full(len(te), y[trva.index].mode()[0]), classes)
    conf = proba.max(axis=1)
    coverage = {f"{q:.1f}": {"share": float((conf >= q).mean()),
                             "accuracy": float((pred[conf >= q] == y[te.index].to_numpy()[conf >= q]).mean())}
                for q in (0.5, 0.7, 0.9)}
    errors = te.assign(true=y[te.index], pred=pred, conf=conf)
    errors = errors[errors.true != errors.pred].sort_values("conf", ascending=False)

    # 2) template letters: near-copies of form letters (exact copies were already removed in cleaning)
    sim = templates.nearest_similarity(df.text)
    df["templated"] = sim >= templates.THRESHOLD
    tmpl = templates.summarise(df, sim)
    tmpl["exact_duplicates_removed"] = stats["duplicates"]
    log.info("%.1f%% of complaints are near-copies of another complaint", 100 * tmpl["share"])

    # 3) themes and emerging issues, on original (non-template) narratives so form letters don't dominate
    organic = df[~df.templated]
    vec, nmf, w = topics.fit_topics(organic.text)
    terms = topics.describe(vec, nmf)
    shares = topics.monthly_shares(organic, w)
    emerg = topics.emerging(shares)
    dom = w.argmax(axis=1)
    theme_products = [organic["product"][dom == k].value_counts(normalize=True).head(3).round(3).to_dict()
                      for k in range(len(terms))]

    # 4) urgency triage and how companies respond
    flags = urgency.flag(df)
    df = df.join(flags)
    df["relief"] = df.response.isin(["Closed with monetary relief", "Closed with non-monetary relief"])
    urg = pd.DataFrame({cat: {"complaints": int(df[cat].sum()), "share": float(df[cat].mean()),
                              "relief_rate": float(df.loc[df[cat], "relief"].mean()) if df[cat].any() else None}
                        for cat in [*urgency.CATEGORIES, "Servicemember", "Older American"]}).T
    relief_urgent = float(df.loc[df.urgent, "relief"].mean())
    relief_other = float(df.loc[~df.urgent, "relief"].mean())

    companies = (df.groupby("company").agg(complaints=("text", "size"), relief_rate=("relief", "mean"),
                                           urgent_share=("urgent", "mean"), top_product=("product", lambda s: s.mode()[0]))
                   .query("complaints >= 150").sort_values("complaints", ascending=False))
    monthly_products = df.groupby(["month", "product"]).size().unstack(fill_value=0)

    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    meta = json.loads(META.read_text())
    result = {
        "data": {**meta, **stats, "months": [df.month.min(), df.month.max()]},
        "split": {s: {"rows": int((df.split == s).sum()), "from": df.month[df.split == s].min(), "to": df.month[df.split == s].max()}
                  for s in ("train", "valid", "test")},
        "classes": classes,
        "class_counts_test": y[te.index].value_counts().to_dict(),
        "baseline": {**base, "tuning": tuning, "top_terms": classify.top_terms(pipe), "confidence_coverage": coverage},
        "majority_class": {k: majority[k] for k in ("accuracy", "macro_f1")},
        "templates": tmpl,
        "themes": [{"id": k, "terms": t, "products": theme_products[k],
                    "recent_share": float(emerg.recent_share.get(k, 0)), "baseline_share": float(emerg.baseline_share.get(k, 0)),
                    "ratio": float(emerg.ratio.get(k, np.nan)), "z": float(emerg.z.get(k, np.nan))} for k, t in enumerate(terms)],
        "theme_shares": {"months": shares.index.tolist(), "series": {str(k): shares[k].round(4).tolist() for k in shares.columns}},
        "urgency": {"by_category": urg.reset_index(names="category").to_dict("records"), "urgent_share": float(df.urgent.mean()),
                    "relief_urgent": relief_urgent, "relief_other": relief_other},
        "companies": companies.head(25).reset_index().to_dict("records"),
        "monthly_products": {"months": monthly_products.index.tolist(),
                             "series": {p: monthly_products[p].tolist() for p in monthly_products.columns}},
        "error_examples": [{"true": r.true, "pred": r.pred, "conf": round(float(r.conf), 3), "text": r.text[:400]}
                           for r in errors.head(8).itertuples()],
    }
    (REPORTS / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    (ARTIFACTS / "web_model.json").write_text(json.dumps(web_model.export(pipe), separators=(",", ":")))
    # a small sample of test complaints for the dashboard's "try an example" buttons
    ex = te.assign(true=y[te.index]).groupby("true").head(3)[["true", "narrative"]]
    (ARTIFACTS / "examples.json").write_text(json.dumps([{"product": p, "text": t[:1200]} for p, t in ex.itertuples(index=False)]))
    log.info("baseline test macro-F1 %.3f, accuracy %.3f (majority-class macro-F1 %.3f) in %.0fs",
             base["macro_f1"], base["accuracy"], majority["macro_f1"], time.time() - t0)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    run()


if __name__ == "__main__":
    main()
