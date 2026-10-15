"""
Figures and the findings report (reports/findings.md) from reports/metrics.json.

    python -m complaints.report
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import REPORTS  # noqa: E402

log = logging.getLogger(__name__)
FIG = REPORTS / "figures"
BASE, BERT, ACCENT, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
})


def theme_label(terms: list[str], n: int = 3) -> str:
    out = []
    for t in terms:
        if not any(t in o or o in t for o in out):
            out.append(t)
        if len(out) == n:
            break
    return " · ".join(out)


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    return FIG / name


def fig_f1(m, bert):
    classes = m["classes"]
    base = [m["baseline"]["per_class"][c]["f1"] for c in classes]
    order = np.argsort(base)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(classes))
    h = 0.38 if bert else 0.6
    ax.barh(y + (h / 2 if bert else 0), np.array(base)[order], height=h, color=BASE, label=f"TF-IDF + logistic regression ({m['baseline']['macro_f1']:.2f})")
    if bert:
        b = [bert["per_class"][c]["f1"] for c in classes]
        ax.barh(y - h / 2, np.array(b)[order], height=h, color=BERT, label=f"Fine-tuned DistilBERT ({bert['macro_f1']:.2f})")
    ax.set_yticks(y, [f"{classes[i]} (n={m['baseline']['per_class'][classes[i]]['support']:,})" for i in order])
    ax.set(xlim=(0, 1), xlabel="F1 score on the most recent 3 months", title="Product classification, per class")
    ax.legend(loc="lower right", title="Model (macro-F1)", fontsize=8.5, title_fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, "f1_by_class.png")


def fig_templates(t):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    bp = [r for r in t["by_product"] if r["complaints"] >= 150][::-1]
    a.barh([r["product"] for r in bp], [r["share"] * 100 for r in bp], color=BASE, height=0.6)
    for i, r in enumerate(bp):
        a.text(r["share"] * 100 + 1, i, f"{r['share']:.0%}", va="center", fontsize=8.5, color=INK2)
    a.set(title="Near-copies of another complaint", xlabel="% of complaints", xlim=(0, 70))
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0)
    months = t["by_month"]["months"]
    b.plot(range(len(months)), np.array(t["by_month"]["share"]) * 100, color=BASE, lw=2)
    b.set_xticks(range(0, len(months), 4), [months[i] for i in range(0, len(months), 4)])
    b.set(title="Share of templated complaints by month", ylabel="%", ylim=(0, 60))
    return _save(fig, "template_letters.png")


def fig_themes(m):
    th = sorted(m["themes"], key=lambda x: -x["z"])[:6]
    months = m["theme_shares"]["months"]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [BASE, BERT, ACCENT, "#eda100", "#e87ba4", "#6250d6"]
    for c, t in zip(colors, th, strict=False):
        s = np.array(m["theme_shares"]["series"][str(t["id"])]) * 100
        ax.plot(range(len(months)), s, color=c, lw=2, label=f"{theme_label(t['terms'])}  (z {t['z']:+.1f})")
    ax.set_xticks(range(0, len(months), 4), [months[i] for i in range(0, len(months), 4)])
    ax.set(title="Fastest-growing complaint themes (excluding template letters)", ylabel="% of monthly complaints")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return _save(fig, "emerging_themes.png")


def fig_urgency(u):
    rows = sorted(u["by_category"], key=lambda r: r["relief_rate"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([r["category"] for r in rows], [r["relief_rate"] * 100 for r in rows], color=ACCENT, height=0.6)
    for i, r in enumerate(rows):
        ax.text(max(r["relief_rate"], u["relief_other"]) * 100 + 0.8, i, f"{r['relief_rate']:.0%}  ({int(r['complaints']):,} complaints)", va="center", fontsize=8.5, color=INK2)
    ax.axvline(u["relief_other"] * 100, color=INK, lw=1, ls=(0, (4, 2)))
    ax.text(u["relief_other"] * 100, -0.75, f" non-urgent complaints {u['relief_other']:.0%}", fontsize=8.5, va="center")
    ax.set(title="How often companies give relief, by type of harm", xlabel="% closed with monetary or non-monetary relief",
           xlim=(0, 80), ylim=(-1.1, len(rows) - 0.4))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, "urgency_relief.png")


def build() -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    m = json.loads((REPORTS / "metrics.json").read_text())
    bpath = REPORTS / "transformer_metrics.json"
    bert = json.loads(bpath.read_text()) if bpath.exists() else None
    figs = [fig_f1(m, bert), fig_templates(m["templates"]), fig_themes(m), fig_urgency(m["urgency"])]
    b, t, u, d = m["baseline"], m["templates"], m["urgency"], m["data"]
    top = sorted(m["themes"], key=lambda x: -x["z"])[:3]
    weakest = sorted(b["per_class"].items(), key=lambda kv: kv[1]["f1"])[:2]
    cov = b["confidence_coverage"]["0.9"]
    cm = np.array(b["confusion"])
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    conf_true, conf_pred, conf_n, conf_tot = m["classes"][i], m["classes"][j], int(off[i, j]), int(cm[i].sum())
    bert_line = (f"| Fine-tuned DistilBERT ({bert['train_examples']:,} training examples, {bert['epochs']} epochs) | "
                 f"{bert['macro_f1']:.3f} | {bert['accuracy']:.3f} |" if bert else "| Fine-tuned DistilBERT | pending | pending |")
    fraud = next(r for r in u["by_category"] if r["category"] == "Fraud or identity theft")
    home = next(r for r in u["by_category"] if r["category"] == "Losing home")
    md = f"""# Findings

_Generated by `python -m complaints.report` from {d['kept']:,} unique CFPB complaint narratives
({d['months'][0]} to {d['months'][1]})._

## 1. A quarter of narratives are exact copies, and over a third of the rest are near-copies
- **{t['exact_duplicates_removed']:,} of {d['rows']:,} sampled narratives ({t['exact_duplicates_removed'] / d['rows']:.0%}) were word-for-word
  duplicates** and were removed before modeling.
- Of the remaining complaints, **{t['share']:.0%} are near-copies** of another complaint (word-trigram cosine similarity
  ≥ {t['threshold']}), rising to **{t['by_product'][0]['share']:.0%} for {t['by_product'][0]['product'].lower()}**. They read like
  form letters: {t['cites_law_templated']:.0%} cite consumer-protection law (FCRA sections, U.S.C. citations) versus
  {t['cites_law_original']:.0%} of original complaints.
- The templated share fell from {t['by_month']['share'][0]:.0%} to {t['by_month']['share'][-1]:.0%} over the period.
- Why it matters: duplicates inflate complaint counts against specific companies, and if they are split
  between train and test they inflate model accuracy.

![Template letters](figures/template_letters.png)

## 2. Text alone identifies the product for most complaints
| Model | Macro-F1 | Accuracy |
|---|---|---|
| Always predict the most common product | {m['majority_class']['macro_f1']:.3f} | {m['majority_class']['accuracy']:.3f} |
| TF-IDF + logistic regression | {b['macro_f1']:.3f} | {b['accuracy']:.3f} |
{bert_line}

- Tested on the **most recent 3 months** ({m['split']['test']['from']} to {m['split']['test']['to']}), trained on earlier
  months only. This temporal split mirrors deployment and is harsher than a random split.
- When the baseline is at least 90% confident ({cov['share']:.0%} of complaints) it is right **{cov['accuracy']:.0%}** of
  the time. That is enough to auto-route those complaints and send the rest to a person.
- Weakest classes: {weakest[0][0]} (F1 {weakest[0][1]['f1']:.2f}) and {weakest[1][0]} (F1 {weakest[1][1]['f1']:.2f}).
  The biggest single confusion is **{conf_true} predicted as {conf_pred}** ({conf_n} of {conf_tot} test complaints): a
  collection account usually shows up on a credit report, so both labels describe the same story.

![F1 by class](figures/f1_by_class.png)

## 3. Emerging themes
Topic modeling (NMF on TF-IDF of non-templated complaints) found {len(m['themes'])} themes. The fastest-growing over
the last 3 months, compared with the 9 months before:
""" + "\n".join(f"- **{theme_label(x['terms'])}**: {x['recent_share']:.1%} of complaints, {x['ratio']:.2f}x its baseline (z = {x['z']:+.1f})"
                for x in top) + f"""

![Emerging themes](figures/emerging_themes.png)

## 4. Urgent harms and how companies respond
- **{u['urgent_share']:.0%} of complaints describe a serious harm** (losing a home, fraud or identity theft, frozen
  funds, repossession, severe hardship), flagged with an auditable phrase list.
- Fraud and identity theft is the largest group ({int(fraud['complaints']):,} complaints, {fraud['share']:.0%}).
- Complaints about **losing a home got relief only {home['relief_rate']:.0%} of the time**, versus {u['relief_other']:.0%}
  for non-urgent complaints.

![Urgency](figures/urgency_relief.png)

## Data note
In August 2026 the CFPB stopped publishing consumer narratives, so the official export no longer has the text
column. This project uses an archived copy of the database taken while narratives were public (window
{d['window_start']} to {d['window_end']}), sampled at {d['per_month']:,} complaints a month.
"""
    (REPORTS / "findings.md").write_text(md)
    log.info("wrote reports/findings.md and %d figures", len(figs))
    return md


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
