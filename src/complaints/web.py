"""Build the static dashboard (docs/index.html) with metrics, model and examples embedded."""

import json

from .config import ARTIFACTS, DOCS, REPORTS, ROOT


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def build():
    data = {
        "metrics": _load(REPORTS / "metrics.json"),
        "bert": _load(REPORTS / "transformer_metrics.json"),
        "model": _load(ARTIFACTS / "web_model.json"),
        "examples": _load(ARTIFACTS / "examples.json"),
    }
    if data["metrics"] is None or data["model"] is None:
        raise SystemExit("run `python -m complaints` first")
    page = (ROOT / "dashboard" / "template.html").read_text()
    js = (ROOT / "dashboard" / "classifier.js").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("__CLASSIFIER_JS__", js).replace("__DATA__", payload)
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
