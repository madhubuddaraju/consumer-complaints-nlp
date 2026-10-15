import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from complaints import classify, web_model

JS = Path(__file__).resolve().parents[1] / "dashboard" / "classifier.js"


@pytest.fixture(scope="module")
def fitted(sample):
    df, _ = sample
    pipe = classify.make_pipeline(c=4.0, max_features=5000)
    pipe.named_steps["tfidf"].set_params(min_df=2)
    pipe.fit(df.text, df["product"])
    return df, pipe


def test_classifier_beats_majority(fitted):
    df, pipe = fitted
    acc = (pipe.predict(df.text) == df["product"]).mean()
    assert acc > df["product"].value_counts(normalize=True).iloc[0]


def test_evaluate_reports_every_class(fitted):
    df, pipe = fitted
    classes = sorted(df["product"].unique())
    rep = classify.evaluate(df["product"], pipe.predict(df.text), classes)
    assert set(rep["per_class"]) == set(classes) and 0 <= rep["macro_f1"] <= 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_classifier_matches_sklearn(fitted, tmp_path):
    df, pipe = fitted
    model = web_model.export(pipe, decimals=6)
    texts = df.text.head(25).tolist()
    (tmp_path / "m.json").write_text(json.dumps(model))
    (tmp_path / "t.json").write_text(json.dumps(texts))
    script = f"""
      const {{ makeClassifier }} = require({json.dumps(str(JS))});
      const fs = require("fs");
      const c = makeClassifier(JSON.parse(fs.readFileSync("{tmp_path / 'm.json'}")));
      const t = JSON.parse(fs.readFileSync("{tmp_path / 't.json'}"));
      console.log(JSON.stringify(t.map((x) => c.predict(x).probs)));
    """
    js = np.array(json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout))
    py = pipe.predict_proba(texts)
    assert np.abs(js - py).max() < 1e-3
    assert (js.argmax(1) == py.argmax(1)).all()
