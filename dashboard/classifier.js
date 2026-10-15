// In-browser replica of the exported scikit-learn TF-IDF + logistic regression model.
// Mirrors: lowercase -> token_pattern \b[a-z][a-z]+\b -> drop English stop words -> word n-grams
//          -> sublinear tf (1 + ln tf) -> x idf -> L2 normalise -> linear scores -> softmax.
function makeClassifier(M) {
  const index = new Map(M.vocab.map((t, i) => [t, i]));
  const stop = new Set(M.stop);
  const [nmin, nmax] = M.ngram;
  // same normalisation as complaints.clean.clean_text: CFPB redactions -> placeholder tokens
  function clean(text) {
    return text.replace(/\b(?:XX|\d{1,2})\/(?:XX|\d{1,2})\/(?:XXXX|\d{2,4})\b/g, " datetoken ")
      .replace(/\{\$?[\d,]+(?:\.\d+)?\}/g, " moneytoken ").replace(/\bX{2,}\b/g, " ");
  }
  function features(text) {
    text = clean(text);
    const toks = (text.toLowerCase().match(/\b[a-z][a-z]+\b/g) || []).filter((t) => !stop.has(t));
    const tf = new Map();
    for (let n = nmin; n <= nmax; n++) {
      for (let i = 0; i + n <= toks.length; i++) {
        const g = toks.slice(i, i + n).join(" ");
        const j = index.get(g);
        if (j !== undefined) tf.set(j, (tf.get(j) || 0) + 1);
      }
    }
    let norm = 0;
    const w = new Map();
    for (const [j, c] of tf) { const v = (1 + Math.log(c)) * M.idf[j]; w.set(j, v); norm += v * v; }
    norm = Math.sqrt(norm) || 1;
    for (const [j, v] of w) w.set(j, v / norm);
    return w;
  }
  function predict(text) {
    const x = features(text);
    const scores = M.intercept.slice();
    const contrib = M.classes.map(() => []);
    for (const [j, v] of x) for (const [k, c] of M.coef[j]) { scores[k] += v * c; contrib[k].push([M.vocab[j], v * c]); }
    const mx = Math.max(...scores);
    const e = scores.map((s) => Math.exp(s - mx));
    const z = e.reduce((a, b) => a + b, 0);
    const probs = e.map((v) => v / z);
    const best = probs.indexOf(Math.max(...probs));
    return { probs, best, classes: M.classes, terms: contrib[best].sort((a, b) => b[1] - a[1]), nTerms: x.size };
  }
  return { predict, features };
}
if (typeof module !== "undefined") module.exports = { makeClassifier };
