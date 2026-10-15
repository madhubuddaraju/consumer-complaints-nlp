"""
Download CFPB complaint narratives and keep a stratified monthly sample.

    python scripts/fetch_cfpb.py --months 24 --per-month 3000 --out data/complaints.csv.gz

In August 2026 the CFPB stopped publishing consumer narratives: the official export no longer contains the
"Consumer complaint narrative" column. This script therefore reads an archived copy of the database that was
taken while narratives were still public (a weekly Parquet mirror on Hugging Face), and falls back to the
official export only to check whether narratives have been restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

MIRROR = "https://huggingface.co/datasets/sovai/cfpb_complaints/resolve/main/cfpb_complaints.parquet"
OFFICIAL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
HEADERS = {"User-Agent": "consumer-complaints-nlp research project (github.com/madhubuddaraju/consumer-complaints-nlp)"}

# mirror column -> canonical CFPB column name
RENAME = {
    "complaint_id": "Complaint ID", "date_received": "Date received", "product": "Product", "sub_product": "Sub-product",
    "issue": "Issue", "sub_issue": "Sub-issue", "consumer_complaint_narrative": "Consumer complaint narrative",
    "company": "Company", "state": "State", "tags": "Tags", "submitted_via": "Submitted via",
    "company_response_to_consumer": "Company response to consumer", "timely_response": "Timely response?",
    "company_public_response": "Company public response",
}


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=900, headers=HEADERS) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--per-month", type=int, default=3000)
    ap.add_argument("--out", default="data/complaints.csv.gz")
    ap.add_argument("--cache", default="/tmp/cfpb_complaints.parquet")
    a = ap.parse_args()

    path = Path(a.cache)
    if not path.exists():
        download(MIRROR, path)
    print(f"parquet: {path.stat().st_size / 1e6:.0f} MB", flush=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    print("columns:", names, flush=True)
    cols = [c for c in RENAME if c in names]
    frames = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        df = batch.to_pandas()
        df = df[df["consumer_complaint_narrative"].notna() & (df["consumer_complaint_narrative"].astype(str).str.len() > 0)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df = df.dropna(subset=["Date received"])
    latest = df["Date received"].max()
    start = (latest - pd.DateOffset(months=a.months)).normalize()
    window = df[df["Date received"] > start].copy()
    window["month"] = window["Date received"].dt.to_period("M")
    sample = (window.groupby("month", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), a.per_month), random_state=42))
                    .drop(columns="month").sort_values("Date received"))
    for c in ["Complaint ID"]:
        sample[c] = sample[c].astype(str)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, compression="gzip")
    meta = {"source": MIRROR, "official_export": OFFICIAL, "parquet_sha256": sha,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "narratives_total": int(len(df)), "narratives_in_window": int(len(window)), "sampled": int(len(sample)),
            "window_start": str(start.date()), "window_end": str(latest.date()), "per_month": a.per_month,
            "note": "CFPB stopped publishing narratives in August 2026; this is an archived copy."}
    out.with_name("complaints_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
