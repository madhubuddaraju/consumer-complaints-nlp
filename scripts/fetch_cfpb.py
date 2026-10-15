"""
Download the CFPB Consumer Complaint Database and keep complaints that include a consumer narrative.

    python scripts/fetch_cfpb.py --months 24 --per-month 3000 --out data/complaints.csv.gz

The full export is large (several GB unzipped), so it is streamed in chunks and filtered on the fly.
A stratified sample per month keeps the working set small enough to commit and train on a laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
KEEP = ["Complaint ID", "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
        "Consumer complaint narrative", "Company", "State", "Tags", "Submitted via",
        "Company response to consumer", "Timely response?", "Company public response"]


def download(dest: Path) -> Path:
    headers = {"User-Agent": "consumer-complaints-nlp research project (github.com/madhubuddaraju/consumer-complaints-nlp)"}
    with requests.get(URL, stream=True, timeout=600, headers=headers) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--per-month", type=int, default=3000)
    ap.add_argument("--out", default="data/complaints.csv.gz")
    ap.add_argument("--zip", default="/tmp/complaints.csv.zip")
    a = ap.parse_args()

    zpath = Path(a.zip)
    if not zpath.exists():
        download(zpath)
    print(f"zip: {zpath.stat().st_size / 1e6:.0f} MB", flush=True)
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        print("reading", name, flush=True)
        frames = []
        # filter while streaming so memory stays small: only narratives inside the window (plus a month of slack)
        floor = pd.Timestamp.now().normalize() - pd.DateOffset(months=a.months + 1)
        with z.open(name) as fh:
            reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), dtype=str, chunksize=100_000,
                                 on_bad_lines="skip")
            for i, chunk in enumerate(reader):
                if i == 0:
                    print("columns:", list(chunk.columns), flush=True)
                chunk = chunk[[c for c in KEEP if c in chunk.columns]]
                chunk = chunk[chunk["Consumer complaint narrative"].notna()]
                d = pd.to_datetime(chunk["Date received"], errors="coerce")
                frames.append(chunk[d >= floor])
    df = pd.concat(frames, ignore_index=True)
    df["Date received"] = pd.to_datetime(df["Date received"])
    latest = df["Date received"].max()
    start = (latest - pd.DateOffset(months=a.months)).normalize()
    df = df[df["Date received"] > start]
    df["month"] = df["Date received"].dt.to_period("M")
    sample = (df.groupby("month", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), a.per_month), random_state=42))
                .drop(columns="month").sort_values("Date received"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, compression="gzip")
    meta = {"source": URL, "zip_sha256": sha, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "narratives_in_window": int(len(df)), "sampled": int(len(sample)),
            "window_start": str(start.date()), "window_end": str(latest.date()), "per_month": a.per_month}
    out.with_name("complaints_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
