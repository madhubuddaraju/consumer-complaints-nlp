"""
Load and clean complaint narratives.

CFPB scrubs personal data before publishing: names and account numbers become "XXXX", dates become
"XX/XX/XXXX" and amounts are wrapped in braces. Those artifacts carry no meaning for a model, so they are
replaced with neutral placeholder tokens. Many credit-reporting complaints are near-identical template
letters; exact duplicates (after normalisation) are removed so they cannot leak between train and test.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

from .config import DATA, PRODUCT_MAP, TEST_MONTHS, VALID_MONTHS

log = logging.getLogger(__name__)

_DATE = re.compile(r"\b(?:XX|\d{1,2})/(?:XX|\d{1,2})/(?:XXXX|\d{2,4})\b")
_REDACT = re.compile(r"\bX{2,}\b")
_MONEY = re.compile(r"\{\$?([\d,]+(?:\.\d+)?)\}")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = _DATE.sub(" datetoken ", s)
    s = _MONEY.sub(" moneytoken ", s)
    s = _REDACT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def fingerprint(s: str) -> str:
    return hashlib.md5(re.sub(r"[^a-z]", "", s.lower()).encode()).hexdigest()


def load(path: Path = DATA) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype=str, parse_dates=["Date received"])
    df = raw.rename(columns={"Complaint ID": "complaint_id", "Date received": "date", "Product": "product_raw",
                             "Sub-product": "sub_product", "Issue": "issue", "Sub-issue": "sub_issue",
                             "Consumer complaint narrative": "narrative", "Company": "company", "State": "state",
                             "Tags": "tags", "Submitted via": "channel", "Company response to consumer": "response",
                             "Timely response?": "timely", "Company public response": "public_response"})
    stats = {"rows": len(df)}
    df["product"] = df.product_raw.map(PRODUCT_MAP).fillna("Other")
    df["text"] = df.narrative.fillna("").map(clean_text)
    short = df.text.str.len() < 50
    stats["too_short"] = int(short.sum())
    df = df[~short]
    df["fp"] = df.text.map(fingerprint)
    dup = df.duplicated("fp", keep="first")
    stats["duplicates"] = int(dup.sum())
    df = df[~dup].drop(columns="fp")
    df["month"] = df.date.dt.to_period("M").astype(str)
    sizes = df.month.value_counts()
    partial = sizes[sizes < 0.3 * sizes.median()].index  # e.g. a final month cut off after a few days
    stats["partial_month_rows"] = int(df.month.isin(partial).sum())
    df = df[~df.month.isin(partial)]
    df["timely"] = df.timely.eq("Yes")
    df = df.sort_values(["date", "complaint_id"]).reset_index(drop=True)
    stats["kept"] = len(df)
    log.info("loaded %d complaints (%d too short, %d duplicate texts removed)", len(df), stats["too_short"], stats["duplicates"])
    return df, stats


def temporal_split(df: pd.DataFrame) -> pd.Series:
    """train = older months, valid = next VALID_MONTHS, test = the latest TEST_MONTHS."""
    months = sorted(df.month.unique())
    test = set(months[-TEST_MONTHS:])
    valid = set(months[-(TEST_MONTHS + VALID_MONTHS):-TEST_MONTHS])
    return df.month.map(lambda m: "test" if m in test else "valid" if m in valid else "train")
