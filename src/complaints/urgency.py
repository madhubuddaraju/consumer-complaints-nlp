"""
Urgency triage: flag complaints describing serious, time-sensitive harm, and check whether companies treat
them any differently.

A transparent phrase lexicon is used on purpose: triage rules must be auditable, and each flag shows the
phrase that triggered it. Categories follow harms the CFPB itself prioritises (CFPB tags already mark
servicemembers and older Americans; they are combined with text signals here).
"""

from __future__ import annotations

import re

import pandas as pd

CATEGORIES: dict[str, list[str]] = {
    "Losing home": [r"foreclos\w*", r"evict\w*", r"sheriff sale", r"lose (?:my|our) home", r"homeless"],
    "Fraud or identity theft": [r"identity theft", r"stolen identity", r"\bscam\w*", r"fraudulent(?:ly)? (?:charge|account|transaction)s?",
                                r"unauthori[sz]ed (?:charge|transaction|withdrawal|account)s?"],
    "Lost access to money": [r"account (?:was )?(?:frozen|closed|locked|restricted)", r"can ?not access (?:my )?(?:funds|money|account)",
                             r"funds (?:are )?(?:on hold|being held)", r"garnish\w*", r"levy"],
    "Vehicle or property seized": [r"repossess\w*", r"\brepo\b", r"towed"],
    "Severe hardship": [r"bankrupt\w*", r"medical (?:bills?|debt|emergency)", r"cancer", r"disabilit\w*", r"suicid\w*",
                        r"can'?t (?:afford|pay) (?:rent|food|bills)"],
}
_COMPILED = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in CATEGORIES.items()}


def flag(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for cat, rx in _COMPILED.items():
        out[cat] = df.narrative.fillna("").str.contains(rx)
    tags = df.tags.fillna("")
    out["Servicemember"] = tags.str.contains("Servicemember")
    out["Older American"] = tags.str.contains("Older American")
    out["urgent"] = out[list(CATEGORIES)].any(axis=1)
    return out


def matched_phrase(text: str) -> str | None:
    for cat, rx in _COMPILED.items():
        m = rx.search(text or "")
        if m:
            return f"{cat}: “{m.group(0)}”"
    return None
