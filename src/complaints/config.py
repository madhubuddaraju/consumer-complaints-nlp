"""Paths, label taxonomy and modeling parameters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "complaints.csv.gz"
META = ROOT / "data" / "complaints_meta.json"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

SEED = 42
TEST_MONTHS = 3          # the most recent months are held out: a temporal split, like real deployment
VALID_MONTHS = 2

# CFPB renamed several products over the years; map every variant to one stable label.
PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card or prepaid card": "Credit card",
    "Credit card": "Credit card",
    "Checking or savings account": "Bank account",
    "Bank account or service": "Bank account",
    "Mortgage": "Mortgage",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Virtual currency": "Money transfer",
    "Vehicle loan or lease": "Vehicle loan",
    "Consumer Loan": "Vehicle loan",
    "Student loan": "Student loan",
    "Payday loan, title loan, or personal loan": "Personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal loan",
    "Payday loan": "Personal loan",
    "Prepaid card": "Prepaid card",
    "Debt or credit management": "Debt management",
}
