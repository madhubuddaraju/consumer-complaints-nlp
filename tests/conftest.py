import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FIXTURE = Path(__file__).parent / "fixtures" / "sample.csv.gz"


@pytest.fixture(scope="session")
def sample():
    from complaints import clean
    df, stats = clean.load(FIXTURE)
    return df, stats
