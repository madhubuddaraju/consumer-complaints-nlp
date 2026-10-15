import numpy as np
import pandas as pd

from complaints import clean, templates, topics, urgency


def test_clean_text_replaces_scrubbing_artifacts():
    s = "On XX/XX/2023 XXXX charged me {$1,250.00} without   permission"
    assert clean.clean_text(s) == "On datetoken charged me moneytoken without permission"


def test_duplicates_removed_and_products_mapped(sample):
    df, stats = sample
    assert not df.text.duplicated().any()
    assert "Credit reporting, credit repair services, or other personal consumer reports" not in set(df["product"])
    assert stats["kept"] == len(df)


def test_temporal_split_keeps_order():
    df = pd.DataFrame({"month": [f"2023-{m:02d}" for m in range(1, 13)]})
    split = clean.temporal_split(df)
    assert list(split[-3:]) == ["test"] * 3 and list(split[-5:-3]) == ["valid"] * 2
    assert set(split[:7]) == {"train"}


def test_urgency_flags_with_phrase():
    df = pd.DataFrame({"narrative": ["They started foreclosure on my house", "I was charged a late fee",
                                     "This is identity theft, I never opened it"], "tags": [None, "Servicemember", None]})
    f = urgency.flag(df)
    assert f["Losing home"].tolist() == [True, False, False]
    assert f["Fraud or identity theft"].tolist() == [False, False, True]
    assert f["Servicemember"].tolist() == [False, True, False]
    assert urgency.matched_phrase("my car was repossessed").startswith("Vehicle or property seized")


def test_template_detection_finds_near_copies():
    letter = ("I am writing to dispute the following accounts on my credit report which are inaccurate and "
              "unverified under the fair credit reporting act and must be removed within thirty days of this letter")
    texts = pd.Series([letter, letter.replace("thirty", "30"),
                       "My checking account was closed without any warning and the bank kept my paycheck deposit for weeks",
                       "The dealer added fees to my car loan that I never agreed to and refuses to refund them"])
    sim = templates.nearest_similarity(texts)
    assert sim[0] >= templates.THRESHOLD and sim[1] >= templates.THRESHOLD
    assert sim[2] < templates.THRESHOLD and sim[3] < templates.THRESHOLD


def test_emerging_ranks_growing_theme_first():
    months = [f"2023-{m:02d}" for m in range(1, 13)]
    noise = np.tile([0.01, -0.01], 6)
    shares = pd.DataFrame({0: np.r_[np.full(9, 0.10), [0.20, 0.22, 0.24]] + noise / 10, 1: np.full(12, 0.3) + noise}, index=months)
    out = topics.emerging(shares)
    assert out.index[0] == 0 and out.loc[0, "ratio"] > 2
