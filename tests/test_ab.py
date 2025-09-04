import pandas as pd

from src.stats.ab import conversion_summary


def test_conversion_summary_smoke():
    df = pd.DataFrame({
        "user_id": [1,2,3,4],
        "variant": ["A","A","B","B"],
        "converted": [0,1,0,1],
        "revenue": [0,0,0,0],
    })
    res = conversion_summary(df)
    assert set(["nA","nB","conv_A","conv_B"]).issubset(res.keys())
