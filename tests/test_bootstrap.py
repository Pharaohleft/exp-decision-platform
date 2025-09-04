import pandas as pd

from src.stats.fastboot import fast_conv_ci


def test_fast_conv_ci_runs():
    df = pd.DataFrame({
        "user_id": [1,1,2,2],
        "variant": ["A","A","B","B"],
        "converted": [0,1,0,1],
        "revenue": [0,0,0,0],
    })
    user = df.groupby(["user_id","variant"], as_index=False).agg(converted=("converted","max"))
    lo, hi = fast_conv_ci(user_df=user, B=100, seed=0)
    assert lo <= hi
