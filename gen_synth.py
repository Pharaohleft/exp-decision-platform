import numpy as np
import pandas as pd

rng = np.random.default_rng(7); N=5000
variant = rng.choice(["A","B"], size=N)
pre = rng.gamma(2.0,2.0,N)
p = 0.10 + 0.015*(variant=="B")
conv = (rng.random(N)<p).astype(int)
rev = (conv * (np.maximum(0, rng.normal(10,4,N)) + 0.3*pre + 1.0*(variant=="B"))).clip(0)
df = pd.DataFrame({"user_id":np.arange(N),"variant":variant,"converted":conv,
                   "revenue":np.round(rev,2),"pre_metric":np.round(pre,2)})
df.to_csv(r"data\synth_experiment.csv", index=False)
print("Wrote data\\synth_experiment.csv", len(df))
