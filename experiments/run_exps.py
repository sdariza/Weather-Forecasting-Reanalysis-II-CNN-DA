import os

import pandas as pd

df = pd.read_excel('./experiments/configurations.xlsx')

for i in range(len(df)):
    _, r, n, p, v, alg = df.iloc[i, :6]
    print(f'Running exp No:{i}')
    if alg == 'EnKF-MC':
        os.system(
            f"python experiments/EnKF-MC/main.py --variable={v} --alpha={float(_)} --n_members={n} --p_obs={int(p)} --r={int(r)}")
    else:
        os.system(f"python experiments/EnKF-DM/main.py --variable={v} --n_members={n} --p_obs={int(p)} --r={int(r)}")
