import os

import pandas as pd

df = pd.read_excel('./experiments/configurations.xlsx')
for i in df.index:
    r, n, p, v, alg = df.iloc[i]
    print(f'Running exp No:{i}, r:{r}, n:{n}, p:{p}, v:{v}, alg:{alg}')
    # # if alg == 'EnKF-MC':
    # #     os.system(
    # #         f"python experiments/EnKF-MC/main.py --variable={v} --alpha={float(_)} --n_members={n} --p_obs={int(p)} --r={int(r)}")
    os.system(f"python experiments/EnKF-DM/main.py --variable={v} --n_members={int(n)} --p_obs={int(p)} --r={int(r)}")
