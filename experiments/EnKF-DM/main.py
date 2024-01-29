"""
This code generate simulations steps for (LEnKF) Data Assimilation
process based on : Decorrelation matrix
"""
import sys

sys.path.append(r'C:\Users\Lab6k\Desktop\Weather-Forecasting-Reanalysis-II-CNN-DA')

import warnings
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from data_assimilation.commons import load_data, forecast_ensemble, load_model, observation_cov_matrix, \
    NUMBER_OF_VARIABLES
import numpy as np
import pandas as pd
import time

np.random.seed(123)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--variable', type=str, help='Variable to predict')
parser.add_argument('--n_members', type=int, help='Number of ensemble members')
parser.add_argument('--p_obs', type=int, help='% of observation')
parser.add_argument('--r', type=int, help='Influence radio')

args = parser.parse_args()

VARIABLE = args.variable
NUMBER_OF_MEMBERS = args.n_members

P = args.p_obs / 100
r = args.r

data_state, time_units, unit_data = load_data(variable=VARIABLE)

model = load_model(variable=VARIABLE)

Xb0 = np.load(
    f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}_{NUMBER_OF_MEMBERS}.npy')

M, R, _ = observation_cov_matrix(P)

NUMBER_OF_ASSIMILATION_STEPS = len(data_state[0])

Xa = np.array(Xb0.copy()).astype('float32')

err_a = [[], [], []]
err_b = [[], [], []]
print("Getting decorrelation matrix...")
L = np.load(f'./data_assimilation/decorrelation_matrices/decorrelation_r{r}.npy')
print("Data assimilation steps...")
TOTAL_TIME = 0
for day in range(NUMBER_OF_ASSIMILATION_STEPS - 1):
    for state in [0, 1, 2, 3]:
        NEXT_STATE = (state + 1) % 4
        if NEXT_STATE == 0:
            xt_ = data_state[NEXT_STATE][day + 1].flatten()
        else:
            xt_ = data_state[NEXT_STATE][day].flatten()
        Xb = forecast_ensemble(model, x_background=Xa, state_h=state, number_of_members=NUMBER_OF_MEMBERS)
        xb = np.mean(Xb, axis=0)

        # Create observation - data for the assimilation process
        H = np.random.permutation(np.arange(NUMBER_OF_VARIABLES))[
            :M]  # we observe a different part of the domain
        y = np.random.multivariate_normal(xt_[H], R)

        # Assimilation step
        start_time = time.time()
        Pb = L * np.cov(Xb.T)  # Localization method
        Ys = np.random.multivariate_normal(
            y, R, NUMBER_OF_MEMBERS).T  # synthetic observations
        Ds = Ys - Xb[:, H].T  # Synthetic innovations
        Pa = R + Pb[H, :][:, H]
        Za = np.linalg.solve(Pa, Ds)
        Dx = Pb[:, H] @ Za
        Xa = Xb + Dx.T
        end_time = time.time()
        TOTAL_TIME += (end_time - start_time)
        xa = np.mean(Xa, axis=0)
        err_b[0].append(mean_absolute_error(xt_, xb))
        err_b[1].append(mean_squared_error(xt_, xb) ** 0.5)
        err_b[2].append(mean_absolute_percentage_error(xt_, xb))
        err_a[0].append(mean_absolute_error(xt_, xa))
        err_a[1].append(mean_squared_error(xt_, xa) ** 0.5)
        err_a[2].append(mean_absolute_percentage_error(xt_, xa))

df = pd.read_excel('./experiments/EnKF-DM/EnKF_DM.xlsx')
df = pd.concat(
    [df, pd.DataFrame(
        {'radius': r, 'n_members': NUMBER_OF_MEMBERS, 'p_obs': int(P * 100), 'variable': VARIABLE, 'algorithm': 'EnKF-DM',
         'b_err': [err_b],
         'a_err': [err_a],
         'time': TOTAL_TIME
         })],
    ignore_index=True)
df.to_excel('./experiments/EnKF-DM/EnKF_DM.xlsx', index=False)
