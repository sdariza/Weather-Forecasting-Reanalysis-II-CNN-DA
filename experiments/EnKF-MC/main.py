"""
This code generate simulations steps for Data Assimilation
process based on : Modified Cholesky Decomposition
"""
import sys

sys.path.append(r'C:\Users\Lab6k\Desktop\Weather-Forecasting-Reanalysis-II-CNN-DA')

from data_assimilation.commons import load_data, forecast_ensemble, load_model, observation_cov_matrix, \
    NUMBER_OF_VARIABLES, get_inv
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import argparse
import numpy as np
import pickle
import pandas as pd

np.random.seed(123)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--variable', type=str, help='Variable to predict')
parser.add_argument('--n_members', type=int, help='Number of ensemble members')
parser.add_argument('--p_obs', type=int, help='% of observation')
parser.add_argument('--r', type=int, help='Influence radio')
parser.add_argument('--alpha', type=float, help='Alfa Ridge')

args = parser.parse_args()

VARIABLE = args.variable
NUMBER_OF_MEMBERS = args.n_members
P = args.p_obs / 100
r = args.r
alpha = args.alpha

data_state, time_units, unit_data = load_data(variable=VARIABLE)
model = load_model(variable=VARIABLE)

with open(f"./data_assimilation/predecessors/predecessor_r{r}", "rb") as fp:
    predecessors = pickle.load(fp)

Xb0 = np.load(f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}_{NUMBER_OF_MEMBERS}.npy')

M, R, Rinv = observation_cov_matrix(P)
NUMBER_OF_ASSIMILATION_CYCLES = len(data_state[0])
Xa = np.array(Xb0.copy()).astype('float32')

err_a = [[], [], []]
err_b = [[], [], []]

for day in range(NUMBER_OF_ASSIMILATION_CYCLES - 1):
    print(f'Day:{day}')
    for state in [0, 1, 2, 3]:
        NEXT_STATE = (state + 1) % 4
        if NEXT_STATE == 0:
            xt_ = data_state[NEXT_STATE][day + 1].flatten()
        else:
            xt_ = data_state[NEXT_STATE][day].flatten()
        Xb = forecast_ensemble(model, x_background=Xa, state_h=state, number_of_members=NUMBER_OF_MEMBERS)
        xb = np.mean(Xb, axis=0)
        Db = Xb - xb
        Binv = get_inv(Db, predecessors, alpha)
        # Create observation - data for the assimilation process
        H = np.random.permutation(np.arange(NUMBER_OF_VARIABLES))[
            :M]  # we observe a different part of the domain
        err_b[0].append(mean_absolute_error(xt_, xb))
        err_b[1].append(mean_squared_error(xt_, xb) ** 0.5)
        err_b[2].append(mean_absolute_percentage_error(xt_, xb))
        y = np.random.multivariate_normal(xt_[H], R)

        # Assimilation step
        Ys = np.random.multivariate_normal(
            y, R, NUMBER_OF_MEMBERS).T  # synthetic observations
        Ds = Ys - Xb[:, H].T  # Synthetic innovations
        HRinv = np.zeros((NUMBER_OF_VARIABLES,))
        HRinv[H] = Rinv
        HTRinvH = np.diag(HRinv)
        A = Binv + HTRinvH
        W = np.zeros((NUMBER_OF_VARIABLES, NUMBER_OF_MEMBERS))
        W[H, :] = np.linalg.solve(R, Ds)
        Za = np.linalg.solve(A, W)
        Xa = Xb + Za.T
        xa = np.mean(Xa, axis=0)
        err_a.append(mean_absolute_error(xt_, xa))
        err_a[0].append(mean_absolute_error(xt_, xa))
        err_a[1].append(mean_squared_error(xt_, xa) ** 0.5)
        err_a[2].append(mean_absolute_percentage_error(xt_, xa))

df = pd.read_excel('./experiments/EnKF-MC/EnKF_MC.xlsx')
df = pd.concat(
    [df, pd.DataFrame(
        {'alpha': alpha, 'radius': r, 'n_members': NUMBER_OF_MEMBERS, 'p_obs': P * 100, 'variable': VARIABLE,
         'algorithm': 'EnKF-MC',
         'b_err': [err_b],
         'a_err': [err_a],
         'mean_b_err': [[np.array(err_b[0]).mean(), np.array(err_b[1]).mean(), np.array(err_b[2]).mean()]],
         'mean_a_err': [[np.array(err_a[0]).mean(), np.array(err_a[1]).mean(), np.array(err_a[2]).mean()]],
         'variance_b_err': [[np.array(err_b[0]).var(), np.array(err_b[1]).var(), np.array(err_b[2]).var()]],
         'variance_a_err': [[np.array(err_a[0]).var(), np.array(err_a[1]).var(), np.array(err_a[2]).var()]],
         })],
    ignore_index=True)
df.to_excel('./experiments/EnKF-MC/EnKF_MC.xlsx', index=False)
