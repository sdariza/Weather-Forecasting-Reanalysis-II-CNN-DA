"""
This code generate simulations steps for Data Assimilation 
process based on : Modified Cholesky Decomposition
"""
from commons import get_inv, plot, load_data, load_model, observation_cov_matrix, forecast_ensemble, \
    NUMBER_OF_VARIABLES, plot_background_analysis_error
import warnings
import argparse
import imageio
import numpy as np
import pickle

np.random.seed(123)

from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--variable', type=str, help='Variable to predict')
parser.add_argument('--n_members', type=int, help='Number of ensemble members')
parser.add_argument('--p_obs', type=int, help='% of observation')
parser.add_argument('--r', type=int, help='Influence radio')
parser.add_argument('--alpha', type=int, help='Alfa Ridge')

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

x0 = data_state[0][0].flatten()
Xb0 = np.load(f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}.npy')

M, R, Rinv = observation_cov_matrix(P)
NUMBER_OF_ASSIMILATION_CYCLES = len(data_state[0])
Xa = np.array(Xb0.copy()).astype('float32')

err_a = []
err_b = []
images = []
from mpl_toolkits.basemap import Basemap

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
        err_b.append(mean_absolute_error(xt_, xb))
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
        plot(Basemap, xb, xt_, xa, day, state, VARIABLE, images, unit_data)
imageio.mimsave(f'./data_assimilation/plots/{VARIABLE}/predictions.gif', images, duration=1000, loop=0)

plot_background_analysis_error(err_b, err_a, VARIABLE)
