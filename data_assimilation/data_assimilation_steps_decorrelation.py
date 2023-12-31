"""
This code generate simulations steps for (LEnKF) Data Assimilation
process based on : Decorrelation matrix
"""
import warnings
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from commons import load_data, forecast_ensemble, load_model, observation_cov_matrix, plot, NUMBER_OF_VARIABLES, \
    plot_background_analysis_error
import numpy as np
import imageio

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

images = []

Xb0 = np.load(
    f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}.npy')

M, R, _ = observation_cov_matrix(P)

NUMBER_OF_ASSIMILATION_CYCLES = len(data_state[0])

Xa = np.array(Xb0.copy()).astype('float32')

err_a = [[], [], []]
err_b = [[], [], []]
print("Getting decorrelation matrix...")
L = np.load(f'./data_assimilation/decorrelation_matrices/decorrelation_r{r}.npy')
print("Data assimilation steps...")

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
        Pb = L * np.cov(Xb.T)  # Localization method

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
        Pa = R + Pb[H, :][:, H]
        Za = np.linalg.solve(Pa, Ds)
        Dx = Pb[:, H] @ Za
        Xa = Xb + Dx.T
        xa = np.mean(Xa, axis=0)
        err_a[0].append(mean_absolute_error(xt_, xa))
        err_a[1].append(mean_squared_error(xt_, xa) ** 0.5)
        err_a[2].append(mean_absolute_percentage_error(xt_, xa))
        plot(Basemap, xb, xt_, xa, day, state, VARIABLE, images, unit_data)
imageio.mimsave(
    f'./data_assimilation/plots/{VARIABLE}/predictions.gif', images, duration=1000, loop=0)

plot_background_analysis_error(err_b, err_a, VARIABLE)
