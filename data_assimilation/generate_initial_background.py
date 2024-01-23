from commons import load_model
import argparse
import warnings
import tensorflow as tf
import iris
import numpy as np

np.random.seed(123)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--variable', type=str, help='Variable to predict')
parser.add_argument('--n_members', type=int, help='Number of ensemble members')

args = parser.parse_args()
N_MEMBERS = args.n_members

VARIABLE = args.variable

xb = iris.load_cube(f"./data/test-data/0/{VARIABLE}{0}.nc")

start = iris.time.PartialDateTime(year=2023, month=11, day=1)

query = iris.Constraint(time=lambda cell: start == cell.point)
xb = xb.extract(query).data.data

if VARIABLE == 'air':
    xb = xb - 273.15

xb = xb[..., np.newaxis]

model = load_model(variable=VARIABLE)

N_LATS = 73
N_LONS = 144
NUMBER_OF_VARIABLES = N_LATS * N_LONS


def forecast(X_h, state_h: int):
    """Make predictions with CNN model

    Args:
        X_h (np.array): State to be predicted by CNN model
        state_h (int): initial state "h" to forecast "h+1"

    Returns:
        np.array: prediction X̂_h+1
    """
    return model[state_h].predict(X_h, verbose=2)


def create_initial_ensemble(x_b, number_of_members):
    """Create initial background given x_b like initial value

    Args:
        x_b (np.array): _description_
        number_of_members (int): _description_
    Returns:
        np.array: initial ensemble Xb0
    """
    X_b0 = tf.convert_to_tensor([x_b for _ in np.arange(number_of_members)])
    for k_t in np.arange(1, 181):
        t_h = (k_t - 1) % 4
        error = np.random.randn(number_of_members, N_LATS, N_LONS, 1)
        X_b0 = tf.convert_to_tensor(forecast(X_b0 + 0.01 * error, t_h))
    return X_b0.numpy().reshape(number_of_members, N_LATS * N_LONS)


if __name__ == "__main__":
    print(f'Generating initial ensemble with of {VARIABLE} with {N_MEMBERS} members')
    Xb0 = create_initial_ensemble(xb, N_MEMBERS)
    np.save(
        f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}_{N_MEMBERS}.npy', Xb0)
