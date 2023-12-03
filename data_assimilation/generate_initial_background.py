import warnings
import tensorflow as tf
import iris
import numpy as np
np.random.seed(123)
import datetime
import cftime
warnings.filterwarnings("ignore")
import pickle


# parser = argparse.ArgumentParser(prefix_chars='--')
# parser.add_argument('--start', type=str,
#                     help='Initial day to generate initial ensemble : DD-MM-YYYY')
# parser.add_argument('--variable', type=str, help='Variable to predict')


# args = parser.parse_args()
# START = args.variable.start

# VARIABLE = args.variable

VARIABLE = 'air'

xb = iris.load_cube(f"./data/test-data/0/{VARIABLE}{0}.nc")

start = iris.time.PartialDateTime(year=2023, month=1, day=1)

query = iris.Constraint(time=lambda cell: start == cell.point)

xb = xb.extract(query).data.data.flatten() - 273.15

model = [tf.keras.models.load_model(
    f'./data_driven/cnn-models/{state}/{VARIABLE}{state}.h5', compile=False) for state in [0, 6, 12, 18]]

N_LATS = 73
N_LONS = 144
NUMBER_OF_VARIABLES = N_LATS*N_LONS




def forecast(x_h, state_h: int):
    """Make predictions with CNN model

    Args:
        x_h (np.array): State to be predicted by CNN model 
        state_h (int): inital state "h" to forecast "h+1"

    Returns:
        np.array: prediction X̂_h+1
    """
    x_h = x_h.reshape(73, 144)
    x_forecast = model[state_h].predict(
        tf.reshape(x_h, [1, 73, 144, 1]))[0, ..., 0]
    return x_forecast.flatten()


def create_initial_ensemble(x_b, number_of_members):
    """Create initial background given x_b like initial value

    Args:
        x_b (np.array): _description_
        number_of_members (int): _description_
    Returns:
        np.array: initial ensemble Xb0
    """
    X_b0 = np.vstack([x_b for _ in np.arange(number_of_members)])

    for k_t in np.arange(1, 61):
        t_h = (k_t-1) % 4
        for e_member, _ in enumerate(X_b0):
            X_b0[e_member, :] = forecast(X_b0[e_member, :] + 0.05 *
                                         np.random.randn(NUMBER_OF_VARIABLES,), t_h)
    return X_b0


if __name__ == "__main__":
    Xb0 = create_initial_ensemble(xb, 50)
    np.save(f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}.npy', Xb0)