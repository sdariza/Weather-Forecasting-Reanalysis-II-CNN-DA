"""
 This function is the work of my life :D
"""
import tensorflow as tf
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


VARIABLE = 'air'

model = [tf.keras.models.load_model(
    f'./cnn-models/{state}/{VARIABLE}{state}.h5', compile=False) for state in [0, 6, 12, 18]]


def forecast(x_forecast, t_state: int):
    """_summary_

    Args:
        x_forecast (_type_): _description_
        t_state (int): _description_

    Returns:
        _type_: _description_
    """
    x_forecast = x_forecast.reshape(73, 144)
    if t_state == 0:
        states = [3, 0, 1, 2, 3]
    elif t_state == 6:
        states = [0, 1, 2, 3, 0]
    elif t_state == 12:
        states = [1, 2, 3, 0, 1]
    else:
        states = [2, 3, 0, 1, 2]
    for _ in range(1):
        for state in states:
            x_forecast = model[state].predict(tf.reshape(x_forecast, [1, 73, 144, 1]))[0, ..., 0]
    return x_forecast.flatten()


def create_initial_ensemble(x_b, t_state, n_members):
    """_summary_

    Args:
        x_b (_type_): _description_
        t_state (_type_): _description_
        n_members (_type_): _description_

    Returns:
        _type_: _description_
    """
    ensemble = []
    for _ in range(0, n_members):
        # e-th pertured ensemble member (no consistent with model dynamics)
        xbe_pert = x_b+0.01*np.random.randn(73*144)
        # e-th ensemble member (consistent with model dynamics)
        xbe = forecast(xbe_pert, t_state)
        ensemble.append(xbe)
    return np.array(ensemble)


def forecast_ensemble(ensemble, t_state):
    """_summary_

    Args:
        ensemble (_type_): _description_
        t_state (_type_): _description_

    Returns:
        _type_: _description_
    """
    for element, _ in enumerate(ensemble):
        ensemble[element, :] = forecast(ensemble[element, :], t_state)
    return Xb


NMEMBERS = 20
x_random = np.random.randn(73*144,)
x_0 = forecast(x_random, 6)
x_b0 = forecast(x_0, 6)
x_t = forecast(x_0, 6) # reference
Xb = create_initial_ensemble(x_b0, 6, NMEMBERS)
lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)


lon, lat = np.meshgrid(lons, lats)

min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()

m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=min_lat, urcrnrlat=max_lat,
            llcrnrlon=min_lon, urcrnrlon=max_lon)

m.drawcoastlines(linewidth=0.5)
cax = m.contourf(lon, lat, Xb.mean(axis=0).reshape(73, 144), levels=100)
plt.savefig('DataAssimilation/Xb.png')
plt.show()
plt.close()
