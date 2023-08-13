"""
 This function is the work of my life :D
"""
import tensorflow as tf
import numpy as np
np.random.seed(123)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset


VARIABLE = 'air'
NUMBER_OF_VARIABLES = 73*144


nc_data = [Dataset(f'./test-data/{state}/{VARIABLE}{state}.nc')
           for state in [0, 6, 12, 18]]

data_state = [nc_data_i['air'][:].data - 273.15 for nc_data_i in nc_data]

nc_data = [nc_data_i.close() for nc_data_i in nc_data]
del nc_data

model = [tf.keras.models.load_model(
    f'./cnn-models/{state}/{VARIABLE}{state}.h5', compile=False) for state in [0, 6, 12, 18]]


def forecast(x_h, state_h: int):
    """Make predictions with CNN model

    Args:
        x_h (np.array): State to be predicted by CNN model 
        state_h (int): inital state "h" to forecast "h+1"

    Returns:
        np.array: prediction X̂_h+1
    """
    x_h = x_h.reshape(73, 144)
    x_forecast = model[state_h].predict(tf.reshape(x_h, [1, 73, 144, 1]))[0, ..., 0]
    return x_forecast.flatten()


def create_initial_ensemble(x_b, number_of_members):
    """Create initial background given x_b like initial value

    Args:
        x_b (np.array): _description_
        number_of_members (int): _description_
    Returns:
        np.array: initial ensemble Xb0
    """
    x_b0 = np.vstack([x_b for _ in np.arange(number_of_members)])

    for k_t in np.arange(1, 13):
        t_h = (k_t-1) % 4
        for e_member, _ in enumerate(x_b0):
            x_b0[e_member, :] = forecast(x_b0[e_member, :] + 0.01 *
                                  np.random.randn(NUMBER_OF_VARIABLES,), t_h)
    return x_b0


def forecast_ensemble(x_background, state_h, number_of_members):
    """Update background "Xb" state

    Args:
        x_background (np.array): _description_
        state_h (int): _description_
        number_of_members (int): _description_

    Returns:
        np.array: Xb updated h to h+1
    """
    e_member = 0
    while e_member < number_of_members:
        x_background[e_member, :] = forecast(
            x_background[e_member, :], state_h)
        e_member +=1
    return x_background

NUMBER_OF_MEMBERS = 40
Xb0 = create_initial_ensemble(
    data_state[0][0].flatten(), NUMBER_OF_MEMBERS)  # Xb0 in state 0

P = 0.2
M = round(P*NUMBER_OF_VARIABLES)
OBSERVATION_ERROR = 0.01
R = (OBSERVATION_ERROR**2)*np.eye(M, M)
# NUMBER_OF_ASSIMILATION_CYCLES = 20
Xa = np.array(Xb0.copy()).astype('float32')

err_a = []
err_b = []

for day in range(5):
    for state in [0,1,2,3]:
        NEXT_STATE = (state+1)%4
        if NEXT_STATE == 0:
            xt = data_state[NEXT_STATE][day+1].flatten()
        else:
            xt = data_state[NEXT_STATE][day].flatten()

        Xb = forecast_ensemble(x_background=Xa, state_h=state, number_of_members=NUMBER_OF_MEMBERS)
        xb = np.mean(Xb, axis=0)
        err_b.append(np.mean(np.abs(xb-xt)))
        #Create observation - data for the assimilation process
        H = np.random.permutation(np.arange(NUMBER_OF_VARIABLES))[:M] #we observe a different part of the domain
        y = np.random.multivariate_normal(xt[H], R)

        #Assimilation step
        Pb = np.cov(Xb.T)
        Ys = np.random.multivariate_normal(y, R, NUMBER_OF_MEMBERS).T #synthetic observations
        Ds = Ys - Xb[:,H].T #Synthetic innovations
        Pa = R + Pb[H,:][:,H] #Pa = R + H @ Pb @ H.T
        Za = np.linalg.solve(Pa, Ds)
        DX = Pb[:,H] @ Za
        Xa = Xb + DX.T
        xa = np.mean(Xa, axis=0)
        err_a.append(np.mean(np.abs(xa-xt)))

fig = plt.figure()
plt.plot(np.log(np.array(err_b)), '-b', label='Background error')
plt.plot(np.log(np.array(err_a)), '-r', label='Analysis error')
plt.legend()
# plt.savefig('DataAssimilation/backgroundAnalysisError.png')
plt.show()
plt.close(fig)

lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)
lon, lat = np.meshgrid(lons, lats)
min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

fig.set_size_inches(18, 8)
m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=min_lat, urcrnrlat=max_lat,
            llcrnrlon=min_lon, urcrnrlon=max_lon)
m.ax = ax[0]
m.drawcoastlines(linewidth=0.5)

cax = m.contourf(lon, lat, xt.reshape(73,144))

cbar = m.colorbar(cax)
cbar.set_label('°C')

m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                labels=[0, 0, 0, 1], linewidth=0.2)
ax[0].set_title('Real state')

m.ax = ax[1]
m.drawcoastlines(linewidth=0.5)

cax = m.contourf(lon, lat, xb.reshape(73,144))

cbar = m.colorbar(cax)
cbar.set_label('°C')

m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                labels=[0, 0, 0, 1], linewidth=0.2)
ax[1].set_title('Estimated state')
plt.tight_layout()
fig.suptitle('CNN - Forecasting Global Weather', y=0.79)
plt.savefig('DataAssimilation/results.png')
plt.close(fig)
