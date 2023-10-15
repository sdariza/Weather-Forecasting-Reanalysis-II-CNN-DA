"""
This code generate simulations steps for Data Assimilation 
process based on : Modified Cholesky Decomposition
"""
from generate_initial_background import NUMBER_OF_VARIABLES, VARIABLE, forecast

import warnings
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
import iris
import numpy as np
np.random.seed(123)
import datetime
import cftime
from sklearn.linear_model import Ridge
import scipy.sparse as spa
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
import pickle



print(iris.load_cube('./data/test-data/6/air6.nc'))
nc_data = []
for state in [0,6,12,18]:
    nc_data.append(iris.load_cube(f'./data/test-data/{state}/{VARIABLE}{state}.nc'))
nc_data = [iris.load_cube(f'./data/test-data/{state}/{VARIABLE}{state}.nc')
           for state in [0, 6, 12, 18]]
start = iris.time.PartialDateTime(year=2023, month=1, day=1)
end = iris.time.PartialDateTime(year=2023, month=1, day=10)
query = iris.Constraint(time=lambda cell: start <= cell.point <= end)
nc_data = [nc_data_i.extract(query) for nc_data_i in nc_data]
data_state = [nc_data_i.data.data for nc_data_i in nc_data]
date_time = [nc_data_i.coord('time') for nc_data_i in nc_data]
time_units = date_time[0][0].units
del nc_data
from mpl_toolkits.basemap import Basemap
if VARIABLE == 'air':
    data_state = [nc_data_i - 273.15 for nc_data_i in data_state]


model = [tf.keras.models.load_model(
    f'./data_driven/cnn-models/{state}/{VARIABLE}{state}.h5', compile=False) for state in [0, 6, 12, 18]]


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
        e_member += 1
    return x_background

lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)


lon, lat = np.meshgrid(lons, lats)

min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()
images = []
def plot(xb, xt, xa, day, state):

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
    fig.set_size_inches(18, 8)
    
    for i in range(3):  # Itera para configurar cada mapa
        m = Basemap(projection='robin', resolution='c', lat_0=lat[0][0], lon_0=lon[0][0], ax=ax[i])
        m.drawcoastlines(linewidth=0.5)
        cax = m.contourf(lon, lat, (xt if i == 0 else xb if i == 1 else xa).reshape(73, 144), latlon=True)

        # Agrega una barra de colores en la parte inferior de cada mapa
        cbar = plt.colorbar(cax, ax=ax[i], location='bottom')
        cbar.set_label(f'°C')

        m.drawparallels(np.arange(min_lat, max_lat+1, 20), labels=[1, 0, 0, 0], linewidth=0.2)
        m.drawmeridians(np.arange(min_lon, max_lon+1, 55), labels=[0, 0, 0, 1], linewidth=0.2)
        ax[i].set_title('Real state' if i == 0 else 'Background state' if i == 1 else 'Estimated state')

    plt.tight_layout()
    fig.suptitle('Data Assimilation - Forecasting Global Weather', y=0.73)
    plt.savefig(f'./data_assimilation/plots/{VARIABLE}/subplot_{day}{state}.png')
    plt.close(fig)
    images.append(imageio.imread(f'./data_assimilation/plots/{VARIABLE}/subplot_{day}{state}.png'))

def get_decorrelation_matrix(r):
    L = np.zeros((73*144, 73*144))
    for i in range(73): # lats
        for j in range(144): # lons
            for ii in range(73):
                for jj in range(144):
                    dij = np.sqrt(np.power(i-ii,2)+np.power(j-jj,2))
                    L[i*144+j, ii*144+jj] = np.exp(-0.5*(dij)**2/r**2)
    return L

NUMBER_OF_MEMBERS = 50
x0 = data_state[0][0].flatten()
Xb0 = np.load(f'./data_assimilation/InitialBackground/initialBackground_{VARIABLE}.npy')


P = 0.05
M = round(P*NUMBER_OF_VARIABLES)
OBSERVATION_ERROR = 0.01
R = (OBSERVATION_ERROR**2)*np.eye(M, M)
Rinv = 1/(OBSERVATION_ERROR**2)*np.ones((M,)) #covariance matrix of data errors
NUMBER_OF_ASSIMILATION_CYCLES = len(data_state[0])
print(NUMBER_OF_ASSIMILATION_CYCLES)
Xa = np.array(Xb0.copy()).astype('float32')

err_a = []
err_b = []
print("Getting decorrelation matrix...")
L = get_decorrelation_matrix(5)
print("Data assimilation steps...")
for day in range(NUMBER_OF_ASSIMILATION_CYCLES-1):
    print(f'Day:{day}')
    for state in [0, 1, 2, 3]:
        NEXT_STATE = (state+1) % 4
        if NEXT_STATE == 0:
            xt_ = data_state[NEXT_STATE][day+1].flatten()
        else:
            xt_ = data_state[NEXT_STATE][day].flatten()
        Xb = forecast_ensemble(
            x_background=Xa, state_h=state, number_of_members=NUMBER_OF_MEMBERS)
        xb = np.mean(Xb, axis=0)
        Pb = L * np.cov(Xb.T)


        # Create observation - data for the assimilation process
        H = np.random.permutation(np.arange(NUMBER_OF_VARIABLES))[
            :M]  # we observe a different part of the domain
        err_b.append(mean_absolute_error(xt_,xb))
        y = np.random.multivariate_normal(xt_[H], R)

        # Assimilation step
        Ys = np.random.multivariate_normal(
            y, R, NUMBER_OF_MEMBERS).T # synthetic observations
        Ds = Ys - Xb[:, H].T  # Synthetic innovations
        Pa = R + Pb[H,:][:,H]
        Za = np.linalg.solve(Pa,Ds)
        Dx = Pb[:,H] @ Za
        Xa = Xb + Dx.T
        xa = np.mean(Xa, axis=0)
        err_a.append(mean_absolute_error(xt_,xa))
        # print(xb.shape, xt_.shape, xa.shape)
        plot(xb, xt_, xa, day,state)
imageio.mimsave(f'./data_assimilation/plots/{VARIABLE}/predictions.gif', images, duration=1000, loop=0)

fig = plt.figure()
plt.plot(np.log(err_b), '-ob', label='Background error')
plt.plot(np.log(err_a), '-or', label='Analysis error')
# plt.plot(np.log(np.array(err_cnn)), '-g', label='CNN error')
plt.legend()
plt.xlabel('Assimilation step')
plt.ylabel('$ln(MAE)$')
plt.savefig(f'./data_assimilation/plots/{VARIABLE}/backgroundAnalysisError_id{1}.png')
plt.show()
plt.close(fig)