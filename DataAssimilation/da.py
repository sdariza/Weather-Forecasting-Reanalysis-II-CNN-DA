"""
 This function is the work of my life :D
"""
import tensorflow as tf
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset

nc_data_x0 = Dataset('./test-data/0/air0.nc')
nc_data_x6 = Dataset('./test-data/6/air6.nc')

X0 = nc_data_x0['air'][:].data - 273.15
X6 = nc_data_x6['air'][:].data - 273.15

nc_data_x0.close()
nc_data_x6.close()

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
    x_forecast = model[(t_state-1)%4].predict(tf.reshape(x_forecast, [1, 73, 144, 1]))[0, ..., 0]
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
        xbe_pert = x_b+0.01*np.random.randn(73,144)
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


NMEMBERS = 40

Xb = create_initial_ensemble(X0[0],1, NMEMBERS)

p = 0.2; #[0, 1] 0: no observations and 1: full observational network
m = round(p*73*144); #number of observations m<=n
err_obs = 0.01; #standard deviations of errors (a typical value)
R = (err_obs**2)*np.eye(m, m); #covariance matrix of data errors
M = 20

xt = X6[0].flatten()


Pb = np.cov(Xb.T)
#Create observation - data for the assimilation process
H = np.random.permutation(np.arange(73*144))[:m] #we observe a different part of the domain
y = np.random.multivariate_normal(xt[H], R)
#Assimilation step
Ys = np.random.multivariate_normal(y, R, NMEMBERS).T #synthetic observations
Ds = Ys - Xb[:,H].T #Synthetic innovations
Pa = R + Pb[H,:][:,H] #Pa = R + H @ Pb @ H.T
Za = np.linalg.solve(Pa, Ds)
DX = Pb[:,H] @ Za
Xa = Xb + DX.T
xa = np.mean(Xa, axis=0)


print(np.mean(np.abs(xa-xt)))
print(100*np.mean(np.abs((xa-xt)/xt)))

fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True)
fig.set_size_inches(25, 20)
lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)


lon, lat = np.meshgrid(lons, lats)

min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()

m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=min_lat, urcrnrlat=max_lat,
            llcrnrlon=min_lon, urcrnrlon=max_lon)
m.ax = ax[0,0]
m.drawcoastlines(linewidth=0.5)

cax = m.contourf(lon, lat, xt.reshape(73,144))

cbar = m.colorbar(cax)
cbar.set_label('C')

m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                labels=[0, 0, 0, 1], linewidth=0.2)
ax[0,0].set_title('Real state')


m.ax = ax[0,1]
m.drawcoastlines(linewidth=0.5)

cax = m.contourf(lon, lat, model[0].predict(tf.reshape(
    X6[0], [1, 73, 144, 1]))[0, ..., 0])

cbar = m.colorbar(cax)
cbar.set_label('C')

m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                labels=[0, 0, 0, 1], linewidth=0.2)
ax[0,1].set_title('Estimated state - CNN')

m.ax = ax[1,0]
m.drawcoastlines(linewidth=0.5)

cax = m.contourf(lon, lat, Xb.mean(axis=0).reshape(73, 144))

cbar = m.colorbar(cax)
cbar.set_label('C')

m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                labels=[0, 0, 0, 1], linewidth=0.2)
ax[1,0].set_title('Background')

m.ax = ax[1,1]
m.drawcoastlines(linewidth=0.5)

cax = m.contourf(lon, lat, xa.reshape(73,144))

cbar = m.colorbar(cax)
cbar.set_label('C')

m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                labels=[0, 0, 0, 1], linewidth=0.2)
ax[1,1].set_title('Data Assimilation')


# Ajustar el tamaño de la letra en títulos y etiquetas
for row in ax:
    for ax_single in row:
        ax_single.title.set_fontsize(20)  # Tamaño de letra para los títulos
        ax_single.xaxis.label.set_fontsize(25)  # Tamaño de letra para la etiqueta del eje X
        ax_single.yaxis.label.set_fontsize(25)  # Tamaño de letra para la etiqueta del eje Y
        ax_single.tick_params(axis='both', which='major', labelsize=25)  # Tamaño de letra para los números de los ejes


plt.tight_layout()
fig.suptitle('CNN - Forecasting Global Weather', fontsize=26)
plt.savefig(f'DataAssimilation/results.png')
plt.close(fig)