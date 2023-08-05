import argparse

from mpl_toolkits.basemap import Basemap
import tensorflow as tf
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import imageio


parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--state', type=int, help='State to predict')
parser.add_argument('--variable', type=str, help='Variable to predict')


args = parser.parse_args()
state = args.state
variable = args.variable

best_model = tf.keras.models.load_model(f'./cnn-models/{state}/{variable}{state}.h5', compile=False)


nc_t_i = nc.Dataset(f'./test-data/{state}/{variable}{state}.nc')
nc_t_j = nc.Dataset(f'./test-data/{state}/{variable}{state}.nc')

X = nc_t_i.variables[f'{variable}'][:].data
Y = nc_t_j.variables[f'{variable}'][:].data

nc_t_i.close()
nc_t_j.close()

if variable == 'air':
    X = X.reshape(-1, 73, 144, 1) - 273.15
    Y = Y.reshape(-1, 73, 144, 1) - 273.15


lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)


lon, lat = np.meshgrid(lons, lats)

min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()


images = []

for i in np.random.randint(0, len(X), 10):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.set_size_inches(18, 8)
    m = Basemap(projection='cyl', resolution='c',
                llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon)
    m.ax = ax[0]
    m.drawcoastlines(linewidth=0.5)

    cax = m.contourf(lon, lat, Y[i, ..., 0], levels=100)

    cbar = m.colorbar(cax)
    cbar.set_label('C')

    m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                    labels=[1, 0, 0, 0], linewidth=0.2)
    m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                    labels=[0, 0, 0, 1], linewidth=0.2)
    ax[0].set_title('Real state')

    m.ax = ax[1]
    m.drawcoastlines(linewidth=0.5)

    cax = m.contourf(lon, lat, best_model.predict(tf.reshape(
        X[i], [1, 73, 144, 1]))[0, ..., 0], levels=100)

    cbar = m.colorbar(cax)
    cbar.set_label('C')

    m.drawparallels(np.arange(min_lat, max_lat+1, 20),
                    labels=[1, 0, 0, 0], linewidth=0.2)
    m.drawmeridians(np.arange(min_lon, max_lon+1, 30),
                    labels=[0, 0, 0, 1], linewidth=0.2)
    ax[1].set_title('Estimated state')
    plt.tight_layout()
    fig.suptitle('CNN - Forecasting Global Weather', y=0.79)
    plt.savefig(f"plots/subplot_{i}.png")
    plt.close(fig)
    images.append(imageio.imread(f"plots/subplot_{i}.png"))
imageio.mimsave("plots/subplots.gif", images, duration=1000, loop=0)

del X, Y