import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
import iris
import numpy as np
from sklearn.linear_model import Ridge
import scipy.sparse as spa

np.random.seed(123)

lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)

lon, lat = np.meshgrid(lons, lats)
min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()
NUMBER_OF_VARIABLES = 73 * 144


def load_data(variable):
    nc_data = []
    for state in [0, 6, 12, 18]:
        nc_data.append(iris.load_cube(
            f'./data/test-data/{state}/{variable}{state}.nc'))
    nc_data = [iris.load_cube(f'./data/test-data/{state}/{variable}{state}.nc')
               for state in [0, 6, 12, 18]]

    start = iris.time.PartialDateTime(year=2023, month=10, day=1)
    end = iris.time.PartialDateTime(year=2023, month=10, day=16)

    query = iris.Constraint(time=lambda cell: start <= cell.point <= end)
    nc_data = [nc_data_i.extract(query) for nc_data_i in nc_data]
    data_state = [nc_data_i.data.data for nc_data_i in nc_data]
    date_time = [nc_data_i.coord('time') for nc_data_i in nc_data]
    time_units = date_time[0][0].units
    unit_data = nc_data[0].units
    del nc_data
    if variable == 'air':
        data_state = [nc_data_i - 273.15 for nc_data_i in data_state]
    return data_state, time_units, unit_data


def load_model(variable):
    """
    Load CNN model - Markov Chain
    :param variable:
    :return: list with model
    """
    model = []
    for state in [0, 6, 12, 18]:
        model.append(tf.keras.models.load_model(
            f'./data_driven/cnn-models/{state}/{variable}{state}.h5', compile=False))
    return model


def forecast_ensemble(model, x_background, state_h, number_of_members):
    """
    Update ensemble Xb_t to Xb_{t+1}
    :param model:
    :param x_background:
    :param state_h:
    :param number_of_members:
    :return:
    """
    return model[state_h].predict(tf.convert_to_tensor(x_background.reshape(number_of_members, 73, 144, 1)), verbose=0).reshape(
        number_of_members, 73 * 144)


def observation_cov_matrix(p, observation_error=0.01):
    M = round(p * NUMBER_OF_VARIABLES)
    Rinv = 1 / (observation_error ** 2) * np.ones((M,))  # covariance matrix of data errors
    return M, (observation_error ** 2) * np.eye(M, M), Rinv


def plot(Basemap, xb, xt, xa, day, state, variable, images, unit_data):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
    fig.set_size_inches(22, 8)

    for i in range(3):  # Itera para configurar cada mapa
        m = Basemap(projection='robin', resolution='c',
                    lat_0=lat[0][0], lon_0=lon[0][0], ax=ax[i])
        m.drawcoastlines(linewidth=0.5)
        x = (xt if i == 0 else xb if i == 1 else xa).reshape(
            73, 144)
        cax = m.contourf(lon, lat, x if variable != 'air' else x + 273.15, latlon=True, cmap='Blues')

        # Agrega una barra de colores en la parte inferior de cada mapa
        cbar = plt.colorbar(cax, ax=ax[i], location='bottom', pad=0.1)
        cbar.set_label(f'{unit_data}')

        m.drawparallels(np.arange(min_lat, max_lat + 1, 20),
                        labels=[1, 0, 0, 0], linewidth=0.2)
        m.drawmeridians(np.arange(min_lon, max_lon + 1, 80),
                        labels=[0, 0, 0, 1], linewidth=0.2, labelstyle='N/S')
        ax[i].set_title(
            'Real state' if i == 0 else 'Background state' if i == 1 else 'Estimated state')

    plt.savefig(
        f'./data_assimilation/plots/{variable}/subplot_{day}{state}.png', bbox_inches='tight')
    plt.savefig(
        f'./data_assimilation/plots/{variable}/subplot_{day}{state}.pdf', bbox_inches='tight')
    plt.close(fig)
    images.append(imageio.imread(
        f'./data_assimilation/plots/{variable}/subplot_{day}{state}.png'))


def plot_background_analysis_error(err_b, err_a, variable):
    fig = plt.figure()
    plt.plot(np.log(err_b), '-ob', label='Background error')
    plt.plot(np.log(err_a), '-or', label='Analysis error')
    # plt.plot(np.log(np.array(err_cnn)), '-g', label='CNN error')
    plt.legend()
    plt.xlabel('Assimilation step')
    plt.ylabel('$ln(MAE)$')
    plt.savefig(
        f'./data_assimilation/plots/{variable}/backgroundAnalysisError_id{1}.png')
    plt.show()
    plt.close(fig)


# Cholesky


def get_inv(Db, predecessors, alpha=1):
    total_pred = int(predecessors['total'])
    I = np.zeros(total_pred + NUMBER_OF_VARIABLES)
    J = np.zeros(total_pred + NUMBER_OF_VARIABLES)
    V = np.zeros(total_pred + NUMBER_OF_VARIABLES)

    Q = np.zeros(NUMBER_OF_VARIABLES)

    indx = 0
    for i in np.arange(0, NUMBER_OF_VARIABLES):
        pi = predecessors[f'{i}']

        Zi = Db[:, pi]
        y = Db[:, i]

        if len(pi) > 0:
            lr = Ridge(alpha=alpha, fit_intercept=False)
            beta_i = lr.fit(Zi, y).coef_

            I[indx:indx + len(pi)] = i
            J[indx:indx + len(pi)] = pi
            V[indx:indx + len(pi)] = - beta_i
            Q[i] = 1 / np.var(y - Zi @ beta_i)
            indx = indx + len(pi)
        else:
            Q[i] = 1 / np.var(y)

    In = np.arange(0, NUMBER_OF_VARIABLES)
    I[indx:indx + NUMBER_OF_VARIABLES] = In
    J[indx:indx + NUMBER_OF_VARIABLES] = In
    V[indx:indx + NUMBER_OF_VARIABLES] = np.ones(NUMBER_OF_VARIABLES)

    T = spa.coo_matrix((V, (I, J)), shape=(NUMBER_OF_VARIABLES, NUMBER_OF_VARIABLES))
    Q = spa.coo_matrix((Q, (In, In)), shape=(NUMBER_OF_VARIABLES, NUMBER_OF_VARIABLES))
    return T.T @ Q @ T
