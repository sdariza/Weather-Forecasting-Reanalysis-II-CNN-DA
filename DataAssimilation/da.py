import argparse
import tensorflow as tf
import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(prefix_chars='--')
# parser.add_argument('--state', type=int, help='State to predict')
# parser.add_argument('--variable', type=str, help='Variable to predict')


# args = parser.parse_args()
# state = args.state
# variable = args.variable




# states: 0 - 6 - 12 -18

state = 6
variable = 'air'
nc_data_i = nc.Dataset(f'./test-data/0/air0.nc')
nc_data_i_1 = nc.Dataset(f'./test-data/6/air6.nc')

data_i = nc_data_i[f'{variable}'][:].data
data_i_1 = nc_data_i_1[f'{variable}'][:].data

nc_data_i.close()
nc_data_i_1.close()

xb = data_i[0].flatten()
xr = data_i_1[0].flatten()
n_members = 20

def forecast(x,t):
    x = x.reshape(73,144)
    if t == 0:
        states = [18, 0, 6, 12, 18]
    elif t == 6:
        states = [0, 6, 12, 18, 0]
    elif t == 12:
        states = [6, 12, 18, 0, 6]
    else:
        states = [12, 18, 0, 6, 12]
    for state in states:
        model = tf.keras.models.load_model(f'./cnn-models/{state}/{variable}{state}.h5', compile=False)
        x = model.predict(tf.reshape(x, [1, 73, 144, 1]))[0, ..., 0]
        del model
    return x.flatten()

def create_initial_ensemble(xb, t, N):
  Xb = []
  for _ in range(0,N):
    xbe_pert = xb+0.01*np.random.randn(73*144) #e-th pertured ensemble member (no consistent with model dynamics)
    xbe = forecast(xbe_pert, t) #e-th ensemble member (consistent with model dynamics)
    Xb.append(xbe)
  return Xb

def forecast_ensemble(Xb, t):
  for e, _ in enumerate(Xb):
    Xb[e,:] = forecast(Xb[e,:], t);
  return Xb


Xb = create_initial_ensemble(xb,6,n_members)
Xb = np.array(Xb)
xt0 = forecast(xb,6)
lons = np.linspace(0, 357.5, 144)
lats = np.linspace(90, -90, 73)


lon, lat = np.meshgrid(lons, lats)

min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()

m = Basemap(projection='cyl', resolution='c',
                llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon)

m.drawcoastlines(linewidth=0.5)
cax = m.contourf(lon, lat,Xb.mean(axis=0).reshape(73,144), levels=100)
plt.savefig('DataAssimilation/Xb.png')
plt.close()

# p = 0.8; #[0, 1] 0: no observations and 1: full observational network
# m = round(p*73*144); #number of observations m<=n
# err_obs = 0.01; #standard deviations of errors (a typical value)
# R = (err_obs**2)*np.eye(m, m); #covariance matrix of data errors
# M = 5; #number of assimilation cycles
# t = [0, 0.1]
# xt = np.array(xt0).astype('float32');
# Xa = np.array(Xb.copy()).astype('float32')
# err_a = [];
# err_b = [];

# for k in range(M):
#   #Forecast step
#   xt = forecast(xt, 0)
#   Xb = forecast_ensemble(Xa, t)
#   Pb = np.cov(Xb.T)
#   xb = np.mean(Xb, axis=0)
#   err_b.append(np.linalg.norm(xb-xt))
 
#   #Create observation - data for the assimilation process
#   H = np.random.permutation(np.arange(73*144))[:m]; #we observe a different part of the domain
#   y = np.random.multivariate_normal(xt[H], R);

#   #Assimilation step
#   Ys = np.random.multivariate_normal(y, R, N).T; #synthetic observations
#   Ds = Ys - Xb[:,H].T; #Synthetic innovations
#   Pa = R + Pb[H,:][:,H]; #Pa = R + H @ Pb @ H.T
#   Za = np.linalg.solve(Pa, Ds);
#   DX = Pb[:,H] @ Za;
#   Xa = Xb + DX.T;
#   xa = np.mean(Xa, axis=0);
#   err_a.append(np.linalg.norm(xa-xt));

# plt.plot(np.log(np.array(err_b)), '-b')
# plt.plot(np.log(np.array(err_a)), '-r')