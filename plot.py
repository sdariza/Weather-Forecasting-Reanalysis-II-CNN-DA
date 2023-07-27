import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


nc_data = nc.Dataset('./data-training/0/air0.nc')

lats = nc_data.variables['latitude'][:].data
lons = nc_data.variables['longitude'][:].data
lon, lat = np.meshgrid(lons, lats)

min_lat, max_lat, min_lon, max_lon = lats.min(), lats.max(), lons.min(), lons.max()

plt.figure(figsize=(10,5))
m = Basemap(projection='cyl', resolution='c', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon)
m.drawcoastlines(linewidth=.5)
cax = m.contourf(lon, lat, nc_data['air'][-1].data, levels=19, cmap='jet')
cbar = m.colorbar(cax)
m.drawparallels(np.arange(min_lat, max_lat+1, 20), labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30), labels=[0, 0, 0, 1], linewidth=0.2)
cbar.set_label(nc_data.variables['air'].units)
plt.title(nc.num2date(nc_data['time'][-1], units=nc_data['time'].units))
plt.show()

print(nc_data['air'][:].data.shape)
