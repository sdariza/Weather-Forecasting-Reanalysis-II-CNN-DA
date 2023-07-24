import iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

X = []

cube = iris.load_cube('./nc-files/uwnd/uwnd.2020.nc')

lat = cube.coord('latitude').points
lon = cube.coord('longitude').points

print(lat)
print(lon)

query = iris.Constraint(Level = lambda cell: cell == 1000, time= lambda cell: cell.point.hour == 0)



for i in range(3):
    X.append(cube.extract(query).data)

X = np.vstack(X)

lon, lat = np.meshgrid(lon, lat)

min_lat, max_lat, min_lon, max_lon = lat.min(), lat.max(), lon.min(), lon.max()


plt.figure(figsize=(10,5))
m = Basemap(projection='cyl', resolution='c', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon)
m.drawcoastlines(linewidth=.5)
cax = m.contourf(lon, lat, X[0], levels=14)
cbar = m.colorbar(cax)
m.drawparallels(np.arange(min_lat, max_lat+1, 20), labels=[1, 0, 0, 0], linewidth=0.2)
m.drawmeridians(np.arange(min_lon, max_lon+1, 30), labels=[0, 0, 0, 1], linewidth=0.2)
cbar.set_label(cube.units)
plt.show()


