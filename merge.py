import os
import glob
import iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc


variables = ['air', 'uwnd', 'vwnd']
units = ['K', 'm s**-1', 'm s**-1']
hours = [0, 6, 12, 18]
nc_data = {}


for indx, v in enumerate(variables):
    for h in hours:
        nc_data[f'{v}{h}'] = {
            'data' : [],
            'units': units[indx],
        }
    nc_data['time'] = []




for variable in variables:
    nc_paths = glob.glob(f'./nc-files/{variable}/*.nc')
    print(nc_paths)
    for nc_path in nc_paths:
        cube = iris.load_cube(nc_path)
        print(cube)
        print(cube.coords('time')[0].points)
        for h in hours:
            query = iris.Constraint(Level = lambda cell: cell == 1000, time= lambda cell: cell.point.hour == h)
            nc_data[f'{variable}{h}']['data'].append(cube.extract(query).data)
            nc_data[f'{variable}{h}']['units'].append(cube.units)


# X = []



# cube = iris.load_cube('./nc-files/uwnd/uwnd.2020.nc')

# lat = cube.coord('latitude').points
# lon = cube.coord('longitude').points

# query = iris.Constraint(Level = lambda cell: cell == 1000, time= lambda cell: cell.point.hour == 0)



# for i in range(3):
#     X.append(cube.extract(query).data)

# X = np.vstack(X)

# lon, lat = np.meshgrid(lon, lat)

# min_lat, max_lat, min_lon, max_lon = lat.min(), lat.max(), lon.min(), lon.max()


# plt.figure(figsize=(10,5))
# m = Basemap(projection='cyl', resolution='c', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon)
# m.drawcoastlines(linewidth=.5)
# cax = m.contourf(lon, lat, X[0], levels=14)
# cbar = m.colorbar(cax)
# m.drawparallels(np.arange(min_lat, max_lat+1, 20), labels=[1, 0, 0, 0], linewidth=0.2)
# m.drawmeridians(np.arange(min_lon, max_lon+1, 30), labels=[0, 0, 0, 1], linewidth=0.2)
# cbar.set_label(cube.units)
# plt.show()


