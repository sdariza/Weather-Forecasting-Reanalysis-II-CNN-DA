import glob
import iris
import numpy as np
from pathlib import Path
import os

variables = ['air', 'uwnd', 'vwnd']
hours = [0, 6, 12, 18]
lats = np.linspace(90, -90, 73)
lons = np.linspace(0, 357.5, 144)


Path('data-training').mkdir(parents=True, exist_ok=True)
os.chdir('./data-training')


for variable in variables:
    for h in hours:
        Path(f'{h}').mkdir(parents=True, exist_ok=True)
        os.chdir(f'{h}')
        nc_paths = glob.glob(f'./../../nc-files/{variable}/*.nc')
        nc_paths.sort()
        grid_data = []
        time_data = []
        var_units = None
        time_units = None
        calendar = None
        for nc_path in nc_paths:
            cube = iris.load_cube(nc_path, constraint=iris.Constraint(
                Level=lambda cell: cell == 1000, time=lambda cell: cell.point.hour == h))
            if not time_units:
                var_units = cube.units
                time_units = str(cube.coord('time').units)
                calendar = cube.coord('time').units.calendar
            time_data.append(cube.coord('time').points)
            grid_data.append(cube.data)
            del cube
        grid_data = np.vstack(grid_data)
        time_data = np.hstack(time_data)
           
        filename = f'{variable}{h}.nc'
        cube = iris.cube.Cube(grid_data, units=var_units, var_name=variable)
        time_coord = iris.coords.DimCoord(time_data, standard_name='time', units=time_units)
        cube.add_dim_coord(time_coord, 0)
        lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
        lon_coord = iris.coords.DimCoord(lons, standard_name='longitude', units='degrees')
        cube.add_dim_coord(lat_coord, 1)
        cube.add_dim_coord(lon_coord, 2)
    
        iris.save(cube, filename)
        print(f'{filename} saved')
        os.chdir(f'./../')


print('All nc files saved, please check data-training')