import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import numpy as np

plt.figure(figsize=(15, 7))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.7)
ax.coastlines(resolution='110m')
gl = ax.gridlines(color='black',draw_labels=False, linewidth=1)
gl.xlocator = mticker.FixedLocator([-90, -135, -180,-45,0, 45, 90, 135, 180])
gl.ylocator = mticker.FixedLocator([-60,-40,-20, 0, 20,40,60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# Coordenadas de los puntos que deseas colocar en el mapa

lons = np.array([-90, -135, -180,-45,0, 45, 90, 135, 180])
lats = np.array([-60,-40,-20, 0, 20,40,60])

# for i in lons:
#     for j in lats:
#         # Dibuja los puntos sobre el mapa
#         ax.scatter( i,j, color='green', marker='o', transform=ccrs.PlateCarree(), s=100)
# plt.show()

# Crear el meshgrid
X, Y = np.meshgrid(lons, lats)



# Convertir los meshgrid a arreglos 1D
X_flat = X.flatten()
Y_flat = Y.flatten()

# Número de pares de coordenadas a seleccionar
num_pairs = 20

# Seleccionar pares de coordenadas de forma aleatoria
random_indices = np.random.choice(len(X_flat), size=num_pairs, replace=False)
random_pairs = list(zip(X_flat[random_indices], Y_flat[random_indices]))

print("Pares de coordenadas seleccionados aleatoriamente:", random_pairs)

for i,j in random_pairs:
        # Dibuja los puntos sobre el mapa
    ax.scatter( i,j, color='blue', marker='X', transform=ccrs.PlateCarree(), s=100)
plt.axis('off')
plt.show()