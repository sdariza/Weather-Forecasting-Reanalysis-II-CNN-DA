"""
This code generates decorrelation matrix for influence radio like: 3, 5, 10, 15, 30, 45
the number of latitudes has to be an even number!
"""

import numpy as np

N_LATS = 73
N_LONS = 144

j_temp = list(range(-(N_LONS//2), (N_LONS//2)))  # To center any column: (i,0)


def get_distances_matrix():
    """Compute distances matrix "L" based on Euclidian distance
    Returns:
        np.array: distances matrix L
    """

    D = np.zeros((N_LATS*N_LONS, N_LATS*N_LONS))

    for i in range(N_LATS):
        for j in range(N_LONS):
            for ii in range(N_LATS):
                for jj in range(N_LONS):
                    delta_i = abs(i-ii)
                    delta_j = abs(j-jj)
                    D[i*N_LONS+j, ii*N_LONS+jj] = delta_i**2 + \
                        min(delta_j, N_LONS - delta_j)**2
    return D


print('Generating distances matrix')
D = get_distances_matrix()

for r in [3, 5, 10, 15, 30, 45]:
    print(f'Generating decorrelation matrix for influence radio: {r}')
    np.save(
        f"./data_assimilation/decorrelation_matrices/decorrelation_r{r}", np.exp(-.5*D/r**2))
