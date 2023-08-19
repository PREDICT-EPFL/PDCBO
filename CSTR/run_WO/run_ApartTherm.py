import numpy as np
from ApartmentsThermal import get_ApartTherm_kpis

num_discret = 4
grid_energy_mat = np.zeros((num_discret, num_discret))
grid_dev_mat = np.zeros((num_discret, num_discret))
lower_tol_range = [0.0, 0.5]
upper_tol_range = [0.5, 1.0]

for j in range(num_discret):
    for k in range(num_discret):
        lower_tol = lower_tol_range[0] + j / num_discret * (
            lower_tol_range[1] - lower_tol_range[0])
        upper_tol = upper_tol_range[0] + k / num_discret * (
            upper_tol_range[1] - upper_tol_range[0])
        energy, avg_dev = get_ApartTherm_kpis(lower_tol, upper_tol)
        grid_energy_mat[j, k] = energy
        grid_dev_mat[j, k] = avg_dev

np.savez('./data/apart_therm_grid_data', grid_energy_mat, grid_dev_mat)
