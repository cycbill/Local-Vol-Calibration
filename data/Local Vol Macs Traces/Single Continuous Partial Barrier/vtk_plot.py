import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

folder = 'D:\OneDrive - Murex\Desktop\Local Vol Macs Traces\Single Continuous Partial Barrier'
file = 'pde1d_mx21483vm_11652_9.vtk'

vtk_data = pd.read_csv(folder + '\\' + file)
vtk_data.columns = ['Data']


data = vtk_data.iloc[:,0].str.find('POINTS')


a_idx = vtk_data.index[vtk_data['Data'].str.contains('DATASET STRUCTURED_GRID')].tolist()[0]
b_idx = vtk_data.index[vtk_data['Data'].str.contains('SCALARS Op_Q0_Npv double')].tolist()[0]
c_idx = vtk_data.index[vtk_data['Data'].str.contains('SCALARS Spot double')].tolist()[0]

grid_start_idx = a_idx + 3
grid_end_idx = b_idx - 2
value_start_idx = b_idx + 2
value_end_idx = c_idx - 1


grid_table = vtk_data.loc[grid_start_idx:grid_end_idx, :]
grid_table2 = grid_table['Data'].str.split(' ', expand=True)
grid_table3 = grid_table2.values
grid_table4 = grid_table3[:,:-1]
grid_table5 = grid_table4.astype(float)


t_grid = grid_table5[:, 1]
k_index = [i for i in range(0, grid_table5.shape[1], 3)]
k_grid = grid_table5[0, k_index]
print(k_grid)


value_table = vtk_data.loc[value_start_idx:value_end_idx, :]
value_table2 = value_table['Data'].str.split(' ', expand=True)
value_table3 = value_table2.values
value_table4 = value_table3[:,:-1]
value_table5 = value_table4.astype(float)
print(value_table5)
prem = value_table5
print(prem.shape, len(t_grid), len(k_grid))

fig = plt.figure()
ax = fig.gca(projection='3d')
k, t = np.meshgrid(k_grid, t_grid)
ax.plot_surface(k, t, prem, cmap=cm.coolwarm)
plt.show()


'''
for row in grid_table3:
    print(type(row))
    grid_table5 = np.fromstring(row, sep=' ')
    print(grid_table5)

#grid_table = grid_table['Data'].str.rstrip()
df_value = vtk_data.loc[value_start_idx:value_end_idx, :]
#df_value = df_value['Data'].str.rstrip()

grid_table2 = grid_table['Data'].str.split(' ', expand=True) #

t_grid = grid_table2[1]


np_grid_string = grid_table2.values
#np_grid = np_grid_string.astype(np.float)
print(np_grid_string)
'''