import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from vtk_visualization import vtk_visualization

folder = r'D:\OneDrive - Murex\Desktop\Local Vol Calibration\data\Local Vol Macs Traces\Vanilla TARF'
file = 'pde1d_mx21483vm_11652_22.vtk'

vtk_data = pd.read_csv(folder + '\\' + file)
vtk_data.columns = ['Data']


data = vtk_data.iloc[:,0].str.find('POINTS')


a_idx = vtk_data.index[vtk_data['Data'].str.contains('DATASET STRUCTURED_GRID')].tolist()[0]

b_idx_list = vtk_data.index[vtk_data['Data'].str.contains('SCALARS Op_Q0_Npv double')].tolist()
if b_idx_list != []:
    b_idx = b_idx_list[0]
else:
    b_idx = vtk_data.index[vtk_data['Data'].str.contains('SCALARS Npv_Q0_Npv double')].tolist()[0]


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


value_table = vtk_data.loc[value_start_idx:value_end_idx, :]
value_table2 = value_table['Data'].str.split(' ', expand=True)
value_table3 = value_table2.values
value_table4 = value_table3[:,:-1]
value_table5 = value_table4.astype(float)


k, t = np.meshgrid(k_grid, t_grid)
prem = value_table5


vtk_visualization(k_grid, t_grid, k, t, prem)