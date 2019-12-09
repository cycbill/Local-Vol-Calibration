import numpy as np 
import xlwings as xw 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['VTK']

t_grids = sht.range('A2').options(np.array, expand='down').value
k_grids = sht.range('B1').options(np.array, expand='right').value
v_values = sht.range('B2').options(np.array, expand='table').value

print(v_values.shape)

k, t = np.meshgrid(k_grids, t_grids)

## VTK plot
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(k, t, v_values, cmap=cm.coolwarm)
ax.set_xlabel('k')
ax.set_ylabel('t')
ax.set_zlabel('v')
ax.set_title('VTK')
plt.show()

## T steps plot
fig = plt.figure(figsize=plt.figaspect(0.5))
fig.add_subplot(121)
ax1 = fig.add_subplot(121)
ax1.plot(t_grids, '.-', linewidth=1.0, ms=4.0)
ax1.grid(True, which='both')

ax2 = fig.add_subplot(122)
ax2.plot(-1.0 * np.diff(t_grids), '.-', linewidth=1.0, ms=4.0)
ax2.grid(True, which='both')
ax2.set_ylim(bottom = 0.0)

plt.suptitle('T steps')
plt.show()

## k steps plot
fig = plt.figure(figsize=plt.figaspect(0.5))
fig.add_subplot(121)
ax1 = fig.add_subplot(121)
ax1.plot(k_grids, '.-', linewidth=1.0, ms=4.0)
ax1.grid(True, which='both')

ax2 = fig.add_subplot(122)
ax2.plot(np.diff(k_grids), '.-', linewidth=1.0, ms=4.0)
ax2.grid(True, which='both')
ax2.set_ylim(bottom = 0.0)

plt.suptitle('k steps')
plt.show()