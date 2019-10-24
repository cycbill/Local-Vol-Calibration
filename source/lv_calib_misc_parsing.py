import numpy as np
import pandas as pd
import xlwings as xw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotting import plot3d_multi_line1, plot3d_multi_line2

## Read excel
df = pd.read_excel("../processed/LV_CALIB_MISC_CPU_mpph2014-s38_27578_Fx0_Q0_B0.xlsx",header=None)


n_row = df.shape[0]
n_col = df.shape[1]

## Get number of maturities
n_mat = 0
row_indices_mat = []
for i_row, row in df.iterrows():
    if isinstance(row[0], str):
        if row[0].strip() == 'Option Maturity':
            n_mat += 1
            row_indices_mat.append(i_row)
print('number of maturities: ', n_mat)
print(row_indices_mat)

## Get info at each maturity
maturities  = np.zeros(n_mat)
nt          = np.zeros(n_mat)
ns          = np.zeros(n_mat)
xmin        = np.zeros(n_mat)
xmax        = np.zeros(n_mat)
# Input Smile
input_money             = np.zeros((n_mat, 5))      # Money
input_bs_price          = np.zeros((n_mat, 5))      # BlackPrice
input_bs_fwd_pde_price  = np.zeros((n_mat, 5))      # Fwd PDE BlackPrice
# Loc Vol First Guess 
input_logmoney          = np.zeros((n_mat, 7))      # LogMoney
input_loc_vol_guess     = np.zeros((n_mat, 7))      # LocVol
# Loc Vol LV Calib
output_lv_fwd_pde_price = np.zeros((n_mat, 5))      # PDE Price
output_price_error      = np.zeros((n_mat, 5))      # Calib IV Error (in bp)
output_loc_vol          = np.zeros((n_mat, 7))      # LocVol


for i, index_mat in enumerate(row_indices_mat):
    maturities[i]   = df.iloc[index_mat, 1]
    nt[i]           = df.iloc[index_mat+2, 1]
    ns[i]           = df.iloc[index_mat+5, 1]
    xmin[i]         = df.iloc[index_mat+5, 3]
    xmax[i]         = df.iloc[index_mat+5, 5]
    input_money[i]            = df.iloc[index_mat+11, 1:6]
    input_bs_price[i]         = df.iloc[index_mat+12, 1:6]
    input_bs_fwd_pde_price[i] = df.iloc[index_mat+13, 1:6]
    input_logmoney[i]         = df.iloc[index_mat+21, 1:8]
    input_loc_vol_guess[i]    = df.iloc[index_mat+22, 1:8]
    output_lv_fwd_pde_price[i] = df.iloc[index_mat+27, 1:6]
    output_price_error[i]     = df.iloc[index_mat+28, 1:6]
    output_loc_vol[i]         = df.iloc[index_mat+31, 1:8]

## Write market data
wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['LV_CALIB_MISC']

sht.range('B4').options(transpose=True).value = np.arange(1, n_mat+1)
sht.range('C4').options(transpose=True).value = maturities
sht.range('D4').options(transpose=True).value = nt
sht.range('E4').options(transpose=True).value = ns
sht.range('F4').options(transpose=True).value = xmin
sht.range('G4').options(transpose=True).value = xmax

sht.range('J4').value = input_money
sht.range('Q4').value = input_logmoney

sht.range('I27').value = input_loc_vol_guess
#sht.range('J4').value = input_logmoney
sht.range('Z27').value = input_bs_fwd_pde_price

sht.range('I50').value = output_loc_vol
#sht.range('J4').value = input_logmoney
sht.range('Z50').value = output_lv_fwd_pde_price

## Generate plots
'''
quotes_list5 = ['10P','25P','ATM','25C','10C']
quotes_list7 = ['min','10P','25P','ATM','25C','10C', 'max']
fig = plot3d_multi_line2(quotes_list7, np.arange(0, n_mat), input_loc_vol_guess, 'Quote', 'Maturity', 'Loc Vol Guess')

sht.pictures.add(fig, name='Loc Vol Guess', update=True,
                 left = sht.range('A26').left, top = sht.range('A26').top)
'''

for i, index_mat in enumerate(row_indices_mat):
    money_grid = df.iloc[index_mat+35, 1:(int(ns[i])+1)]
    callprice_grid = df.iloc[index_mat+36, 1:(int(ns[i])+1)]
    sht.range('D'+str(i*2+96)).value = np.array(money_grid)
    sht.range('D'+str(i*2+97)).value = np.array(callprice_grid)

