import numpy as np 
import xlwings as xw 

def print_to_excel(a, b):
    wb = xw.Book('LocVol Parameters.xlsx')
    sht = wb.sheets['Testing']

    sht.range('A2').options(transpose=True).value = a
    sht.range('B2').options(transpose=True).value = b

def read_from_excel():
    wb = xw.Book('LocVol Parameters.xlsx')
    sht = wb.sheets['Testing']

    k_prev = sht.range('A2').options(np.array, expand='down').value
    price_prev = sht.range('B2').options(np.array, expand='down').value

    return k_prev, price_prev