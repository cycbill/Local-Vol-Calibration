import numpy as np
import xml.etree.ElementTree as ET
import xlwings as xw

def stringToNumpyArray(string, split_str):
    str_list = string.split(split_str)
    np_array = np.array([float(elem.strip()) for elem in str_list])
    return np_array


tree = ET.parse('../processed/irsmform_mx21483vm_5939_0_main_0.xml')
root = tree.getroot()

## zero curve 1
OneDCurve = root.findall('./ProcessesList/Processes/Process/RateCurve/OneDCurve')[0]
CurveLabel = OneDCurve.find('CurveLabel')
Buckets = OneDCurve.find('Buckets')
Values = OneDCurve.find('Values')
curve1_label = CurveLabel.text
curve1_maturities = stringToNumpyArray(Buckets.text, ';')[1:]
curve1_values = stringToNumpyArray(Values.text, ';')[1:]

## zero curve 2
OneDCurve = root.findall('./ProcessesList/Processes/Process/RateCurve/OneDCurve')[1]
CurveLabel = OneDCurve.find('CurveLabel')
Buckets = OneDCurve.find('Buckets')
Values = OneDCurve.find('Values')
curve2_label = CurveLabel.text
curve2_maturities = stringToNumpyArray(Buckets.text, ';')[1:]
curve2_values = stringToNumpyArray(Values.text, ';')[1:]

## Capitalization Spread Curve
CapitalizationSpreadCurve = root.find('.//CapitalizationSpreadCurve')
Buckets = CapitalizationSpreadCurve.find('./OneDCurve/Buckets')
Values = CapitalizationSpreadCurve.find('./OneDCurve/Values')
spreadcurve_label = CapitalizationSpreadCurve.tag
spreadcurve_maturities = stringToNumpyArray(Buckets.text, ';')[1:]
spreadcurve_values = stringToNumpyArray(Values.text, ';')[1:]

## Volatility Curve
VolatilityCurve = root.find('.//VolatilityCurve')
Buckets = VolatilityCurve.find('./OneDCurve/Buckets')
Values = VolatilityCurve.find('./OneDCurve/Values')
vol_label = VolatilityCurve.tag
vol_maturities = stringToNumpyArray(Buckets.text, ';')[1:]
vol_values = stringToNumpyArray(Values.text, ';')[1:]

## CalibrationBaskets
len_smile_mat = len(vol_maturities)
strikes = np.zeros((len_smile_mat,5))
blackVols = np.zeros((len_smile_mat,5))
for idx, CalibrationInstrument in enumerate(root.findall('.//CalibrationInstrument')):
    row = idx // 5
    column = idx % 5
    Strike = CalibrationInstrument.find('Strike')
    strikes[row, column] = float(Strike.text)
    BlackVol = CalibrationInstrument.find('BlackVol')
    blackVols[row, column] = float(BlackVol.text)
print(strikes)

## Write market data
wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['IRSMFORM']

sht.range('B4').options(transpose=True).value = curve1_maturities
sht.range('C4').options(transpose=True).value = curve1_values
sht.range('F4').options(transpose=True).value = curve2_maturities
sht.range('G4').options(transpose=True).value = curve2_values
sht.range('J4').options(transpose=True).value = spreadcurve_maturities
sht.range('K4').options(transpose=True).value = spreadcurve_values
sht.range('N4').options(transpose=True).value = vol_maturities
sht.range('O4').options(transpose=True).value = vol_values
sht.range('R4').value = strikes
sht.range('Y4').value = blackVols


