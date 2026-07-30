import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib
import math
from netCDF4 import Dataset
import numpy as np

from mpl_toolkits import basemap
# from pyhdf.SD import SD
from pylab import *
#
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
#
import os

def absolute_increase(lnRR, T, item):
    # print('lnRR', lnRR.shape, C.shape)
    if item == 'SOC':
        abs = (1 - np.exp(-lnRR)) * T
        imp = np.exp(lnRR) - 1
    else:
        abs = ( np.exp(-lnRR) - 1) * T
        imp = 1 - np.exp(lnRR)

    return abs, imp


def calculate_are(mask, nlats, EARTH_RADIUS, dlat_rad, dlon_rad):
    valid_lats = np.deg2rad(nlats[~mask])
    area = (EARTH_RADIUS ** 2) * dlat_rad * dlon_rad * np.cos(valid_lats)
    total_area = np.sum(area)

    total_area_ha = total_area / 10000

    return total_area, total_area_ha
#

bdfilename = "./bd.xlsx"
bd = pd.read_excel(bdfilename, index_col=0).values

path = r'.\Figures&Tables\Fig.6.xlsx'  #'./prediction/lnRR.xlsx'

path_delta = r'.\Figures&Tables\Pred_C1.xlsx'
sheet_name = ['SOC', 'NL', 'CO2', 'N2O']
set_value = [0.05, -0.05, -0.05, -0.05]
# set_value = [0.03, -0.006, -0.002, -0.02]
material = ['Gt', 'Tg', 'Gt', 'Gt']

EARTH_RADIUS = 6371000

dlatout = 0.5  # size of lat grid
dlonout = 0.5  # size of lon grid

outlats = np.arange(-90 + dlatout / 2, 90, dlatout)
outlons = np.arange(-180 + dlonout / 2, 180, dlonout)
nlons, nlats = np.meshgrid(outlons, outlats)
dlat_rad = np.deg2rad(dlatout)
dlon_rad = np.deg2rad(dlonout)
area = np.zeros_like(nlats)

for i in range(4):
    lnRR = pd.read_excel(path, sheet_name=sheet_name[i], index_col=0).values
    NL_Abs = pd.read_excel(path_delta, sheet_name=sheet_name[i], index_col=0).values     #header=None

    item = sheet_name[i].split('.')[0]

    if  item == 'SOC':
        lnRR = np.ma.masked_where((lnRR <= set_value[i]) | (lnRR == -1000), lnRR)
        mask = lnRR.mask
    else:
        lnRR = np.ma.masked_where((lnRR >= set_value[i]) | (lnRR == -1000), lnRR)
        mask = lnRR.mask

    #sheet_name = ['SOC', 'NL', 'CO2', 'N2O']
    # material = ['Mg/ha', 'Kg/ha', 'Mg/ha', 'Mg/ha']
    if item == 'NL':
        NL_Abs /= 1e9  # 1Tg=10⁶ Mg=10⁹ Kg
    elif item == 'SOC':
        NL_Abs /= 1e9  # 1Tg=10^6 Mg
    else:
        NL_Abs /= 1e9 # 1Gt = 1000 Tg=10⁹ Mg

    lnRR = np.abs(lnRR)
    delta, improve = absolute_increase(lnRR, NL_Abs, item)

    valid_lats = np.deg2rad(nlats[~mask])
    area[~mask] = (EARTH_RADIUS ** 2) * dlat_rad * dlon_rad * np.cos(valid_lats) / 1e4
    area[mask] = 0.

    # sum = np.sum(delta * bd * area)
    sum = np.sum(delta * area)
    improvement = np.mean(improve) * 100
    print('improvement', improvement)

    print(f"\n{ sheet_name[i].split('.')[0]}  Absolute Increasement: {sum} {material[i]}  Improvement: {improvement}")



