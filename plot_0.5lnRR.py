import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits import basemap
import warnings
from pylab import *
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from pyhdf.SD import SD
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import TwoSlopeNorm
import os

warnings.filterwarnings('ignore')

def calculate_are(mask, nlats, EARTH_RADIUS, dlat_rad, dlon_rad):
    valid_lats = np.deg2rad(nlats[~mask])
    area = (EARTH_RADIUS ** 2) * dlat_rad * dlon_rad * np.cos(valid_lats)
    total_area = np.sum(area)

    total_area_ha = total_area / 10000

    return total_area, total_area_ha

crop_path = r'./Cropland.xlsx'
cropland = pd.read_excel(crop_path).values
masked_crop = np.ma.masked_where(cropland == -1000, cropland)
mask_land = masked_crop.mask

EARTH_RADIUS = 6371000

dlatout = 0.5  # size of lat grid
dlonout = 0.5  # size of lon grid

latsize = int(180 / dlatout)  # as integer
lonsize = int(360 / dlonout)  # as integer
area = np.zeros((latsize, lonsize,))
outlats = np.arange(-90 + dlatout / 2, 90, dlatout)
outlons = np.arange(-180 + dlonout / 2, 180, dlonout)
nlons, nlats = np.meshgrid(outlons, outlats)
dlat_rad = np.deg2rad(dlatout)
dlon_rad = np.deg2rad(dlonout)
total_crop, total_crop_ha = calculate_are(mask_land, nlats, EARTH_RADIUS, dlat_rad, dlon_rad)

print(' opening file: ' + str(path1))
SOC = pd.read_excel(path1, sheet_name='SOC', index_col=0).values
NL = pd.read_excel(path1,  sheet_name='NL', index_col=0).values
CO2 = pd.read_excel(path1,  sheet_name='CO2', index_col=0).values
N2O = pd.read_excel(path1,  sheet_name='N2O', index_col=0).values

valid = ((SOC == -1000) & (NL == -1000) & (CO2 == -1000) & (N2O == -1000))

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 21}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
materials = [SOC, NL, CO2, N2O]

name_list = ["Predicted lnRR (SOC stock)", "Predicted lnRR (Nitrate leaching)",
             "Predicted lnRR (CO${_2}$ emission)", "Predicted lnRR (N${_2}$O emission)"]

material_list = ["SOC stock: ", "Nitrate leaching:",
                 "CO${_2}$ emission:", "N${_2}$O emission:"]

colors_dict = [
    (0.0, 0.2, 1.0),
    (0.2, 0.4, 1.0),
    (0.0, 0.8, 1.0),
    (0.4, 0.8, 1.0),
    (0.6, 0.8, 1.0),
    (1.0, 0.8, 0.4),
    (1.0, 0.6, 0.3),
    (1.0, 0.4, 0.2),
    (1.0, 0.2, 0.1),
    (1.0, 0.0, 0.0),
]

bounds = np.array([-0.8, -0.5, -0.2, -0.1, -0.05, 0, +0.05, +0.1, +0.2, +0.5, +0.8])
cmap_custom = ListedColormap(colors_dict)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(colors_dict))

fig = plt.figure(figsize=(15, 9))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.1)

# 使用自定义色带
cmap = cmap_custom
cmap.set_under('white')
cmap.set_over('#E65100')

masked_soc = np.ma.masked_where((SOC <= 0.05) | (SOC == -1000), SOC)
masked_nl = np.ma.masked_where((NL >= -0.05) | (NL == -1000), NL)
masked_CO2 = np.ma.masked_where((CO2 >= -0.05) | (CO2 == -1000), CO2)
masked_N2O = np.ma.masked_where((N2O >= -0.05) | (N2O == -1000), N2O)

mask = (masked_soc.mask | masked_N2O.mask | masked_nl.mask | masked_CO2.mask)
mask2 = (np.ma.masked_where((SOC == -1000), SOC).mask |
         np.ma.masked_where((NL == -1000), NL).mask |
         np.ma.masked_where((CO2 == -1000), CO2).mask |
         np.ma.masked_where((N2O == -1000), N2O).mask)
total_area, total_area_ha = calculate_are(mask, nlats, EARTH_RADIUS, dlat_rad, dlon_rad)
_, total_area_ha_all = calculate_are(mask2, nlats, EARTH_RADIUS, dlat_rad, dlon_rad)
percent = total_area_ha / total_area_ha_all
print(f"{name_list[0]} >5% 面积: {total_area_ha:.2f} 公顷 (占农田 {percent * 100:.1f}%)")

ax = plt.gca()

# 背景色（整个画布）
ax.set_facecolor('lightgray')
m = basemap.Basemap(
    projection='cyl',
    llcrnrlon=-180,
    urcrnrlon=180,
    llcrnrlat=-60,
    urcrnrlat=85,
    resolution='c')
# 海洋
m.drawmapboundary(fill_color='white')

# 陆地
m.fillcontinents(color='lightgray', lake_color='white')

# 海岸线
m.drawcoastlines(linewidth=0.5, color='black')

# 国界线
m.drawcountries(linewidth=0.3, color='black')

masked = (~mask & ~mask2)
nlons = nlons[masked]
nlats = nlats[masked]

cs = m.scatter(
    nlons,
    nlats,
    c='red',
    s=1,
    latlon=True,
    cmap="RdYlGn_r",
    norm=norm
)

plt.tight_layout()
plt.savefig('./figures/0.5_lnRR.pdf'.format(fnt), dpi=400, bbox_inches='tight')
plt.show()
