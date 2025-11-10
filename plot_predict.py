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
import os

warnings.filterwarnings('ignore')

def calculate_are(mask, nlats, EARTH_RADIUS, dlat_rad, dlon_rad):
    valid_lats = np.deg2rad(nlats[~mask])
    area = (EARTH_RADIUS ** 2) * dlat_rad * dlon_rad * np.cos(valid_lats)
    total_area = np.sum(area)

    total_area_ha = total_area / 10000

    return total_area, total_area_ha


crop_path = r'./Cropcombination.xlsx'
cropland = pd.read_excel(crop_path).values
# masked_crop = np.ma.masked_where(cropland == -1000, cropland)
masked_crop = np.ma.masked_where(cropland == -1000, cropland)
mask_land = masked_crop.mask
# # non_crop_mask = (cropland == -1000)

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
# print('dlat_rad', dlat_rad)

total_crop, total_crop_ha = calculate_are(mask_land, nlats, EARTH_RADIUS, dlat_rad, dlon_rad)

path = r'./lnRR.xlsx'

print(' opening file: ' + str(path))
SOC = pd.read_excel(path, sheet_name='SOC', index_col=0).values
NL = pd.read_excel(path,  sheet_name='NL', index_col=0).values
CO2 = pd.read_excel(path,  sheet_name='CO2', index_col=0).values
N2O = pd.read_excel(path,  sheet_name='N2O', index_col=0).values

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 21}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
# materials = [SOC, NL, CO2, N2O]
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
            (1.0, 0.0, 0.0),]

bounds = np.array([-0.8, -0.5, -0.2, -0.1, -0.05, 0, +0.05, +0.1, +0.2, +0.5, +0.8])
cmap_custom = ListedColormap(colors_dict)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(colors_dict))

fig = plt.figure(figsize=(15, 9))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.1)

percentage = []
square = []
text_stats = []
mask = np.zeros_like(SOC)

for j in range(4):
    fig.add_subplot(2, 2, j+1)
    plt.title(name_list[j], fontdict=font1, loc='center', y=1.05)

    # 使用自定义色带
    cmap = cmap_custom
    cmap.set_under('white')
    cmap.set_over('#E65100')

    if j == 0:
        masked_data = np.ma.masked_where((materials[j] >= -0.05) | (materials[j] == -1000), materials[j])
        masked_data2 = np.ma.masked_where((materials[j] <= 0.05) | (materials[j] == -1000), materials[j])
        # mask = masked_data.mask
    else:
        masked_data = np.ma.masked_where((materials[j] <= 0.05) | (materials[j] == -1000), materials[j])
        masked_data2 = np.ma.masked_where((materials[j] >= -0.05) | (materials[j] == -1000), materials[j])
        # mask = masked_data.mask
    # masked_data = np.ma.masked_where((materials[j] <= 0.1) | (materials[j] == -1000), materials[j])
    # masked_data = np.ma.masked_where((materials[j] >= -0.05) | (materials[j] == -1000), materials[j])
    # # masked_data = np.ma.masked_where((materials[j] >= -0.1) | (materials[j] == -1000), materials[j])
    # # masked_data = np.ma.masked_where((materials[j] <= -0.02) | (materials[j] >= 0.02) | (materials[j] == -1000), materials[j])
    mask = masked_data.mask
    mask2 = masked_data2.mask
    #
    # # 获取有效点的纬度(弧度)
    total_area, total_area_ha = calculate_are(mask, nlats, EARTH_RADIUS, dlat_rad, dlon_rad)
    _, total_area_ha2 = calculate_are(mask2, nlats, EARTH_RADIUS, dlat_rad, dlon_rad)
    percent = total_area_ha / total_crop_ha
    percent2 = total_area_ha2 / total_crop_ha

    # print('%f percent cropland Increased.' % (percent))
    print(f"{name_list[j]} >5% 面积: {total_area_ha:.2f} 公顷 (占农田 {percent * 100:.1f}%)")
    print(f"{name_list[j]} <5% 面积: {total_area_ha2:.2f} 公顷 (占农田 {percent2 * 100:.1f}%)")
    # text_stats.append(f"{material_list[j]} Area: {total_area_ha:.2f} ha (Percentage in Cropland) {percent * 100:.1f}%")

    percentage.append(total_area_ha)
    square.append(percent * 100)
    #
    m = basemap.Basemap(projection='robin', lon_0=-180, resolution='c')
    # m.
    m.drawmapboundary(color='k', fill_color='none')
    m.drawcoastlines(color='k', linewidth=0.4)
    # add longitude and latitude lines
    m.drawmeridians(
        np.arange(0, 720, 60),  # set real range of longitude
        color='gray',
        linewidth=0.5,
        labels=[-1, True, 0, True],
        fontsize=10
    )
    m.drawparallels(
        np.arange(-90, 90, 30),  # set real range of longitude
        color='gray',
        linewidth=0.5,
        labels=[1, True, 0, 1],
        fontsize=10
    )
    # print(landmap.shape, nlons2.shape, nlats2.shape)
    # im1 = m.pcolormesh(nlons2, nlats2, cropfile, alpha=0.3, latlon=True, vmin=0.9, vmax=1.1)
    im1 = m.pcolormesh(nlons, nlats, materials[j], norm=norm, cmap=cmap_custom, latlon=True)
    # im1 = m.pcolormesh(nlons, nlats, masked_data, norm=norm, cmap=cmap_custom, latlon=True)


cb_ax = fig.add_axes([0.485, 0.2, 0.01, 0.7]) #设置colarbar位置
# cb_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = fig.colorbar(im1, cax=cb_ax, pad="4%", extend="both")
cbar.set_ticks(bounds)  # 设置所有bounds为刻度
cbar.set_ticklabels([f'{x:.2f}' for x in bounds])  # 显示全部刻度值
cbar.ax.tick_params(labelsize=10)  # 调整刻度标签大小

# text_stats1 = (
#     text_stats[0] + '\n' + text_stats[2])
#
# text_stats2 = (
#     text_stats[1] + '\n' + text_stats[3])
# # print(text_stats)

# plt.figtext(0.27, 0.05, text_stats1,
#             bbox=dict(
#             boxstyle='round',
#             facecolor='#E0FFFF',
#             alpha=0.8,
#             edgecolor='#E0FFFF'),
#             fontsize=10,
#             ha='center')
# plt.figtext(0.75, 0.05, text_stats2,
#             bbox=dict(
#             boxstyle='round',
#             facecolor='#E0FFFF',
#             alpha=0.8,
#             edgecolor='#E0FFFF'),
#             fontsize=10,
#             ha='center')
plt.savefig('E:\ArticalWriting\Adaboost_Regression\\figures\Predict_lnRR.png', dpi=600, bbox_inches='tight')
plt.show()