import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter

import re

import matplotlib.colors as colors
from mpl_toolkits import basemap
import warnings
warnings.filterwarnings('ignore')

# path = r'F:\Project_Code\Adaboost_Regression\Particial\Particial.xlsx'
path = r'K:/ArticalWriting/NatureCommunications/Adaboost_Regression/Global_Prediction/Tables/final/Submit/new/Figures&Tables2/Fig.S11.xlsx'

def append_trace(ma):
    y = ma[:][1]
    x = ma[:][2]
    fig2, axes = plt.subplots()
    for j in range(4):
        _y = list(float(digit) for digit in y[j].split(',') if digit != '[' and digit != ']')
        _x = list(float(digit) for digit in x[j].split(',') if digit != '[' and digit != ']')

        sub = axes.plot(_x, _y, color=color[j], linewidth=2)


def plot_bar(path):
    path = r'D:\zcl\Particial.xlsx'
    MAT = pd.read_excel(path, sheet_name='MAT')
    MAP = pd.read_excel(path, sheet_name='MAP')
    BD = pd.read_excel(path, sheet_name='BD')
    ISOC = pd.read_excel(path, sheet_name='ISOC')
    pH = pd.read_excel(path, sheet_name='pH')
    Clay = pd.read_excel(path, sheet_name='Clay')
    Cropcombination = pd.read_excel(path, sheet_name='CropCombination')

    height = 0.2
    a1 = list(range(4))
    a2 = [i + height * 1 for i in a1]
    a3 = [i + height * 2 for i in a1]
    a4 = [i + height * 3 for i in a1]
    a5 = [i + height * 4 for i in a1]
    # a6 = [i + height * 5 for i in a1]
    # a7 = [i + height * 6 for i in a1]
    # a8 = [i + height * 7 for i in a1]
    # a9 = [i + height * 8 for i in a1]

    #plot
    plt.figure(figsize=(15, 12), dpi=200)
    plt.barh(a1, MAT['lnRR'], height=height, label='Planting', color='pink')
    plt.barh(a2, MAT['LNRR'], height=height, label='MAP', color='purple')
    plt.barh(a3, MAT['mat'], height=height, label='MAT', color='green')
    plt.barh(a4, MAT['bd'], height=height, label='BD', color='blue')
    plt.barh(a5, MAT['soc'], height=height, label='SOC', color='purple')

    plt.grid(alpha=0.4)
    # plt.yticks(a[2], )
    plt.legend()
    plt.show()

def str2array(str_list):
    x_list = re.split(',', str_list[1:-1])
    _x_list = []
    for x in x_list:
        x = float(x)
        _x_list.append(x)
    # x_list = str_list.split(',')
    x_list = np.array(_x_list)
    return x_list

# materials = ['SOC', 'NL', 'CO2', 'N2O']
label_list = ['SOC stock', 'Nitrate leaching', 'CO${_2}$ emission', 'N${_2}$O emission']  #
# color = ["red", "green", "blue", "orange"]  #
color = ['#004e66', '#fcbe32', '#ff5f2e', '#77919d']  #, '#ff5f2e', '#77919d'
patch = [mpatches.Patch(color=color[l], label=label_list[l]) for l in range(len(color))]

def plot_features(path):
    SOC = pd.read_excel(path, sheet_name='PDP_SOC', index_col=0)
    NL = pd.read_excel(path, sheet_name='PDP_NL', index_col=0)
    CO2 = pd.read_excel(path, sheet_name='PDP_CO2', index_col=0)
    N2O = pd.read_excel(path, sheet_name='PDP_N2O', index_col=0)


    #plot seetings
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 21,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }

    index_silt = []
    index_soc = []
    index_ph = []
    index_sand = []
    index_mat = []
    index_clay = []
    index_map = []
    index_bd = []
    index_fnt = []
    # index_organism = []
    # index_rootbio = []
    # index_period = []

    value_silt = []
    value_soc = []
    value_ph = []
    value_sand = []
    value_mat = []
    value_clay = []
    value_map = []
    value_bd = []
    value_fnt = []
    # value_organism = []
    # value_rootbio = []
    # value_period = []

    for index, data in enumerate([SOC, NL, CO2, N2O]):  #, N2O
        index_silt.append(str2array(data['Index'][8]))
        index_soc.append(str2array(data['Index'][5]))
        index_ph.append(str2array(data['Index'][4]))
        index_sand.append(str2array(data['Index'][7]))
        index_mat.append(str2array(data['Index'][1]))
        index_clay.append(str2array(data['Index'][6]))
        index_map.append(str2array(data['Index'][2]))
        index_bd.append(str2array(data['Index'][3]))
        index_fnt.append(str2array(data['Index'][0]))
        # index_organism.append(str2array(data['Index'][9]))
        # index_rootbio.append(str2array(data['Index'][10]))
        # index_period.append(str2array(data['Index'][11]))

        value_silt.append(str2array(data['PDP'][8]))
        value_soc.append(str2array(data['PDP'][5]))
        value_ph.append(str2array(data['PDP'][4]))
        value_sand.append(str2array(data['PDP'][7]))
        value_mat.append(str2array(data['PDP'][1]))
        value_clay.append(str2array(data['PDP'][6]))
        value_map.append(str2array(data['PDP'][2]))
        value_bd.append(str2array(data['PDP'][3]))
        value_fnt.append(str2array(data['PDP'][0]))
        # value_organism.append(str2array(data['PDP'][9]))
        # value_rootbio.append(str2array(data['PDP'][10]))
        # value_period.append(str2array(data['PDP'][11]))

    #
    # color_list = [(123, 129, 164), (209, 187, 129), (189, 163, 156), (111, 127, 102), (241, 64, 64), (24, 4, 5), (20, 11, 109), (53, 65, 221)]
    # print('index_soc', index_soc)
    materials_value = [value_fnt, value_mat, value_map, value_bd, value_ph, value_soc, value_clay, value_sand, value_silt]  #, value_organism, value_rootbio, value_period
    materials_index = [index_fnt, index_mat, index_map, index_bd, index_ph, index_soc, index_clay, index_sand, index_silt]  #, index_organism, index_rootbio, index_period
    # print(materials_index)
    name_list = ['(a) FNT (kg N ha$^{-1}$)', '(b) MAT ($^\circ$C)', '(c) MAP (mm/yr)', '(d) BD (g/cm3)', '(e) ISOC (%weight)', '(f) pH (-log(H+))',
                 '(g) Clay (%wt)', '(h) Sand (%wt)', '(i) Silt (%wt)',
                 ]       #'Longitude', 'Latitude', 'Organisms', 'Root Biomatics', 'Period'

    # _ym = [-0.5, -0.55, -0.3, -0.3, -0.3, -0.3, -0.55, -0.5]
    # ym = [0.2, 0.1, 0.1, 0.2, 0.5, 0.1, 0.65, 0.2]

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.subplots_adjust(hspace=0.3)
    for j in range(len(materials_index)):
        data_material = materials_value[j]
        data_index = materials_index[j]

        m = j // 3
        n = j % 3
        # (axes[m, n]).set_ylim(ymin=_ym[j], ymax=ym[j])
        if n == 0:
            (axes[m, n]).set_ylabel('lnRR', fontdict=font2)
        (axes[m, n]).set_xlabel(name_list[j], fontdict=font1)

        for i in range(len(label_list)):
            y = data_material[i]
            y_smooth = savgol_filter(
                y,
                window_length=9,   # 必须为奇数
                polyorder=1
            )
            x = data_index[i]

            axes[m, n].plot(x, y_smooth, color=color[i], linewidth=3)


    # for ind, value, name in zip(materials_index, materials_value, name_list):
    #     y = value
    #     x = ind
    #     # print('y', len(y) == len(x))
    #     # print('x', len(x))
    #     m = j // 4
    #     n = j % 4
    #     (axes[m, n]).set_ylim(ymin=_ym[j], ymax=ym[j])
    #     if n == 0:
    #         (axes[m, n]).set_ylabel('lnRR', fontdict=font2)
    #     (axes[m, n]).set_xlabel(name, fontdict=font1)
    #     # print('m is', m, 'n is', n)
    #     # print('i is', i)
    # #
    #     # for j in range(8):
    #     #     # _y = list(float(digit) for digit in y[j].split(',') if digit is not '[' and digit is not ']')
    #     #     # _x = list(float(digit) for digit in x[j].split(',') if digit is not '[' and digit is not ']')
    #     #     _y = y[j]
    #     #     _x = x[j]
    #     axes[m, n].plot(x, y, color=color[i], linewidth=3)
    #     # axes[m, n].plot(x, y, linewidth=3)
    #         # ax.plot(_x, _y, color=color[j], label=label[j], linewidth=5)
    #         # axes.legend()
    #     j += 1
    #
    # i += 1
    #
    labelss = plt.legend(handles=patch, ncol=1, loc="lower right", bbox_to_anchor=(-1.4, 2.65), fontsize="large").get_texts()
    [label.set_fontname('Times New Roman') for label in labelss]
    # plt.tight_layout()
    plt.savefig('./features.png', dpi=400, bbox_inches='tight')
    plt.show()

        # plt.savefig('D:/zcl/mat.tiff', dpi=600)
# path = r'D:\zcl\Particial.xlsx'
plot_features(path)

def RGB_to_Hex(rgb):
    # RGB = rgb.split(',')   # 将RGB格式划分开来
    color = '#'
    for i in rgb:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    print(color)
    return color

