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

path = r'./Figures&Tables/Fig.S11.xlsx'
label_list = ['SOC stock', 'Nitrate leaching', 'CO${_2}$ emission', 'N${_2}$O emission']  #
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

    value_silt = []
    value_soc = []
    value_ph = []
    value_sand = []
    value_mat = []
    value_clay = []
    value_map = []
    value_bd = []
    value_fnt = []

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

        value_silt.append(str2array(data['PDP'][8]))
        value_soc.append(str2array(data['PDP'][5]))
        value_ph.append(str2array(data['PDP'][4]))
        value_sand.append(str2array(data['PDP'][7]))
        value_mat.append(str2array(data['PDP'][1]))
        value_clay.append(str2array(data['PDP'][6]))
        value_map.append(str2array(data['PDP'][2]))
        value_bd.append(str2array(data['PDP'][3]))
        value_fnt.append(str2array(data['PDP'][0]))

    materials_value = [value_fnt, value_mat, value_map, value_bd, value_ph, value_soc, value_clay, value_sand, value_silt]  
    materials_index = [index_fnt, index_mat, index_map, index_bd, index_ph, index_soc, index_clay, index_sand, index_silt]  
    name_list = ['(a) FNT (kg N ha$^{-1}$)', '(b) MAT ($^\circ$C)', '(c) MAP (mm/yr)', '(d) BD (g/cm3)', '(e) ISOC (%weight)', '(f) pH (-log(H+))',
                 '(g) Clay (%wt)', '(h) Sand (%wt)', '(i) Silt (%wt)']       

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

    labelss = plt.legend(handles=patch, ncol=1, loc="lower right", bbox_to_anchor=(-1.4, 2.65), fontsize="large").get_texts()
    [label.set_fontname('Times New Roman') for label in labelss]
    plt.tight_layout()
    plt.savefig('./features.png', dpi=400, bbox_inches='tight')
    plt.show()

 
plot_features(path)


