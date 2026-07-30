import os
import time
import pandas as pd

from torch.utils.data import Dataset

# from pyhdf.SD import SD

from utils.tools import StandardScaler
from mpl_toolkits import basemap

from pylab import *
from data.global_base import read_global

import warnings
warnings.filterwarnings('ignore')

def preprocess(df,
        target='SOCT',
        redudent='lnRR',
        fit=False,
        pt=None):
    """
    数据预处理
    Parameters
    ----------
    df : DataFrame
        输入数据
    target : str
        目标变量
    fit : bool
        True：拟合PowerTransformer
        False：仅transform
    pt : PowerTransformer
        已拟合好的PowerTransformer
    Returns
    -------
    X : DataFrame
    Y : ndarray
    pt : PowerTransformer
    """

    # 拷贝数据
    dfc = df.copy()
    cols = list(dfc.columns);

    lon = dfc['Lo']
    lat = dfc['La']
    lon_block = ((lon + 180)//20).astype(int)
    lat_block = ((lat + 90)//20).astype(int)
    climate_group = lon_block * 100 + lat_block

    # 连续变量
    continuous_cols = dfc.select_dtypes(include=np.number).columns.tolist()

    # 去掉One-Hot变量
    onehot_cols = []
    for c in dfc.columns:
        if set(dfc[c].dropna().unique()).issubset({0, 1}):
            onehot_cols.append(c)
    continuous_cols = [c for c in continuous_cols
                       if c not in onehot_cols]

    # PowerTransformer
    # print('fit', fit)
    if fit:
        pt.fit(dfc[continuous_cols])
        dfc.loc[:, continuous_cols] = pt.transform(dfc[continuous_cols])


    Y = dfc[target]
    cols = [c for c in cols
            if c not in target
            and c not in redudent
            and c != "env_group"]
    X = dfc[cols]

    return X, Y, pt, climate_group

class Dataset_Load(Dataset):
    def __init__(self, root_path, data_path='N2OALL.csv',
                 target='lnRR', redudent='Family1'):
        # size [seq_len, label_len, pred_len]
        # info
        self.target = target
        self.redudent = redudent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # print(os.getcwd())    #G:\Project_Code\Adaboost_Regression\models
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), encoding='ISO-8859-1')
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # Get X, Y
        pt = PowerTransformer(
            method="yeo-johnson",
            standardize=True)
        self.X, self.Y, self.pt, self.climate_group = preprocess(df_raw,
            target=self.target,
            redudent=self.redudent,
            fit=False,
            pt=pt)
        
    seq_x = self.X
        seq_y = self.Y
        pt = self.pt
        climate_group = self.climate_group

        return seq_x, seq_y, pt, climate_group

    def __len__(self):
        return len(self.X)


class Dataset_Pred(Dataset):
    def __init__(self, cru_datafilename, gpcc_filename, fertilizer_filename, clay_filename,
                 bd_filename, soc_filename, ph_filename, landfilename, silt_filename, sand_filename):

        self.cru_file = cru_datafilename
        self.gpcc_file = gpcc_filename
        self.clay_file = clay_filename
        self.silt_file = silt_filename
        self.sand_file = sand_filename
        self.bd_file = bd_filename
        self.soc_file = soc_filename
        self.ph_file = ph_filename
        self.fertilizer_filename = fertilizer_filename
        self.landfile = landfilename

        self.__read_data__()

    def __read_data__(self):
        # cru_datafilename = '.\global_data\MAT\cru_ts4.06.2021.2021.tmp.dat.nc'
        self.soc_avg, self.landcover_all = read_global(self.cru_file, self.gpcc_file, self.fertilizer_filename, self.clay_file, self.bd_file,
                              self.soc_file, self.ph_file, self.landfile, self.silt_file, self.sand_file)
        lon = self.soc_avg[0, :]
        lat = self.soc_avg[1, :]
        lon_block = ((lon + 180)//30).astype(int)
        lat_block = ((lat + 90)//30).astype(int)
        self.climate_group = lon_block * 100 + lat_block
        return self.soc_avg, self.landcover_all, self.climate_group

    def __getitem__(self, index):
            return self.soc_avg, self.landcover_all, self.climate_group

    def __len__(self):
        return len(self.soc_avg)
