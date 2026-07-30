import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from exp.exp_basic import Exp_Basic
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from data.loadData_distribution import Dataset_Load, Dataset_Pred
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import time

import warnings
warnings.filterwarnings('ignore')


def _get_data(args, flag):
    if flag == 'pred':
        Data = Dataset_Pred(args.cru_file, args.gpcc_file, args.clay_file,
                            args.bd_file, args.soc_file, args.ph_file,
                            args.crop_file, args.land_file, args.silt_file, args.sand_file)
    else:
        Data = Dataset_Load(
            root_path=args.root_path,
            data_path=args.data_path,
            target=args.target,
            redudent=args.redudent)

    return Data

class Exp_Adaboost(Exp_Basic):
    def __init__(self, args):
        super(Exp_Adaboost, self).__init__(args)

        self.seed = self.args.seed
        self.material = args.material
        self.root_path = self.args.root_path
        self.data_path = self.args.data_path
        self.target = self.args.target
        self.redudent = self.args.redudent
        self._build_model(self.seed, self.material)

    def _build_model(self, random_state, material):
        mate_dict = {
            'SOCT': {'n_estimators': 10, 'random_state': random_state, 'max_depth': 20, 'max_features': 8, min_samples_leaf': 1},  
            'NNLT': {'n_estimators': 40, 'random_state': random_state, 'max_depth': 9, 'max_features': 10, min_samples_leaf': 3},    
            'CO2T': {'n_estimators': 40, 'random_state': random_state, 'max_depth': 5, 'max_features': 8, min_samples_leaf': 1},
            'N2OT': {'n_estimators': 30, 'random_state': random_state, 'max_depth': 20, 'max_features': 8, min_samples_leaf': 1},
        }
        args_info = mate_dict[material]
        estimators = args_info['n_estimators']
        random_state = args_info['random_state']
        depth = args_info['max_depth']
        max_features = self.args_info['max_features']
        min_samples_leaf = self.args_info['min_samples_leaf']

        rf = RandomForestRegressor(
            n_estimators=estimators, 
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=depth,
            bootstrap=True,
            random_state=random_state,
            max_samples=0.632,
            n_jobs=14)

        self.regression = MultiOutputRegressor(rf)

        return self.regression


    def train(self, setting):
        args = self.args
        
        #get data
        Data = _get_data(args=args, flag='train')[0]
        dx, dy = Data[0], Data[1]

        # Compute feature MESS
        Data_global = _get_data(args=args, flag='pred')[0][0]
        Lo = Data_global[:, 0]
        La = Data_global[:, 1]
        global_x = Data_global[:, 3:]
        feature_names = ['BD', 'MAT', 'MAP', 'pH', 'SOC', 'Clay', 'Sand', 'Silt']
        dx = dx.iloc[:, 3:]
                     
        inside_list = []
        for i, name in enumerate(feature_names):
            inside = ((global_x[:, i] > dx.min(axis=0)[i]) & (global_x[:, i] < dx.max(axis=0)[i]))
            inside_list.append((inside == True).astype(np.uint8))
        inside_array = np.column_stack(inside_list)
        inside_num = inside_array.sum(axis=1)

        inside5 = np.sum(inside_num >= 5) / len(inside_num) * 100
        inside6 = np.sum(inside_num >= 6) / len(inside_num) * 100
        inside7 = np.sum(inside_num >= 7) / len(inside_num) * 100
        inside8 = np.sum(inside_num >= 8) / len(inside_num) * 100

        # MAT-MAP Distribution
        MAT = dx["MAT"].values
        MAP = dx["MAP"].values
        MAT_global = Data_global[:, 4]
        MAP_global = Data_global[:, 5]
        plt.scatter(
             MAT_global,
             MAP_global,
             s=15,
             color="lightgray",
             alpha=0.3,
             label="Global climate space")
        plt.scatter(
             MAT,
             MAP,
             s=15,
             color="royalblue",
             alpha=0.8,
             label="Observations")    
        
        # spatial K-fold cross-validation
        R2_fold_list = []
        NRMSE_fold_list = []

        kf = KFold(
            n_splits=10,
            shuffle=True,
            random_state=args.seed)

        judge = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(dx, groups=climate_group)):
            train_fold_x = dx.iloc[train_idx]
            test_fold_x = dx.iloc[test_idx]

            train_fold_y = dy.iloc[train_idx]
            test_fold_y = dy.iloc[test_idx]
            grid = tune_rf(train_fold_x, train_fold_y)
            params = grid.best_params_.copy()
            rf = train_rf(train_fold_x, train_fold_y, params)
            rf.fit(train_fold_x, train_fold_y)
            pred_mean = rf.predict(test_fold_x)        

            R2_fold = r2_score(test_fold_y, pred_mean)
            R2_fold_list.append(R2_fold)

            rmse_score = np.sqrt(mean_squared_error(test_fold_y, pred_mean))
            nrmse_sd = rmse_score / np.std(test_fold_y)
            nrmse = rmse_score / (test_y.max() - test_y.min())
            NRMSE_fold_list.append(nrmse_sd)

            if judge >= R2_fold:
                pass
            else:
                best_model = rf
                judge = R2_fold

        print("========== Cross Validation ==========")
        rmse_score = np.mean(NRMSE_fold_list)
        rmse_std = np.std(NRMSE_fold_list)
        R2_score = np.mean(R2_fold_list)
        R2_std = np.std(R2_fold_list)
        print('KFold Cross-Validation R2:', R2_score, '+/-', R2_std, \
              'KFold Cross-Validation Nrmse:', rmse_score, '+/-', rmse_std)

        # Bootstrap                     
        train_x, test_x, train_y, test_y = train_test_split(dx, dy, test_size=test_dict[self.material], random_state=args.seed)

        grid = tune_rf(train_x, train_y)
        MAT_bin = pd.qcut(train_x["MAT"], 3)
        MAP_bin = pd.qcut(train_x["MAP"], 3)
        env_group = (
                MAT_bin.astype(str)
                + "_"
                + MAP_bin.astype(str))

        bootstraps_x = [resample(train_x,
                               stratify=env_group,
                               replace=True,
                               random_state=i) for i in range(5, 104)]
        bootstraps_x.append(train_x.copy())
        bootstraps_y = [train_y.loc[df.index].copy() for df in bootstraps_x]

        # Uncertainty
        models, preds = bootstrap_train(
            bootstraps_x,
            bootstraps_y,
            self.args_info,  #grid.best_params_,
            x_test=test_x,
            save_dir="./models")
        r2_scores = []
        for pred in preds:
            r2 = r2_score(
                    test_y,
                    pred,
                    multioutput='raw_values')
            r2_scores.append(r2)
        r2_scores = np.array(r2_scores)
        bootstraps_r2_mean = np.mean(r2_scores)
        bootstraps_r2_std = np.std(r2_scores)
        print("\n========== Bootstraps 100 ==========")
        print('Bootstraps lnRR R2: {0:.7f} ± {1:.3f}'.format(bootstraps_r2_mean, bootstraps_r2_std))
        # 95% CI
        ci = np.percentile(r2_scores, [25, 75])
        print('95% CI', ci)

        self.regression.xmin = dx.iloc[:, 3:].min()   
        self.regression.xmax = dx.iloc[:, 3:].max()

        return r2_scores, models, best_model


class Exp_Predict(Exp_Basic):
    def __init__(self, args):
        super(Exp_Predict, self).__init__(args)

        self.args = args
        # print('self.args', self.args)

    def predict(self, model):
        Predict = _get_data(self.args, flag='pred')[0]
        Predict_x, landcover = Predict[0], Predict[1]

        Ypredicted = model.predict(Predict_x)

        preds = np.array(Ypredicted)
        pred_lnRR_N2O = preds[:, 0]
        pred_N2O = preds[:, 1]
        Lo = Predict_x[:, 0]
        La = Predict_x[:, 1]

        return pred_lnRR_N2O, pred_N2O, Lo, La, landcover



