import argparse
import os
import torch
import numpy as np
import pandas as pd
import _pickle as cPickle

from exp.exp_multioutput import Exp_Adaboost, Exp_Predict
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='[Adaboost] Adaboost regression for fertilizer prediction')

parser.add_argument('--model', type=str, required=True, default='Adaboost', help='model name')
parser.add_argument('--root_path', type=str, required=True, default='./raw_data/',  help='root path of the data file')
parser.add_argument('--data_path', type=str, required=True, default='SOCALL_New.xlsx', help='data file')
parser.add_argument('--repeat', type=int, required=False, default=10, help='epochs of adaboost regressor')
parser.add_argument('--random_state', type=int, required=False, default=2000, help='epochs of adaboost regressor')
parser.add_argument('--krandom_state', type=int, required=False, default=100, help='epochs of adaboost regressor')
parser.add_argument('--seed', type=int, required=False, default=2026, help='random seed of adaboost regressor')  
parser.add_argument('--target', type=str, required=True, default=['SOCT', 'SOCC'], help='target prediction columns')
parser.add_argument('--do_predict', type=str, required=True, default=True, help='predict global fertilizer substance')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--output', type=str, default='./prediction/', help='prediction save path')

parser.add_argument('--redudent', type=str, default=['Planting', 'Family', 'LE', 'lnRR'],
                    help='input sequence length of Informer encoder')
parser.add_argument('--cru_file', type=str, default='./global_data/MAT/cru_ts4.06.2021.2021.tmp.dat.nc', help='cru data')
parser.add_argument('--gpcc_file', type=str, default='.\global_data\MAP\\normals_1991_2020_v2022_05.nc', help='gpcc data')
parser.add_argument('--clay_file', type=str, default='.\global_data\T_CLAY.nc4', help='clay data')
parser.add_argument('--silt_file', type=str, default='.\global_data\T_SILT.nc4', help='clay data')
parser.add_argument('--sand_file', type=str, default='.\global_data\T_SAND.nc4', help='clay data')
parser.add_argument('--bd_file', type=str, default='.\global_data\T_BULK_DEN.nc4', help='bd data')
parser.add_argument('--soc_file', type=str, default='.\global_data\T_OC.nc4', help='soc data')
parser.add_argument('--ph_file', type=str, default='.\global_data\T_PH_H2O.nc4', help='ph data')
parser.add_argument('--crop_file', type=str, default='G:\Project_Code\Adaboost_Regression\Cropcombination.xlsx', help='crop data')
parser.add_argument('--fertizer_file', type=str, default='.\global_data\\nmanure_app_crop.nc', help='fertilizer data')
parser.add_argument('--land_file', type=str, default='.\global_data\MCD12C1.A2021001.061.2022217040006.hdf', help='land data')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

# parser.add_argument('-')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')

target = args.target[-1]
setting = '{}_{}'.format(args.model, target)

Exp = Exp_Adaboost

checkpoint_path = os.path.join(args.checkpoints, setting)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
seed = args.seed
args.material = args.target[0]
exp = Exp(args)  # set experiments
r2, models, model = exp.train(setting)

# result save
folder_path = './results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

print('\n>>>>>>>Bootstrap predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
if args.do_predict:
    Exp_Pre = Exp_Predict(args)
    # pred, Lo, La, landcover = Exp_Pre.predict(model)
    N2OT, N2OC, Lo, La, landcover = Exp_Pre.predict(model)
    lnRR = np.log(N2OT / N2OC)
    Lo = Lo.reshape(360, 720)
    La = La.reshape(360, 720)
    lnRR = lnRR.reshape(360, 720)
    landcover = landcover.reshape(360, 720)
    lnRR[landcover == 0] = np.nan
    lnRR = pd.DataFrame(lnRR)
    writer1 = pd.ExcelWriter('./Global_Prediction/Tables/final/PredGlobal_{}_lnRR.xlsx'.format(target), engine='xlsxwriter')
    lnRR.to_excel(writer1, float_format='%.4f')
    writer1.close()

# Bootstrap prediction
    lnRR_Boots = []
    for md in models:
        N2OT, N2OC, Lo, La, landcover = Exp_Pre.predict(md)
        lnRR = np.log(N2OT / N2OC)
        lnRR_Boots.append(lnRR)
        lnRR = N2OC.reshape(360, 720)
        landcover = landcover.reshape(360, 720)
        lnRR[landcover == 0] = np.nan

    lnRR_Boots = np.array(lnRR_Boots).T
    std_lnRR = lnRR_Boots.std(axis=1)
    median_lnRR = np.median(lnRR_Boots, axis=1)
    
    # Coefficient of Variation
    cv_lnRR = (std_lnRR / (np.abs(median_lnRR) + 1e-10))
    Lo = Lo.reshape(360, 720)
    La = La.reshape(360, 720)
    landcover = landcover.reshape(360, 720)
    cv_lnRR_map = cv_lnRR.reshape(360, 720)
    cv_lnRR_map[landcover == 0] = np.nan
    lnRR_cv = pd.DataFrame(cv_lnRR_map)
    writer2 = pd.ExcelWriter('./Global_Prediction/Tables/final/Bootstrap_{}_cv.xlsx'.format(target), engine='xlsxwriter')
    lnRR_cv.to_excel(writer3, float_format='%.4f')
    writer2.close()

# result save
folder_path = './results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if args.do_predict:
    print('\n>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    Exp_Pre = Exp_Predict(args)

    N2OT, N2OC, Lo, La, landcover = Exp_Pre.predict(model)
    lnRR = np.log(N2OT / N2OC)
    lnRR[landcover.ravel() == 0] = np.nan

    lnRR = pd.DataFrame(lnRR)
    writer3 = pd.ExcelWriter(folder_path + str(args.target[1]) + '{}_lnRR.xlsx'.format(target), engine='xlsxwriter')
    lnRR.to_excel(writer3, float_format='%.4f')
    writer3.close()

print('File writen down')


