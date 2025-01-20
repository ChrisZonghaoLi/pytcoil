import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from datetime import datetime

import tensorflow as tf

# import pymoo
from pymoo.model.problem import FunctionalProblem
from pymoo.model.problem import ConstraintsAsPenaltyProblem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_termination, get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

# import pool for multiprocessing
#import multiprocessing
from multiprocessing.pool import ThreadPool

# import INN for narrowing down the searching space for GA
import os
import sys
TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
sys.path.append(PYTCOIL_DIR)
sys.path.append(os.path.abspath("/fs1/eecg/tcc/lizongh2/TCoil_ML/invertible_neural_networks"))
from utils.tcoil_mlp import mlp_asitic, mlp_poly, mlp_emx
from tcoil_inn import inn
from posteriors import Posteriors

import yaml
import ast

import matplotlib.pyplot as plt
plt.style.use(style='default')
plt.rcParams['font.family']='calibri'
######################## Data import and pre-processing #######################

# load simulation setting
stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
freq_start = float(sim_setups['freq_start'])
freq_step = float(sim_setups['freq_step'])
freq_design = float(sim_setups['freq_design'])
ind_max = float(sim_setups['ind_max'])


#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/data_old/tcoil_results_1.0GHz_12057_5e-10_2021-05-26.csv')[-5000:]
#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/data_old/tcoil_results_15.0GHz_5e-10_1114_2021-06-03.csv')[-800:]
#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_0.1-100GHz_2000_2021-06-06.csv')
tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_0.1-100.0GHz_4000_2021-06-23.csv')


# drop designs whose ind larger than ind_max or negative value (srf)
mask = []
slicer = int((freq_design-freq_start)/freq_step+1)
for i in range(len(tcoil_data)):
    mask.append((ast.literal_eval(tcoil_data['La'][i])[slicer]>ind_max) | (ast.literal_eval(tcoil_data['Lb'][i])[slicer]>ind_max) | (ast.literal_eval(tcoil_data['La'][i])[slicer]<0) | (ast.literal_eval(tcoil_data['Lb'][i])[slicer]<0) )
tcoil_data = tcoil_data.drop(tcoil_data[mask].index) 
# remove duplications
tcoil_data = tcoil_data.drop_duplicates(subset=['L','W','S','Nin','Nout'])

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)


tcoil_x_train = tcoil_train[['L','W','Nin','Nout']].copy()
tcoil_y_train = tcoil_train[['La','Qa','Lb','Qb','k', 'fr']].copy()
tcoil_x_train = tcoil_x_train.reset_index()
tcoil_x_train = tcoil_x_train[['L','W','Nin','Nout']]
tcoil_y_train = tcoil_y_train.reset_index()
    
tcoil_y_train['La'] = [ast.literal_eval(tcoil_y_train['La'][i])[slicer] for i in range(len(tcoil_y_train))]
tcoil_y_train['Qa'] = [ast.literal_eval(tcoil_y_train['Qa'][i])[slicer] for i in range(len(tcoil_y_train))]
tcoil_y_train['Lb'] = [ast.literal_eval(tcoil_y_train['Lb'][i])[slicer] for i in range(len(tcoil_y_train))]
tcoil_y_train['Qb'] = [ast.literal_eval(tcoil_y_train['Qb'][i])[slicer] for i in range(len(tcoil_y_train))]
tcoil_y_train['k'] = [ast.literal_eval(tcoil_y_train['k'][i].replace('nan','None'))[slicer] for i in range(len(tcoil_y_train))]
tcoil_y_train = tcoil_y_train[['La','Qa','Lb','Qb','k','fr']]


tcoil_x_test = tcoil_test[['L','W','Nin','Nout']].copy()
tcoil_y_test = tcoil_test[['La','Qa','Lb','Qb','k', 'fr']].copy()
tcoil_x_test = tcoil_x_test.reset_index()
tcoil_x_test = tcoil_x_test[['L','W','Nin','Nout']]
tcoil_y_test = tcoil_y_test.reset_index()
    
tcoil_y_test['La'] = [ast.literal_eval(tcoil_y_test['La'][i])[slicer] for i in range(len(tcoil_y_test))]
tcoil_y_test['Qa'] = [ast.literal_eval(tcoil_y_test['Qa'][i])[slicer] for i in range(len(tcoil_y_test))]
tcoil_y_test['Lb'] = [ast.literal_eval(tcoil_y_test['Lb'][i])[slicer] for i in range(len(tcoil_y_test))]
tcoil_y_test['Qb'] = [ast.literal_eval(tcoil_y_test['Qb'][i])[slicer] for i in range(len(tcoil_y_test))]
tcoil_y_test['k'] = [ast.literal_eval(tcoil_y_test['k'][i].replace('nan','None'))[slicer] for i in range(len(tcoil_y_test))]
tcoil_y_test = tcoil_y_test[['La','Qa','Lb','Qb','k', 'fr']]

# # one-hot encoder
# from tensorflow.keras.utils import to_categorical 

# L = np.array([i for i in range(32, 81)])
# W = np.array([2.4, 4.2, 5])
# S = np.array([1.2, 1.44])
# Nin = np.array([i for i in range(5, 25)])
# Nout = np.array([i for i in range(4, 13)])
# L_encoded = to_categorical(L)
# W_encoded = to_categorical(W)
# S_encoded = to_categorical(S)
# Nin_encoded = to_categorical(Nin)
# Nout_encoded = to_categorical(Nout)
# # conver the tcoil_x_train/test to one hot encoding:

# def onehot_converter(x_input):
#     x_input_onehot = []
#     for i in range(len(x_input)):
#         # convert L
#         L_onehot = L_encoded[int(x_input[i][0])-L[0]]
#         # convert W
#         if x_input[i][1] == 2.4:
#             W_onehot = W_encoded[0]
#         elif x_input[i][1] == 4.2:
#             W_onehot = W_encoded[1]
#         else:
#             W_onehot = W_encoded[2]
#         # convert S
#         if x_input[i][2] == 1.2:
#             S_onehot = S_encoded[0]
#         else:
#             S_onehot = S_encoded[1]
#         # convert Nin
#         Nin_onehot = Nin_encoded[int(x_input[i][3])-Nin[0]]
#         # convert Nout
#         Nout_onehot = Nout_encoded[int(x_input[i][4])-Nout[0]]
        
#         input_onehot = np.concatenate((L_onehot, W_onehot, S_onehot, Nin_onehot, Nout_onehot))
#         x_input_onehot.append(input_onehot)
#     return x_input_onehot

# tcoil_x_train_onehot = np.array(onehot_converter(np.array(tcoil_x_train)))
# tcoil_x_test_onehot = np.array(onehot_converter(np.array(tcoil_x_test)))

# normalize the input data    
scaler_in = StandardScaler()
mean_std_train_x = scaler_in.fit(tcoil_x_train[tcoil_x_train.columns])
tcoil_x_train[tcoil_x_train.columns]=scaler_in.transform(tcoil_x_train[tcoil_x_train.columns])
tcoil_x_test[tcoil_x_test.columns]=scaler_in.transform(tcoil_x_test[tcoil_x_test.columns])

# normalize the output data    
scaler_out = StandardScaler()
mean_std_train_y = scaler_out.fit(tcoil_y_train[tcoil_y_train.columns])
tcoil_y_train[tcoil_y_train.columns]=scaler_out.transform(tcoil_y_train[tcoil_y_train.columns])

# normalize the output data for INN
# scaler_out_inn = StandardScaler()
# mean_std_train_y_inn = scaler_out_inn.fit(tcoil_y_train_inn[tcoil_y_train_inn.columns])
# tcoil_y_train_inn[tcoil_y_train_inn.columns]=scaler_out_inn.transform(tcoil_y_train_inn[tcoil_y_train_inn.columns])
# tcoil_y_test_inn[tcoil_y_test_inn.columns]=scaler_out_inn.transform(tcoil_y_test_inn[tcoil_y_test_inn.columns])

# convert DataFrame to np array so they can be fed to TF model
tcoil_x_train = np.array(tcoil_x_train) 
tcoil_x_test = np.array(tcoil_x_test) 
tcoil_y_train = np.array(tcoil_y_train)
tcoil_y_test = np.array(tcoil_y_test)
# tcoil_y_train_inn = np.array(tcoil_y_train_inn)
# tcoil_y_test_inn = np.array(tcoil_y_test_inn)

######################### return the trained mlp model ########################

tcoil_model = mlp_emx(tcoil_x_train, tcoil_y_train)

################# test the forward inference tcoil mlp model ##################
tcoil_predictions = tcoil_model.predict(tcoil_x_test) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_


tcoil_labels_test_df = pd.DataFrame(data=tcoil_y_test, columns = ['La', 'Qa', 'Lb', 'Qb', 'k', 'fr'])
tcoil_predictions_df = pd.DataFrame(data=tcoil_predictions, columns = ['La', 'Qa', 'Lb', 'Qb', 'k', 'fr'])

La_mae = np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La']))
Lb_mae = np.mean(np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb']))
Qa_mae = np.mean(np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa']))
Qb_mae = np.mean(np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb']))
k_mae = np.mean(np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k']))
fr_mae = np.mean(np.abs(tcoil_labels_test_df['fr'] - tcoil_predictions_df['fr']))

def above_avg(mae, df_test_column, df_pred_column):
    counter=0
    for i in np.array(np.abs(df_test_column - df_pred_column)):
        if i > mae:
            counter=counter+1
            
    return counter

La_above_mae = above_avg(La_mae, tcoil_labels_test_df['La'], tcoil_predictions_df['La'])
Lb_above_mae = above_avg(Lb_mae, tcoil_labels_test_df['Lb'], tcoil_predictions_df['Lb'])
Qa_above_mae = above_avg(Qa_mae, tcoil_labels_test_df['Qa'], tcoil_predictions_df['Qa'])
Qb_above_mae = above_avg(Qb_mae, tcoil_labels_test_df['Qb'], tcoil_predictions_df['Qb'])
k_above_mae = above_avg(k_mae, tcoil_labels_test_df['k'], tcoil_predictions_df['k'])
fr_above_mae = above_avg(fr_mae, tcoil_labels_test_df['fr'], tcoil_predictions_df['fr'])

# training data histogram
plt.hist((tcoil_y_train*np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_)[:,0],density=False,bins=50)
plt.ylabel('Count')
plt.xlabel('Inductance La')

plt.hist((tcoil_y_train*np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_)[:,2],density=False,bins=50)
plt.ylabel('Count')
plt.xlabel('Inductance Lb')

date = datetime.today().strftime('%Y-%m-%d')

plt.figure('ML Predicted La vs. EMX La')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted La vs. EMX La')
axs[0].plot(tcoil_labels_test_df['La']/1e-12, tcoil_predictions_df['La']/1e-12, '^')
axs[0].plot([np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)],
            [np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)], 'r--')
axs[0].set(ylabel='ML\n Prediction (pH)')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['La']/1e-12,
            np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12,
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)],
            [np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12),np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12)], 'r--')
axs[1].set(ylabel='Absolute\n Error (pH)')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,10])
axs[2].plot(tcoil_labels_test_df['La']/1e-12,
            np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12,
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)],
            [np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12),np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12)], 'r--')
axs[2].set(xlabel='EMX La (pH)',
           ylabel='Absolute\n Error (pH)')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/La_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')

plt.figure('ML Predicted Lb vs. EMX Lb')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted Lb vs. EMX Lb')
axs[0].plot(tcoil_labels_test_df['Lb']/1e-12, tcoil_predictions_df['Lb']/1e-12, '^')
axs[0].plot([np.min(tcoil_labels_test_df['Lb']/1e-12),np.max(tcoil_labels_test_df['Lb']/1e-12)],
            [np.min(tcoil_labels_test_df['Lb']/1e-12),np.max(tcoil_labels_test_df['Lb']/1e-12)], 'r--')
axs[0].set(ylabel='ML\n Prediction (pH)')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['Lb']/1e-12,
            np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/1e-12,
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['Lb']/1e-12),np.max(tcoil_labels_test_df['Lb']/1e-12)],
            [np.mean(np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/1e-12),np.mean(np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/1e-12)], 'r--')
axs[1].set(ylabel='Absolute\n Error (pH)')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,20])
axs[2].plot(tcoil_labels_test_df['Lb']/1e-12,
            np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/1e-12,
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['Lb']/1e-12),np.max(tcoil_labels_test_df['Lb']/1e-12)],
            [np.mean(np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/1e-12),np.mean(np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/1e-12)], 'r--')
axs[2].set(xlabel='EMX Lb (pH)',
           ylabel='Absolute\n Error (pH)')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/Lb_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')


plt.figure('ML Predicted Qa vs. EMX Qa')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted Qa vs. EMX Qa')
axs[0].plot(tcoil_labels_test_df['Qa'], tcoil_predictions_df['Qa'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])],
            [np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])], 'r--')
axs[0].set(ylabel='ML\n Predicted Qa')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['Qa'],
            np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa']),
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])],
            [np.mean(np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa'])),np.mean(np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa']))], 'r--')
axs[1].set(ylabel='Absolute\n Error')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,0.5])
axs[2].plot(tcoil_labels_test_df['Qa'],
            np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa']),
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])],
            [np.mean(np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa'])),np.mean(np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa']))], 'r--')
axs[2].set(xlabel='EMX Qa',
           ylabel='Absolute\n Error')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/Qa_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')


plt.figure('ML Predicted Qb vs. EMX Qb')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted Qb vs. EMX Qb')
axs[0].plot(tcoil_labels_test_df['Qb'], tcoil_predictions_df['Qb'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])],
            [np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])], 'r--')
axs[0].set(ylabel='ML\n Predicted Qb')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['Qb'],
            np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb']),
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])],
            [np.mean(np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb'])),np.mean(np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb']))], 'r--')
axs[1].set(ylabel='Absolute\n Error')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,1])
axs[2].plot(tcoil_labels_test_df['Qb'],
            np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb']),
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])],
            [np.mean(np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb'])),np.mean(np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb']))], 'r--')
axs[2].set(xlabel='EMX Qb',
           ylabel='Absolute\n Error')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/Qb_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')



plt.figure('ML Predicted k vs. EMX k')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted k vs. EMX k')
axs[0].plot(tcoil_labels_test_df['k'], tcoil_predictions_df['k'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])],
            [np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])], 'r--')
axs[0].set(ylabel='ML\n Predicted k')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['k'],
            np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k']),
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])],
            [np.mean(np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k'])),np.mean(np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k']))], 'r--')
axs[1].set(ylabel='Absolute\n Error')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,0.05])
axs[2].plot(tcoil_labels_test_df['k'],
            np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k']),
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])],
            [np.mean(np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k'])),np.mean(np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k']))], 'r--')
axs[2].set(xlabel='EMX k',
           ylabel='Absolute\n Error')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/k_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')

plt.figure('ML Predicted SRF vs. EMX SRF')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted SRF vs. EMX SRF')
axs[0].plot(tcoil_labels_test_df['fr']/1e9, tcoil_predictions_df['fr']/1e9, '^')
axs[0].plot([np.min(tcoil_labels_test_df['fr']/1e9),np.max(tcoil_labels_test_df['fr']/1e9)],
            [np.min(tcoil_labels_test_df['fr']/1e9),np.max(tcoil_labels_test_df['fr']/1e9)], 'r--')
axs[0].set(ylabel='ML\n Predicted SRF\n (GHz)')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['fr']/1e9,
            np.abs(tcoil_labels_test_df['fr']/1e9 - tcoil_predictions_df['fr']/1e9),
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['fr']/1e9),np.max(tcoil_labels_test_df['fr']/1e9)],
            [np.mean(np.abs(tcoil_labels_test_df['fr']/1e9 - tcoil_predictions_df['fr']/1e9)),np.mean(np.abs(tcoil_labels_test_df['fr']/1e9 - tcoil_predictions_df['fr']/1e9))], 'r--')
axs[1].set(ylabel='Absolute\n Error (GHz)')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,2])
axs[2].plot(tcoil_labels_test_df['fr']/1e9,
            np.abs(tcoil_labels_test_df['fr']/1e9 - tcoil_predictions_df['fr']/1e9),
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['fr']/1e9),np.max(tcoil_labels_test_df['fr']/1e9)],
            [np.mean(np.abs(tcoil_labels_test_df['fr']/1e9 - tcoil_predictions_df['fr']/1e9)),np.mean(np.abs(tcoil_labels_test_df['fr']/1e9 - tcoil_predictions_df['fr']/1e9))], 'r--')
axs[2].set(xlabel='EMX SRF (GHz)',
           ylabel='Absolute\n Error (GHz)')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/SRF_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')


############################# import INN  ###############################
'''
# run TF INN model

# run inn and return the trained model
tcoil_model_inn = inn(tcoil_x_train, tcoil_y_train_inn)

# Using the train model to show posterior of the backward inferences:
x_dim = np.shape(tcoil_x_train)[1]
y_dim = np.shape(tcoil_y_train_inn)[1]
z_dim = x_dim
tot_dim = y_dim + z_dim
pad_dim = tot_dim - x_dim

# running the INN to narrow down the searching space for GA
La_target, Lb_target, k_target = (2e-10, 1.5e-10, 0.35)

parser_posterior = Posteriors(x_dim=x_dim, 
                          y_dim=y_dim, 
                          z_dim=z_dim, 
                          pad_dim=pad_dim, 
                          model=tcoil_model_inn, 
                          x_test=tcoil_x_test, 
                          y_test=scaler_out_inn.transform([[La_target, Lb_target, k_target]]), 
                          #y_test=tcoil_y_test_inn[0],
                          mean_std_train_x=mean_std_train_x)    

parser_posterior.show_posteriors(n_plots=1, ground_truth=False)
x_confidence_interval = parser_posterior.confidence_interval(n_plots=1)
x_low = np.array([x_confidence_interval[0][0][0], 
                  x_confidence_interval[0][1][0],
                  x_confidence_interval[0][2][0],
                  int(x_confidence_interval[0][3][0])*4,
                  int(x_confidence_interval[0][4][0])])
x_high = np.array([x_confidence_interval[0][0][1], 
                  x_confidence_interval[0][1][1],
                  x_confidence_interval[0][2][1],
                  int(x_confidence_interval[0][3][1]+1)*4,
                  int(x_confidence_interval[0][4][1]+1)])
'''
############################# GA implementation ###############################
La_target = 2e-10
Lb_target = 1.5e-10
k_target = -0.5

from pymoo.model.problem import Problem

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, 
                         n_obj=2, 
                         n_constr=4, 
                         xl=np.array([32, 1, 5, 4]), # L, W, Nin, Nout 
                         xu=np.array([80, 3, 24, 12]), 
                         elementwise_evaluation=True)
        
    def _evaluate(self, X, out, *args, **kwargs):
        def f1(x):
            # maximize the Q factor Qa + Qb
            x_copy = np.copy(x)
            if x_copy[1] == 1:
                x_copy[1] = float(2.4)
            elif x_copy[1] == 2:
                x_copy[1] = float(4.2)
            elif x_copy[1] == 3:
                x_copy[1] = float(5)
            else: 
                None
            
            print(x_copy)
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
            print(f'[La, Qa, Lb, Qb, k, fr]: {predictions}')
            Q_tot = predictions[0][1] + predictions[0][3]
            return -Q_tot # minimize -Qtot = maximize Q_tot
        
        def f2(x):
            # minimize the area of the tcoil
            x_copy = np.copy(x)
            if x_copy[1] == 1:
                x_copy[1] = float(2.4)
            elif x_copy[1] == 2:
                x_copy[1] = float(4.2)
            elif x_copy[1] == 3:
                x_copy[1] = float(5)
            else: 
                None
                
            L, W, Nin, Nout = x_copy
            area = L * L # This is not the net area of metal but just the tcoil geometry area  
            return area
        
        def g1(x):
            # constraint on the deviation of La should be less than 10%  
            x_copy = np.copy(x)
            if x_copy[1] == 1:
                x_copy[1] = float(2.4)
            elif x_copy[1] == 2:
                x_copy[1] = float(4.2)
            elif x_copy[1] == 3:
                x_copy[1] = float(5)
            else: 
                None
                
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
        
            La = predictions[0][0]
          
            loss = np.abs(La/1e-10 - La_target/1e-10) - 0.1*La_target/1e-10 # <=0
        
            return loss
        
        def g2(x):
            # constraint on the deviation of Lb should be less than 10%  
            x_copy = np.copy(x)
            if x_copy[1] == 1:
                x_copy[1] = float(2.4)
            elif x_copy[1] == 2:
                x_copy[1] = float(4.2)
            elif x_copy[1] == 3:
                x_copy[1] = float(5)
            else: 
                None
                
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
        
            Lb = predictions[0][2]
        
            loss = np.abs(Lb/1e-10 - Lb_target/1e-10) - 0.1*Lb_target/1e-10 # <=0
            # print(La, loss)
            return loss
        
        def g3(x):
            # constraint on the deviation of k should be less than 10%  
            x_copy = np.copy(x)  
            if x_copy[1] == 1:
                x_copy[1] = float(2.4)
            elif x_copy[1] == 2:
                x_copy[1] = float(4.2)
            elif x_copy[1] == 3:
                x_copy[1] = float(5)
            else: 
                None
                
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
        
            k = predictions[0][4]
                
            loss = np.abs(k - k_target) - 0.1*np.abs(k_target) # <=0
            # print(La, loss)
            return loss

        def g4(x):
            # constraints on Nin
            x_copy = np.copy(x) 
            if x_copy[1] == 1:
                x_copy[1] = float(2.4)
            elif x_copy[1] == 2:
                x_copy[1] = float(4.2)
            elif x_copy[1] == 3:
                x_copy[1] = float(5)
            else: 
                None
                
            L = x_copy[0]
            W = x_copy[1]
            Nin = x_copy[2]
            
            if W == 2.4:
                if L>=32 and L<=35:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 14 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=36 and L<=37:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 24 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 25)]
                else:
                    loss = Nin - 24
                    #Nin_list = [i for i in range(5, 25)]
                    
            elif W == 4.2:
                if L>=32 and L<=33:
                    loss = Nin - 5 # <=0, here is taking when Nin = 5, the '=' is taken and the constraint is met
                    #Nin_list = [5]
                elif L>=34 and L<=37:
                    loss = Nin - 6
                    #Nin_list = [5, 6]
                elif L>=38 and L<=40:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 10 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 11)]
                elif L>=41 and L<=45:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 11 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 12)]
                elif L>=46 and L<=49:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 14 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=50 and L<=51:
                    loss = Nin - 14 # <=0
                    #Nin_list = [i for i in range(5, 15)] 
                elif L>=52 and L<=56:
                    loss = Nin - 15 # <=0
                    #Nin_list = [i for i in range(5, 16)]
                else:
                    loss = Nin - 24 # <=0
                    #Nin_list = [i for i in range(5, 25)]
                    
            else:
                if L>=45 and L<=48:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 11 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 12)]
                elif L>=49 and L<=54:
                    if Nin == 8:
                        loss = Nin - 7 # '7' is just a dummy number, since '8' is not an acceptable Nin number here,
                                       # therefore when Nin = 8 the loss will always larger than 0, therefore disobey the constraint
                    else:
                        loss = Nin - 14 # <=0
                    #Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=55 and L<=57:
                    loss = Nin - 14 # <=0
                    #Nin_list = [i for i in range(5, 15)] 
                elif L>=58 and L<=64:
                    loss = Nin - 15 # <=0
                    #Nin_list = [i for i in range(5, 16)]
                else:
                    loss = Nin - 24 # <=0
                    #Nin_list = [i for i in range(5, 25)]
             
            return loss
            
        out['F'] = np.column_stack([f1(X), f2(X)])
        out['G'] = np.column_stack([g1(X), g2(X), g3(X), g4(X)])


mask = ['int', 'int', 'int', 'int']

sampling = MixedVariableSampling(mask, {
    'real': get_sampling('real_random'),
    'int':get_sampling('int_random')
    })

crossover = MixedVariableCrossover(mask, {
    'real': get_crossover('real_sbx', prob=1.0, eta=3.0),
    'int':get_crossover('int_sbx', prob=1.0, eta=3.0)
    })

mutation = MixedVariableMutation(mask, {
    'real': get_mutation('real_pm', eta=3.0),
    'int': get_mutation('int_pm', eta=3.0)
    })

# number of threads to be used
n_process = 8
n_threads = 8

# initialize the pool
# pool = multiprocessing.Pool(n_process)
pool = ThreadPool(n_threads)

algorithm = NSGA2(
            pop_size=100,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True)

# from pymoo.util.display import MultiObjectiveDisplay
# class MyDisplay(MultiObjectiveDisplay):
#     def _do(self, problem, evaluator, algorithm):
#         super()._do(problem, evaluator, algorithm)
#         self.output.append('[L, W, S, N, tap]', algorithm.pop.get('X')[0] * np.array([1,1,1,0.25,1]))
#         self.output.append('[res, area (um^2)]', algorithm.pop.get('F')[0])
        

problem = MyProblem()
problem = ConstraintsAsPenaltyProblem(problem, penalty=1e6)
termination = get_termination("n_gen", 100)
    
# res = minimize(problem, algorithm, termination = termination, seed=1, 
#                save_history=True, verbose=True, display=MyDisplay())

res = minimize(problem, algorithm, termination = termination, seed=1, 
               save_history=True, verbose=True)


print('Threads:', res.exec_time)

pool.close()
pool.join()

opt = res.opt[0]
X, F, CV = opt.get("X","__F__","__CV__")

print("Best solution found: \nX = {}\nF = {} \nCV = {}\n".format(X, F, CV))
# best set: L = 52, W = 2.4 ('1' represents 5 um here), Nin = 7, Nout = 7

# plot Pareto frontier
plt.figure('Objective Space')                                               
plt.scatter(res.F[:,0], res.F[:,1])
plt.xlabel('-(Qa + Qb)')
plt.ylabel('Area ($um^{2}$)')
plt.title('Objective Space')
plt.grid()
plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/plot/gf22/Huawei_2021-06-18/Pareto_{freq_design/1e9}GHz_{date}.eps',format='eps', bbox_inches='tight')


n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt_history = np.array([e.opt[0].F for e in res.history]) * np.array([-1,1]) # flip the sign of Q
opt_history = pd.DataFrame(opt_history, columns=['Qa+Qb', 'area'])
input_history = np.array([e.opt[0].X for e in res.history]) 
for i in range(len(input_history)):
    if input_history[i][1] == 1:
        input_history[i][1] = 2.4
    elif input_history[i][1] == 2:
        input_history[i][1] = 4.2
    else:
        input_history[i][1] = 5
input_history = pd.DataFrame(input_history, columns=['L', 'W', 'Nin', 'Nout'])
output_history = tcoil_model.predict(scaler_in.transform(input_history[input_history.columns])) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
output_history = pd.DataFrame(output_history, columns=['La', 'Qa', 'Lb', 'Qb', 'k', 'fr'])

###############################################################################
plt.figure('Convergence of Objetives')
fig, ax1 = plt.subplots()
fig.suptitle('Convergence of Objetives')
ax2 = ax1.twinx()
ax1.plot(n_evals, opt_history['Qa+Qb'], 'b-', label='Qa+Qb')
ax2.plot(n_evals, opt_history['area'], 'r-', label='Area')
ax1.set_xlabel('n_evals')
ax1.set_ylabel('Q', color='b')
ax2.set_ylabel('Area ($\mu m^2$)', color='r')
ax1.set_yscale('log')
ax2.set_yscale('log')
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()

plt.figure('Convergence of Output Parameters')
fig, ax1 = plt.subplots()
fig.suptitle('Convergence of Output Parameters')
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot(n_evals, output_history['La'], 'b-', label='La')
ax1.plot(n_evals, output_history['Lb'], 'b--', label='Lb')
ax2.plot(n_evals, output_history['Qa'], 'r-', label='Qa')
ax2.plot(n_evals, output_history['Qb'], 'r--', label='Qb')
ax3.plot(n_evals, output_history['k'], 'g-', label='k')
ax1.set_xlabel('n_evals')
ax1.set_ylabel('Inductance (H)', color='b')
ax2.set_ylabel('Q', color='r')
ax3.set_ylabel('k', color='g')
# move the third y axis to the right
ax3.spines['right'].set_position(('outward',60))
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()

plt.figure('Convergence of Iutput Parameters')
fig, ax1 = plt.subplots()
fig.suptitle('Convergence of Iutput Parameters')
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot(n_evals, input_history['L'], 'b-', label='L')
ax2.plot(n_evals, input_history['W'], 'r-', label='W')
#ax2.plot(n_evals, input_history['S'], 'r--', label='S')
ax3.plot(n_evals, input_history['Nin'], 'g-', label='N')
ax3.plot(n_evals, input_history['Nout'], 'g--', label='tap')
ax1.set_xlabel('n_evals')
ax1.set_ylabel('Outer diameter L (um)', color='b')
ax2.set_ylabel('Metal width W (um)', color='r')
ax3.set_ylabel('Nin and Nout segments', color='g')
# move the third y axis to the right
ax3.spines['right'].set_position(('outward',60))
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()
