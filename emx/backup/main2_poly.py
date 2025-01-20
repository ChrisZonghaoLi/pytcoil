import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import errno
import subprocess
import shutil
import skrf as rf
import ast

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
EMX_WORK_DIR = os.environ['EMX_WORK_DIR']
sys.path.append(PYTCOIL_DIR)
sys.path.append(os.path.abspath("/fs1/eecg/tcc/lizongh2/TCoil_ML/invertible_neural_networks"))
from utils.tcoil_mlp import mlp_poly
from tcoil_inn import inn
from posteriors import Posteriors

import yaml
######################## Data import and pre-processing #######################

stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
freq_start = float(sim_setups['freq_start'])
freq_step = float(sim_setups['freq_step'])


#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_5882_2021-05-16.csv')
#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_8430_2021-05-17.csv')
#tcoil_data = pd.read_csv('~/TCoil_ML/data/gf22/train/tcoil_S_12671_2021-05-25.csv') # this only goes up to 500 pH
#tcoil_data = pd.read_csv('~/TCoil_ML/data/gf22/train/data_old/tcoil_S_0.1-100.0GHz_5e-10_2000_2021-05-31.csv')[-1500:] # this only goes up to 500 pH
tcoil_data = pd.read_csv('~/TCoil_ML/data/gf22/train/tcoil_0.1-100.0GHz_4000_2021-06-23.csv')
tcoil_data = tcoil_data.drop_duplicates(subset=['L','W','S','Nin','Nout'])

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)
 
tcoil_x_train = np.array(tcoil_train[['L','W','S','Nin','Nout']].copy())
# tcoil_x_train = np.concatenate((tcoil_x_train,
#                                 np.array(tcoil_train['L']/tcoil_train['W']).reshape(-1,1),
#                                 np.array(tcoil_train['L']/tcoil_train['S']).reshape(-1,1),
#                                 np.array(tcoil_train['L']/tcoil_train['Nin']).reshape(-1,1),
#                                 np.array(tcoil_train['L']/tcoil_train['Nout']).reshape(-1,1)), axis=1)
tcoil_y_train = tcoil_train[['s11', 's12', 's13', 's22', 's23', 's33']].copy()
#tcoil_y_train = tcoil_train[['s12']].copy()
tcoil_srf_train = np.array(tcoil_train[['fr']].copy())
#tcoil_y_train = np.angle(np.array(tcoil_y_train.applymap(ast.literal_eval).values.tolist()))
#tcoil_y_train = np.real(np.array(tcoil_y_train.applymap(ast.literal_eval).values.tolist()))
# tcoil_y_train = np.concatenate(
#                 (np.abs(np.array(tcoil_y_train.applymap(ast.literal_eval).values.tolist())),
#                   np.angle(np.array(tcoil_y_train.applymap(ast.literal_eval).values.tolist()))
#                   ), axis = 1
#                 )
tcoil_y_train = np.concatenate(
                (np.real(np.array(tcoil_y_train.applymap(ast.literal_eval).values.tolist())),
                  np.imag(np.array(tcoil_y_train.applymap(ast.literal_eval).values.tolist()))
                  ), axis = 1
                )
#tcoil_y_train = tcoil_y_train[:,:,:500]
                 
tcoil_x_test = np.array(tcoil_test[['L','W','S','Nin','Nout']].copy())
# tcoil_x_test = np.concatenate((tcoil_x_test,
#                                 np.array(tcoil_test['L']/tcoil_test['W']).reshape(-1,1),
#                                 np.array(tcoil_test['L']/tcoil_test['S']).reshape(-1,1),
#                                 np.array(tcoil_test['L']/tcoil_test['Nin']).reshape(-1,1),
#                                 np.array(tcoil_test['L']/tcoil_test['Nout']).reshape(-1,1)), axis=1)
#tcoil_y_test = tcoil_test[['s12']].copy()
tcoil_srf_test = np.array(tcoil_test[['fr']].copy())
tcoil_y_test = tcoil_test[['s11', 's12', 's13', 's22', 's23', 's33']].copy()
#tcoil_y_test = np.angle(np.array(tcoil_y_test.applymap(ast.literal_eval).values.tolist()))
#tcoil_y_test = np.real(np.array(tcoil_y_test.applymap(ast.literal_eval).values.tolist()))
# tcoil_y_test = np.concatenate(
#                 (np.abs(np.array(tcoil_y_test.applymap(ast.literal_eval).values.tolist())),
#                   np.angle(np.array(tcoil_y_test.applymap(ast.literal_eval).values.tolist()))
#                   ), axis = 1
#                 )
tcoil_y_test = np.concatenate(
                (np.real(np.array(tcoil_y_test.applymap(ast.literal_eval).values.tolist())),
                  np.imag(np.array(tcoil_y_test.applymap(ast.literal_eval).values.tolist()))
                  ), axis = 1
                )
#tcoil_y_test = tcoil_y_test[:,:,:500]

# normalize the input data    
mean_x_train = tcoil_x_train.mean(axis=0)
std_x_train = tcoil_x_train.std(axis=0)
tcoil_x_train = (tcoil_x_train-mean_x_train)/std_x_train
tcoil_x_test = (tcoil_x_test-mean_x_train)/std_x_train

order = 4 # sweet spot 4
poly_real_train_list = []
poly_real_test_list = []
for _ in range(len(tcoil_y_train)):
    srf_idx = int((tcoil_srf_train[_]-freq_start)/freq_step+1)
    if srf_idx >= np.shape(tcoil_y_train)[2]/2:
        srf_idx = int(np.shape(tcoil_y_train)[2])
    f = np.array([0.001*i for i in range(srf_idx)])
    poly_real_train = np.squeeze(np.array([np.polyfit(f,tcoil_y_train[_][0][:srf_idx],order)]))
    poly_real_train_list.append(poly_real_train)

for _ in range(len(tcoil_y_test)):
    srf_idx = int((tcoil_srf_test[_]-freq_start)/freq_step+1)
    f = np.array([0.001*i for i in range(srf_idx)])
    poly_real_test = np.squeeze(np.array([np.polyfit(f,tcoil_y_test[_][0][:srf_idx],order)]))
    poly_real_test_list.append(poly_real_test)
    
poly_real_mean = np.array(poly_real_train_list).mean(axis=0)
poly_real_std = np.array(poly_real_train_list).std(axis=0)

poly_real_train_list = (poly_real_train_list-poly_real_mean)/poly_real_std

# f = np.array([0.001*i for i in range(np.shape(tcoil_y_test)[2])]) 
# poly_real_train = np.array([np.polyfit(f,tcoil_y_train[i][0],order) for i in range(len(tcoil_y_train))])
# poly_real_test = np.array([np.polyfit(f,tcoil_y_test[i][0],order) for i in range(len(tcoil_y_test))])
# poly_real_mean = poly_real_train.mean(axis=0)
# poly_real_std = poly_real_train.std(axis=0)

# poly_real_train = (poly_real_train-poly_real_mean)/poly_real_std



from sklearn.ensemble import RandomForestRegressor


regr = RandomForestRegressor(n_estimators=128, max_depth=None, criterion='mae', random_state=None)
regr.fit(np.array(tcoil_x_train), np.array(poly_real_train_list))

poly_real_predictions = regr.predict(np.array(tcoil_x_test)) * poly_real_std + poly_real_mean


######################### return the trained mlp model ########################

tcoil_model = mlp_poly(tcoil_x_train, poly_real_train_list)
#tcoil_model = mlp_poly(tcoil_x_train, poly_real_train)


################# test the forward inference tcoil mlp model ##################
poly_real_predictions = tcoil_model.predict(tcoil_x_test) * poly_real_std + poly_real_mean

# We only examine the poly fitting up to SRF of each design
s12_real_pred_list = []
s12_real_test_list = []
for _ in range(len(tcoil_y_test)):
    srf_idx = int((tcoil_srf_test[_]-freq_start)/freq_step+1)
    f = np.array([0.001*i for i in range(srf_idx)])
    s12_real_pred = np.squeeze(np.array([np.polyval(poly_real_predictions[_],f)]))
    s12_real_pred_list.append(s12_real_pred)
    s12_real_test = np.squeeze(np.array(tcoil_y_test[_][0][:srf_idx]))
    #s12_real_test = np.squeeze(np.array([np.polyval(poly_real_test[_],f)]))
    s12_real_test_list.append(s12_real_test)

# plt.figure('s12_test vs. s12_pred')                                               
# plt.plot(np.abs(np.array(s12_real_pred_list)-np.array(s12_real_test_list)).mean(axis=0), 'r')
# plt.xlabel('Frequency (MHz)')
# plt.ylabel('k')
# plt.legend(('k_test', 'k_pred'))
# plt.title('k_test vs. k_pred')
# plt.grid()

import scipy.fftpack as ft
import math
case = 4
def s12_real(f):
    return poly_real_predictions[case][0]*f**5 + poly_real_predictions[case][1]*f**4 + \
            poly_real_predictions[case][2]*f**3 + poly_real_predictions[case][3]*f**0 + \
            poly_real_predictions[case][4]*f**0
    

for case in range(1000):
    srf_idx = int((tcoil_srf_test[case]-freq_start)/freq_step+1)
    f = np.array([0.001*i for i in range(srf_idx)])
    w = 2*math.pi*f
    plt.figure('s12 real')
    plt.plot(np.array(s12_real_test_list[case]),np.array(s12_real_pred_list[case]),'r')
    #plt.plot(f,np.array(s12_real_test_list[case]),'b')
    # f_prime = np.linspace(-max(f)*0.5,max(f)*1.2,2000)
    # plt.plot(f_prime,ft.hilbert(s12_real(f_prime)),'g')
    #plt.plot(f,np.array(tcoil_y_test[case][1][:srf_idx]),'y')