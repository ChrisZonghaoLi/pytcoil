import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

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
######################## Data import and pre-processing #######################



#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_5882_2021-05-16.csv')
#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_eq_ckt_5882_2021-05-16.csv')
#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_8430_2021-05-17.csv')
tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_12057_5e-10_2021-05-26.csv')
tcoil_data = tcoil_data.drop_duplicates(subset=['L','W','S','Nin','Nout'])

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)


tcoil_x_train = tcoil_train[['L','W','S','Nin','Nout']].copy()
#tcoil_y_train = tcoil_train[['Ls1','Rs1']].copy()
tcoil_y_train = tcoil_train[['La','Qa','Lb','Qb','k']].copy()
#tcoil_y_train_inn = tcoil_train[['La','Lb','k']].copy()

tcoil_x_test = tcoil_test[['L','W','S','Nin','Nout']].copy()
#tcoil_y_test = tcoil_test[['Ls1','Rs1']].copy()
tcoil_y_test = tcoil_test[['La','Qa','Lb','Qb','k']].copy()
#tcoil_y_test_inn = tcoil_test[['La','Lb','k']].copy()

scaler_in = StandardScaler()
mean_std_train_x = scaler_in.fit(tcoil_x_train[tcoil_x_train.columns])
tcoil_x_train[tcoil_x_train.columns]=scaler_in.transform(tcoil_x_train[tcoil_x_train.columns])
tcoil_x_test[tcoil_x_test.columns]=scaler_in.transform(tcoil_x_test[tcoil_x_test.columns])

scaler_out = StandardScaler()
mean_std_train_y = scaler_out.fit(tcoil_y_train[tcoil_y_train.columns])
tcoil_y_train[tcoil_y_train.columns]=scaler_out.transform(tcoil_y_train[tcoil_y_train.columns])

regr = RandomForestRegressor(max_depth=None, criterion='mae', random_state=None)
regr.fit(np.array(tcoil_x_train), np.array(tcoil_y_train))

tcoil_y_pred = regr.predict(np.array(tcoil_x_test)) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_

tcoil_labels_test_df = tcoil_y_test
tcoil_predictions_df = pd.DataFrame(data=tcoil_y_pred, columns = ['La', 'Qa', 'Lb', 'Qb', 'k'])

plt.figure('ML Predicted La vs. EMX La')
fig, axs = plt.subplots(3)
fig.suptitle('ML Predicted La vs. EMX La')
axs[0].plot(np.array(tcoil_labels_test_df['La'])/1e-12, np.array(tcoil_predictions_df['La'])/1e-12, '^')
axs[0].plot([np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)],
            [np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)], 'r--')
axs[0].set(ylabel='ML\n Prediction (pH)')
axs[0].set_xticklabels([])
axs[1].plot(tcoil_labels_test_df['La']/1e-12,
            np.abs(np.array(tcoil_labels_test_df['La']) - np.array(tcoil_predictions_df['La']))/1e-12,
            '^')
axs[1].plot([np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)],
            [np.mean(np.abs(np.array(tcoil_labels_test_df['La']) - np.array(tcoil_predictions_df['La'])/1e-12)),np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12)], 'r--')
axs[1].set(ylabel='Absolute\n Error (pH)')                            
axs[1].set_xticklabels([])
axs[2].set_ylim([0,60])
axs[2].plot(np.array(tcoil_labels_test_df['La'])/1e-12,
            np.abs(np.array(tcoil_labels_test_df['La']) - np.array(tcoil_predictions_df['La']))/1e-12,
            '^')
axs[2].plot([np.min(tcoil_labels_test_df['La']/1e-12),np.max(tcoil_labels_test_df['La']/1e-12)],
            [np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12),np.mean(np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/1e-12)], 'r--')
axs[2].set(xlabel='EMX La (pH)',
           ylabel='Absolute\n Error (pH)')   
# view eps file with 'gv'
axs[0].grid()
axs[1].grid()
axs[2].grid()




















