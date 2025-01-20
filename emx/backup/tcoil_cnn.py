import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras import backend as K

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
sys.path.append(os.path.abspath("/fs1/eecg/tcc/lizongh2/TCoil_ML/invertible_neural_networks"))
from tcoil_inn import inn
from posteriors import Posteriors

import yaml
######################## Data import and pre-processing #######################


stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
freq_design = float(sim_setups['freq_design'])

tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_{freq_design/1e9}GHz.csv')

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)
 
tcoil_x_train = tcoil_train[['L','W','Nin','Nout']].copy()
tcoil_y_train = tcoil_train[['La','Qa','Lb','Qb','k','Cbr']].copy()
tcoil_y_train_inn = tcoil_train[['La','Lb','k']].copy()

tcoil_x_test = tcoil_test[['L','W','Nin','Nout']].copy()
tcoil_y_test = tcoil_test[['La','Qa','Lb','Qb','k','Cbr']].copy()
tcoil_y_test_inn = tcoil_test[['La','Lb','k']].copy()


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
scaler_out_inn = StandardScaler()
mean_std_train_y_inn = scaler_out_inn.fit(tcoil_y_train_inn[tcoil_y_train_inn.columns])
tcoil_y_train_inn[tcoil_y_train_inn.columns]=scaler_out_inn.transform(tcoil_y_train_inn[tcoil_y_train_inn.columns])
tcoil_y_test_inn[tcoil_y_test_inn.columns]=scaler_out_inn.transform(tcoil_y_test_inn[tcoil_y_test_inn.columns])

# convert DataFrame to np array so they can be fed to TF model
tcoil_x_train = np.array(tcoil_x_train)
tcoil_x_test = np.array(tcoil_x_test)
tcoil_y_train = np.array(tcoil_y_train)
tcoil_y_test = np.array(tcoil_y_test)
tcoil_y_train_inn = np.array(tcoil_y_train_inn)
tcoil_y_test_inn = np.array(tcoil_y_test_inn)


tcoil_x_train = tcoil_x_train.astype('float32')
tcoil_x_test = tcoil_x_test.astype('float32')

input_shape = (np.shape(tcoil_x_train)[0], 1)
tcoil_x_train = tcoil_x_train.reshape(np.shape(tcoil_x_train)[0], np.shape(tcoil_x_train)[1], 1)

tcoil_model = tf.keras.Sequential([
    #normalize,
    layers.Conv1D(64, 3, activation='elu', padding='causal', input_shape = input_shape),
    layers.Conv1D(64, 3, activation='elu', padding='causal', input_shape = input_shape),

    #layers.MaxPooling1D(1),
    layers.Flatten(),
    layers.Dense(512, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dropout(0.1),
    layers.Dense(256, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(128, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(np.shape(tcoil_y_train)[1])
    ])


tcoil_model.compile(loss = tf.keras.losses.MeanAbsoluteError(),
                        #loss_weights=[0.7, 0.3],
                        optimizer = tf.keras.optimizers.Adam(),
                        metrics= ['mean_absolute_error'])
tcoil_model.summary()

tcoil_model.fit(tcoil_x_train, tcoil_y_train, validation_split = 0.2, epochs = 100)







