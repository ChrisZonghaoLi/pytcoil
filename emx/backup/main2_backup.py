import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

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
from utils.tcoil_mlp import mlp
from tcoil_inn import inn
from posteriors import Posteriors

import yaml
######################## Data import and pre-processing #######################



#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_5882_2021-05-16.csv')
#tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_eq_ckt_5882_2021-05-16.csv')
tcoil_data = pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_results_1.0GHz_8430_2021-05-17.csv')


# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)
 
tcoil_x_train = tcoil_train[['L','W','Nin','Nout']].copy()
#tcoil_y_train = tcoil_train[['Ls1','Rs1']].copy()
tcoil_y_train = tcoil_train[['La','Qa','Lb','Qb','k','Cbr']].copy()
#tcoil_y_train_inn = tcoil_train[['La','Lb','k']].copy()

tcoil_x_test = tcoil_test[['L','W','Nin','Nout']].copy()
#tcoil_y_test = tcoil_test[['Ls1','Rs1']].copy()
tcoil_y_test = tcoil_test[['La','Qa','Lb','Qb','k','Cbr']].copy()
#tcoil_y_test_inn = tcoil_test[['La','Lb','k']].copy()


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

tcoil_model = mlp(tcoil_x_train, tcoil_y_train)

################# test the forward inference tcoil mlp model ##################
tcoil_predictions = tcoil_model.predict(tcoil_x_test) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_


tcoil_labels_test_df = pd.DataFrame(data=tcoil_y_test, columns = ['La', 'Qa', 'Lb', 'Qb', 'k', 'Cbr'])
tcoil_predictions_df = pd.DataFrame(data=tcoil_predictions, columns = ['La', 'Qa', 'Lb', 'Qb', 'k', 'Cbr'])


plt.figure('La Eq. Ckt vs. La ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['La'], tcoil_predictions_df['La'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['La']),np.max(tcoil_labels_test_df['La'])],
            [np.min(tcoil_labels_test_df['La']),np.max(tcoil_labels_test_df['La'])], 'r--')
axs[0].set(ylabel='La = imag($Z_{11}$/$\omega$)\nML Prediction (H)')
axs[1].plot(tcoil_labels_test_df['La'],
            np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/np.array(tcoil_labels_test_df['La']) * 100,
            '^')
axs[1].set(xlabel='La = imag($Z_{11}$/$\omega$) Eq. Ckt (H)',
           ylabel='Deviation (%)')                            
fig.suptitle('True La vs. Predicted La')
axs[0].grid()
axs[1].grid()

plt.figure('Lb Eq. Ckt vs. Lb ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Lb'], tcoil_predictions_df['Lb'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Lb']),np.max(tcoil_labels_test_df['Lb'])],
            [np.min(tcoil_labels_test_df['Lb']),np.max(tcoil_labels_test_df['Lb'])], 'r--')
axs[0].set(ylabel='Lb = imag($Z_{22}$/$\omega$)\nML Prediction (H)')
axs[1].plot(tcoil_labels_test_df['Lb'],
            np.abs(tcoil_labels_test_df['Lb'] - tcoil_predictions_df['Lb'])/np.array(tcoil_labels_test_df['Lb']) * 100,
            '^')
axs[1].set(xlabel='Lb = imag($Z_{22}$/$\omega$) Eq. Ckt (H)',
           ylabel='Deviation (%)')                            
fig.suptitle('True Lb vs. Predicted Lb')
axs[0].grid()
axs[1].grid()

plt.figure('Qa Eq. Ckt vs. Qa ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Qa'], tcoil_predictions_df['Qa'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])],
            [np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])], 'r--')
axs[0].set(ylabel='Qa')
axs[1].plot(tcoil_labels_test_df['Qa'],
            np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa'])/np.array(tcoil_labels_test_df['Qa']) * 100,
            '^')
axs[1].set(xlabel='Qa$)',
           ylabel='Deviation (%)')                            
fig.suptitle('True Qa vs. Predicted Qa')
axs[0].grid()
axs[1].grid()

plt.figure('Qb Eq. Ckt vs. Qb ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Qb'], tcoil_predictions_df['Qb'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])],
            [np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])], 'r--')
axs[0].set(ylabel='Qb')
axs[1].plot(tcoil_labels_test_df['Qb'],
            np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb'])/np.array(tcoil_labels_test_df['Qb']) * 100,
            '^')
axs[1].set(xlabel='Qb',
           ylabel='Deviation (%)')                            
fig.suptitle('True Qb vs. Predicted Qb')
axs[0].grid()
axs[1].grid()

plt.figure('k Eq. Ckt vs. k ML Predicted')                                               
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['k'], tcoil_predictions_df['k'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])],
            [np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])], 'r--')
axs[0].set(ylabel='k ML Prediction')
axs[1].plot(tcoil_labels_test_df['k'],
            np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k'])/np.abs(tcoil_labels_test_df['k']) * 100,
            '^')
axs[1].set(xlabel='k Eq. Ckt',
           ylabel='Deviation (%)')                            
fig.suptitle('True k vs. Predicted k')
axs[0].grid()
axs[1].grid()

plt.figure('Cbr Eq. Ckt vs. Cbr ML Predicted')                                               
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Cbr'], tcoil_predictions_df['Cbr'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Cbr']),np.max(tcoil_labels_test_df['Cbr'])],
            [np.min(tcoil_labels_test_df['Cbr']),np.max(tcoil_labels_test_df['Cbr'])], 'r--')
axs[0].set(ylabel='Cbr ML Prediction')
axs[1].plot(tcoil_labels_test_df['Cbr'],
            np.abs(tcoil_labels_test_df['Cbr'] - tcoil_predictions_df['Cbr'])/np.abs(tcoil_labels_test_df['Cbr']) * 100,
            '^')
axs[1].set(xlabel='Cbr Eq. Ckt',
           ylabel='Deviation (%)')                            
fig.suptitle('True Cbr vs. Predicted Cbr')
axs[0].grid()
axs[1].grid()

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
La_target = 150e-10
Lb_target = 100e-10
k_target = -0.5

from pymoo.model.problem import Problem

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, 
                         n_obj=2, 
                         n_constr=3, 
                         xl=np.array([32, 2.4, 4, 5]), 
                         xu=np.array([80, 5, 12, 25]), 
                         elementwise_evaluation=True)
        
    def _evaluate(self, X, out, *args, **kwargs):
        def f1(x):
            # maximize the Q factor Qa + Qb
            x_copy = np.copy(x)
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
            print(f'[La, Qa, Lb, Qb, k, Cbr]: {predictions}')
            Q_tot = predictions[0][1] + predictions[0][3]
            return Q_tot
        
        def f2(x):
            # minimize the area of the tcoil
            x_copy = np.copy(x)
            L, W, Nin, Nout = x_copy
            area = L * L # This is not the net area of metal but just the tcoil geometry area  
            return area
        
        def g1(x):
            # constraint on the deviation of La should be less than 10%  
            x_copy = np.copy(x)
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
        
            La = predictions[0][0]
          
            loss = np.abs(La/1e-10 - La_target/1e-10) - 0.1*La_target/1e-10 # <=0
        
            return loss
        
        def g2(x):
            # constraint on the deviation of Lb should be less than 10%  
            x_copy = np.copy(x)
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
        
            Lb = predictions[0][2]
        
            loss = np.abs(Lb/1e-10 - Lb_target/1e-10) - 0.1*Lb_target/1e-10 # <=0
            # print(La, loss)
            return loss
        
        def g3(x):
            # constraint on the deviation of k should be less than 10%  
            x_copy = np.copy(x)  
            x_copy = scaler_in.transform(x_copy.reshape(1,-1))
            predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
        
            k = predictions[0][4]
                
            loss = np.abs(k - k_target) - 0.1*np.abs(k_target) # <=0
            # print(La, loss)
            return loss

        out['F'] = np.column_stack([f1(X), f2(X)])
        print(f"Qa+Qb: {out['F'][0][0]} | t-coil area: {out['F'][0][1]} um^2")
        out['G'] = np.column_stack([g1(X), g2(X), g3(X)])
        print(X)


from pymoo.model.sampling import Sampling
import random

class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 4), None, dtype=np.float32)
        
        for i in range(n_samples):
            
            # W
            W_list = [2.4, 4.2, 5]
            W = random.choice(W_list)
            
            # # S
            # if W == 2.4:
            #     S = 1.2
            # else:
            #     S = 1.44    
            
            # L
            if W == 2.4 or W == 4.2:
                L_list = [i for i in range(32, 81)]
                L = random.choice(L_list)
            else:
                L_list = [i for i in range(45, 81)]
                L = random.choice(L_list)
            
            # Nout    
            Nout_list = [i for i in range(4, 13)]
            Nout = random.choice(Nout_list)
            
            # Nin
            if W == 2.4:
                if L>=32 and L<=35:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=36 and L<=37:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 25)]
                else:
                    Nin_list = [i for i in range(5, 25)]
                    
            elif W == 4.2:
                if L>=32 and L<=33:
                    Nin_list = [5]
                elif L>=34 and L<=37:
                    Nin_list = [5, 6]
                elif L>=38 and L<=40:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 11)]
                elif L>=41 and L<=45:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 12)]
                elif L>=46 and L<=49:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=50 and L<=51:
                    Nin_list = [i for i in range(5, 15)] 
                elif L>=52 and L<=56:
                    Nin_list = [i for i in range(5, 16)]
                else:
                    Nin_list = [i for i in range(5, 25)]
                    
            else:
                if L>=45 and L<=48:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 12)]
                elif L>=49 and L<=54:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=55 and L<=57:
                    Nin_list = [i for i in range(5, 15)] 
                elif L>=58 and L<=64:
                    Nin_list = [i for i in range(5, 16)]
                else:
                    Nin_list = [i for i in range(5, 25)]
                    
            Nin = random.choice(Nin_list)  
            
            X[i][0] = L
            X[i][1] = W
            X[i][2] = Nin
            X[i][3] = Nout
        
        print(X)
        print(len(X))
        return X


from pymoo.model.mutation import Mutation

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()
        
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            r = np.random.random() # randomly generate a probability
            
            if r < 0.8: # 80% - change a variable randomly
                var = random.choice(['L', 'W', 'Nin', 'Nout'])
                L = X[i][0]
                W = X[i][1]
                Nin = X[i][2]
                Nout = X[i][3]
                
                if var == 'L':
                    if W == 2.4: # if W = 2.4
                        if Nin == 8 : # Nin 
                            L = random.choice([j for j in range(38, 81)])
                        elif Nin <= 14: 
                            L = random.choice([j for j in range(32, 81)])
                        else: # Nin >= 15:
                            L = random.choice([j for j in range(36, 81)])
      
                    elif W == 4.2:
                        if Nin == 5: # Nin 
                            L = random.choice([j for j in range(32, 81)])
                        elif Nin == 6: 
                            L = random.choice([j for j in range(34, 81)])
                        elif Nin == 7: 
                            L = random.choice([j for j in range(38, 81)])
                        elif Nin == 8: 
                            L = random.choice([j for j in range(50, 81)])
                        elif Nin == 9 or Nin == 10: 
                            L = random.choice([j for j in range(38, 50)])
                        elif Nin == 11: 
                            L = random.choice([j for j in range(41, 81)])
                        elif Nin >= 12 and Nin <= 14: 
                            L = random.choice([j for j in range(46, 81)])
                        elif Nin == 15: 
                            L = random.choice([j for j in range(52, 81)])
                        else: # Nin >= 16: 
                            L = random.choice([j for j in range(57, 81)])
                        
                    else: # W == 5:
                        if Nin == 8 : # Nin 
                            L = random.choice([j for j in range(55, 81)])
                        elif Nin <= 11: 
                            L = random.choice([j for j in range(45, 81)])
                        elif Nin >= 12 and Nin <= 14:
                            L = random.choice([j for j in range(49, 81)])
                        elif Nin == 15: 
                            L = random.choice([j for j in range(58, 81)])
                        else: # Nin >= 16: 
                            L = random.choice([j for j in range(65, 81)])
                   
                    X[i][0] = L      
                   
                if var == 'W':
                    if L <= 33: # if L <= 33um
                        if Nin == 5: # if N = 5
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L >= 34 and L <= 35:
                        if Nin == 5 or Nin == 6: # if N = 5 or 6
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L >= 36 and L <= 37:
                        if Nin == 5 or Nin == 6: # if N = 5 or 6
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L >= 38 and L <= 40:
                        if Nin == 8: 
                            W = 2.4
                        elif Nin <= 10:
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L >= 41 and L <= 44:
                        if Nin == 8: 
                            W = 2.4
                        elif Nin <= 11:
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L == 45:
                        if Nin == 8:
                            W = 2.4
                        elif Nin <= 11:
                            W = random.choice([2.4, 4.2, 5.0])
                        else:
                            W = 2.4
                            
                    elif L >= 46 and L <= 48:
                        if Nin == 8:
                            W = 2.4
                        elif Nin <= 11:
                            W = random.choice([2.4, 4.2, 5.0])
                        elif Nin >= 12 and Nin <= 14:
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                        
                    elif L == 49:
                        if Nin == 8:
                            W = 2.4
                        elif Nin <= 14:
                            W = random.choice([2.4, 4.2, 5.0])
                        else:
                            W = 2.4
                            
                    elif L >= 50 and L <= 51:
                        if Nin == 8:
                            W = random.choice([2.4, 4.2])
                        elif Nin <= 14:
                            W = random.choice([2.4, 4.2, 5.0])
                        else:
                            W = 2.4
                            
                    elif L >= 52 and L <= 54:
                        if Nin == 8:
                            W = random.choice([2.4, 4.2])
                        elif Nin <= 14:
                            W = random.choice([2.4, 4.2, 5.0])
                        elif Nin == 15:
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L >= 55 and L <= 56:
                        if Nin <= 14:
                            W = random.choice([2.4, 4.2, 5.0])
                        elif Nin == 15:
                            W = random.choice([2.4, 4.2])
                        else:
                            W = 2.4
                            
                    elif L == 57:
                        if Nin <= 14:
                            W = random.choice([2.4, 4.2, 5.0])
                        else:
                            W = random.choice([2.4, 4.2])
                       
                    elif L >= 58 and L <= 64:
                        if Nin <= 15:
                            W = random.choice([2.4, 4.2, 5.0])
                        else:
                            W = random.choice([2.4, 4.2])
                    
                    else:
                        W = random.choice([2.4, 4.2, 5.0])
                
                    X[i][1] = W
                    
                if var == 'Nin':       
                    if W == 2.4:
                        if L>=32 and L<=35:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                        elif L>=36 and L<=37:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 25)]
                        else:
                            Nin_list = [i for i in range(5, 25)]
                            
                    elif W == 4.2:
                        if L>=32 and L<=33:
                            Nin_list = [5]
                        elif L>=34 and L<=37:
                            Nin_list = [5, 6]
                        elif L>=38 and L<=40:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 11)]
                        elif L>=41 and L<=45:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 12)]
                        elif L>=46 and L<=49:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                        elif L>=50 and L<=51:
                            Nin_list = [i for i in range(5, 15)] 
                        elif L>=52 and L<=56:
                            Nin_list = [i for i in range(5, 16)]
                        else:
                            Nin_list = [i for i in range(5, 25)]
                            
                    else:
                        if L>=45 and L<=48:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 12)]
                        elif L>=49 and L<=54:
                            Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                        elif L>=55 and L<=57:
                            Nin_list = [i for i in range(5, 15)] 
                        elif L>=58 and L<=64:
                            Nin_list = [i for i in range(5, 16)]
                        else:
                            Nin_list = [i for i in range(5, 25)]
                                       
                    X[i][2] = random.choice(Nin_list)
                    
                if var == 'Nout':       
                    Nout_list = [i for i in range(4, 13)]
                    Nout = random.choice(Nout_list)
                    X[i][3] = Nout
                    
        return X


# number of threads to be used
n_process = 8
n_threads = 8

# initialize the pool
# pool = multiprocessing.Pool(n_process)
pool = ThreadPool(n_threads)

algorithm = NSGA2(
            pop_size=100,
            sampling=MySampling(),
            crossover=get_crossover('real_sbx', prob=0.9, eta=3),
            mutation=get_mutation('real_pm', eta=3),
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
# best set: L = 36.15, W = 2, S = 1.3, N = 3.75, tap = 6 -> Rtot = 3.46, La = 1.8, Lb = 1.36, k = 0.373

from pymoo.visualization.scatter import Scatter
plot = Scatter(title = 'Objective Space')
plot.add(res.F)
plot.show()

n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt_history = np.array([e.opt[0].F for e in res.history])
opt_history = pd.DataFrame(opt_history, columns=['res', 'area'])
input_history = np.array([e.opt[0].X for e in res.history]) * np.array([1,1,1,0.25,1])
input_history = pd.DataFrame(input_history, columns=['L', 'W', 'S', 'N', 'tap'])
output_history = tcoil_model.predict(scaler_in.transform(input_history[input_history.columns])) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
output_history = pd.DataFrame(output_history, columns=['La', 'Ra', 'Lb', 'Rb', 'k'])

###############################################################################
plt.figure('Convergence of Objetives')
fig, ax1 = plt.subplots()
fig.suptitle('Convergence of Objetives')
ax2 = ax1.twinx()
ax1.plot(n_evals, opt_history['res'], 'b-', label='Resistance')
ax2.plot(n_evals, opt_history['area'], 'r-', label='Area')
ax1.set_xlabel('n_evals')
ax1.set_ylabel('Resistance Ra+Rb ($\Omega$)', color='b')
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
ax2.plot(n_evals, output_history['Ra'], 'r-', label='Ra')
ax2.plot(n_evals, output_history['Rb'], 'r--', label='Rb')
ax3.plot(n_evals, output_history['k'], 'g-', label='Ra')
ax1.set_xlabel('n_evals')
ax1.set_ylabel('Inductance (H)', color='b')
ax2.set_ylabel('Resistance ($\Omega$)', color='r')
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
ax2.plot(n_evals, input_history['S'], 'r--', label='S')
ax3.plot(n_evals, input_history['N'], 'g-', label='N')
ax3.plot(n_evals, input_history['tap'], 'g--', label='tap')
ax1.set_xlabel('n_evals')
ax1.set_ylabel('Outer diameter (um)', color='b')
ax2.set_ylabel('Metal width and space (um)', color='r')
ax3.set_ylabel('Turns and tap location', color='g')
# move the third y axis to the right
ax3.spines['right'].set_position(('outward',60))
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()
