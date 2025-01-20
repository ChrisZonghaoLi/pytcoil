import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
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
sys.path.append(PYTCOIL_DIR)
sys.path.append(os.path.abspath("/fs1/eecg/tcc/lizongh2/TCoil_ML/invertible_neural_networks"))
from utils.tcoil_mlp import mlp
from tcoil_inn import inn
from posteriors import Posteriors

import yaml
######################## Data import and pre-processing #######################

tcoil_data = pd.read_csv('~/TCoil_ML/data/gf22/train/tcoil_S_12671_2021-05-21.csv')

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)
 
tcoil_x_train = np.array(tcoil_train[['L','W','S','Nin','Nout']].copy())
tcoil_S_train = tcoil_train[['s11', 's12', 's13', 's22', 's23', 's33']].copy()
tcoil_S_train = np.concatenate(
                (np.abs(np.array(tcoil_S_train.applymap(ast.literal_eval).values.tolist())),
                 np.angle(np.array(tcoil_S_train.applymap(ast.literal_eval).values.tolist()))
                 ), axis = 1
                )
                 
tcoil_x_test = np.array(tcoil_test[['L','W','S','Nin','Nout']].copy())
tcoil_S_test = tcoil_test[['s11', 's12', 's13', 's22', 's23', 's33']].copy()
tcoil_S_test = np.concatenate(
                (np.abs(np.array(tcoil_S_test.applymap(ast.literal_eval).values.tolist())),
                 np.angle(np.array(tcoil_S_test.applymap(ast.literal_eval).values.tolist()))
                 ), axis = 1
                )


# normalize the input data    
mean_x_train = tcoil_x_train.mean(axis=0)
std_x_train = tcoil_x_train.std(axis=0) 
tcoil_x_train_norm = (tcoil_x_train-mean_x_train)/std_x_train
tcoil_x_test_norm = (tcoil_x_test-mean_x_train)/std_x_train

# normalize the output data   
mean_y_train = tcoil_S_train.mean(axis=2)
std_y_train = tcoil_S_train.std(axis=2)
tcoil_y_train= np.concatenate(
                (mean_y_train,
                 std_y_train
                 ), axis = 1
                )
tcoil_y_train_norm = -1+2*(tcoil_y_train - tcoil_y_train.min(axis=0)) / (tcoil_y_train.max(axis=0) - tcoil_y_train.min(axis=0))

mean_y_test = tcoil_S_test.mean(axis=2)
std_y_test = tcoil_S_test.std(axis=2)
tcoil_y_test = np.concatenate(
                (mean_y_test,
                 std_y_test
                 ), axis = 1
                )

######################### return the trained mlp model ########################

tcoil_model = mlp(tcoil_x_train_norm, tcoil_y_train_norm)

################# test the forward inference tcoil mlp model ##################
tcoil_predictions = (tcoil_model.predict(tcoil_x_test_norm) + 1)/2 * (tcoil_y_train.max(axis=0) - tcoil_y_train.min(axis=0)) + tcoil_y_train.min(axis=0)


tcoil_labels_test_df = pd.DataFrame(data=tcoil_y_test, columns = ['mean(mag(s11))', 'mean(mag(s12))', 
                                                                       'mean(mag(s13))', 'mean(mag(s22))', 
                                                                       'mean(mag(s23))', 'mean(mag(s33))',
                                                                       'mean(ph(s11))', 'mean(ph(s12))',
                                                                       'mean(ph(s13))', 'mean(ph(s22))',
                                                                       'mean(ph(s23))', 'mean(ph(s33))',
                                                                       
                                                                       'std(mag(s11))', 'std(mag(s12))', 
                                                                       'std(mag(s13))', 'std(mag(s22))', 
                                                                       'std(mag(s23))', 'std(mag(s33))',
                                                                       'std(ph(s11))', 'std(ph(s12))',
                                                                       'std(ph(s13))', 'std(ph(s22))',
                                                                       'std(ph(s23))', 'std(ph(s33))'])

tcoil_predictions_df = pd.DataFrame(data=tcoil_predictions, columns = ['mean(mag(s11))', 'mean(mag(s12))', 
                                                                       'mean(mag(s13))', 'mean(mag(s22))', 
                                                                       'mean(mag(s23))', 'mean(mag(s33))',
                                                                       'mean(ph(s11))', 'mean(ph(s12))',
                                                                       'mean(ph(s13))', 'mean(ph(s22))',
                                                                       'mean(ph(s23))', 'mean(ph(s33))',
                                                                       
                                                                       'std(mag(s11))', 'std(mag(s12))', 
                                                                       'std(mag(s13))', 'std(mag(s22))', 
                                                                       'std(mag(s23))', 'std(mag(s33))',
                                                                       'std(ph(s11))', 'std(ph(s12))',
                                                                       'std(ph(s13))', 'std(ph(s22))',
                                                                       'std(ph(s23))', 'std(ph(s33))'])

s11_mag_test_list = (tcoil_S_test[:,0,:] - 
                     np.array(tcoil_labels_test_df['mean(mag(s11))']).reshape(len(tcoil_S_test),-1))/(np.array(tcoil_labels_test_df['std(mag(s11))']).reshape(len(tcoil_S_test),-1))
s11_mag_pred_list = (tcoil_S_test[:,0,:] - 
                     np.array(tcoil_predictions_df['mean(mag(s11))']).reshape(len(tcoil_S_test),-1))/(np.array(tcoil_predictions_df['std(mag(s11))']).reshape(len(tcoil_S_test),-1))



plt.figure('La Eq. Ckt vs. La ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['La'], tcoil_predictions_df['La'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['La']),np.max(tcoil_labels_test_df['La'])],
            [np.min(tcoil_labels_test_df['La']),np.max(tcoil_labels_test_df['La'])], 'r--')
axs[0].set(ylabel='La = imag($Z_{11}$/$\omega$)\nPrediction (H)')
axs[1].plot(tcoil_labels_test_df['La'],
            np.abs(tcoil_labels_test_df['La'] - tcoil_predictions_df['La'])/np.array(tcoil_labels_test_df['La']) * 100,
            '^')
axs[1].set(xlabel='La = imag($Z_{11}$/$\omega$) True (H)',
           ylabel='Deviation (%)')                            
fig.suptitle('Predicted La vs. True La')
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
axs[1].set(xlabel='Lb = imag($Z_{22}$/$\omega$) True (H)',
           ylabel='Deviation (%)')                            
fig.suptitle('Predicted Lb vs. True Lb')
axs[0].grid()
axs[1].grid()

plt.figure('Qa Eq. Ckt vs. Qa ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Qa'], tcoil_predictions_df['Qa'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])],
            [np.min(tcoil_labels_test_df['Qa']),np.max(tcoil_labels_test_df['Qa'])], 'r--')
axs[0].set(ylabel='Qa Prediction')
axs[1].plot(tcoil_labels_test_df['Qa'],
            np.abs(tcoil_labels_test_df['Qa'] - tcoil_predictions_df['Qa'])/np.array(tcoil_labels_test_df['Qa']) * 100,
            '^')
axs[1].set(xlabel='Qa True',
           ylabel='Deviation (%)')                            
fig.suptitle('Predicted Qa vs. True Qa')
axs[0].grid()
axs[1].grid()

plt.figure('Qb Eq. Ckt vs. Qb ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Qb'], tcoil_predictions_df['Qb'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])],
            [np.min(tcoil_labels_test_df['Qb']),np.max(tcoil_labels_test_df['Qb'])], 'r--')
axs[0].set(ylabel='Qb Prediction')
axs[1].plot(tcoil_labels_test_df['Qb'],
            np.abs(tcoil_labels_test_df['Qb'] - tcoil_predictions_df['Qb'])/np.array(tcoil_labels_test_df['Qb']) * 100,
            '^')
axs[1].set(xlabel='Qb True',
           ylabel='Deviation (%)')                            
fig.suptitle('Predicted Qb vs. True Qb')
axs[0].grid()
axs[1].grid()

plt.figure('k Eq. Ckt vs. k ML Predicted')                                               
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['k'], tcoil_predictions_df['k'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])],
            [np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])], 'r--')
axs[0].set(ylabel='k Prediction')
axs[1].plot(tcoil_labels_test_df['k'],
            np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k'])/np.abs(tcoil_labels_test_df['k']) * 100,
            '^')
axs[1].set(xlabel='k True',
           ylabel='Deviation (%)')                            
fig.suptitle('Predicted k vs. True k')
axs[0].grid()
axs[1].grid()

plt.figure('Cbr Eq. Ckt vs. Cbr ML Predicted')                                               
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Cbr'], tcoil_predictions_df['Cbr'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Cbr']),np.max(tcoil_labels_test_df['Cbr'])],
            [np.min(tcoil_labels_test_df['Cbr']),np.max(tcoil_labels_test_df['Cbr'])], 'r--')
axs[0].set(ylabel='Cbr Prediction (F)')
axs[1].plot(tcoil_labels_test_df['Cbr'],
            np.abs(tcoil_labels_test_df['Cbr'] - tcoil_predictions_df['Cbr'])/np.abs(tcoil_labels_test_df['Cbr']) * 100,
            '^')
axs[1].set(xlabel='Cbr Eq. Ckt',
           ylabel='Deviation (%)')                            
fig.suptitle('Predicted Cbr vs. True Cbr')
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
La_target = 2e-10
Lb_target = 1.5e-10
k_target = -0.5

from pymoo.model.problem import Problem

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, 
                         n_obj=2, 
                         n_constr=4, 
                         xl=np.array([32, 1, 4, 5]), 
                         xu=np.array([80, 3, 12, 24]), 
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
            print(f'[La, Qa, Lb, Qb, k, Cbr]: {predictions}')
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
# best set: L = 66, W = 5 ('3' represents 5 um here), Nin = 6, Nout = 6 -> La = 2.02e-10, Qa = 3.62, Lb = 1.53e-10, Qb = 2.69, k = -4.84

from pymoo.visualization.scatter import Scatter
plot = Scatter(title = 'Objective Space')
plot.add(res.F)
plot.show()

n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt_history = np.array([e.opt[0].F for e in res.history]) * np.array([-1,1]) # flip the sign of Q
opt_history = pd.DataFrame(opt_history, columns=['Qa+Qb', 'area'])
input_history = np.array([e.opt[0].X for e in res.history]) 
input_history = pd.DataFrame(input_history, columns=['L', 'W', 'Nin', 'Nout'])
output_history = tcoil_model.predict(scaler_in.transform(input_history[input_history.columns])) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
output_history = pd.DataFrame(output_history, columns=['La', 'Qa', 'Lb', 'Qb', 'k', 'Cbr'])

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
