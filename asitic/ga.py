import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# import pymoo
from pymoo.model.problem import FunctionalProblem
from pymoo.model.problem import ConstraintsAsPenaltyProblem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_termination, get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

# import pool for multiprocessing
import multiprocessing
from multiprocessing.pool import ThreadPool

# import mlp_forward
from tcoil_mlp import mlp

# import INN for narrowing down the searching space for GA
import os
import sys
sys.path.append(os.path.abspath("/fs1/eecg/tcc/lizongh2/TCoil_ML/invertible_neural_networks"))
from invertible_neural_networks.tcoil_inn import inn
from invertible_neural_networks.posteriors import Posteriors

######################## Data import and pre-processing #######################

tcoil_data = pd.read_csv('./data/7nm/train/tcoil_results_1GHz_6198_middle_branch=False.csv',names=['L', 'W', 'S',
                                 'N', 'tap', 'La',
                                 'Ra', 'Lb', 'Rb',
                                 'k', 'Cbr'])

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)
 
tcoil_x_train = tcoil_train[['L','W','S','N','tap']].copy()
tcoil_y_train = tcoil_train[['La','Ra','Lb','Rb','k']].copy()
tcoil_y_train_inn = tcoil_train[['La','Lb','k']].copy()

tcoil_x_test = tcoil_test[['L','W','S','N','tap']].copy()
tcoil_y_test = tcoil_test[['La','Ra','Lb','Rb','k']].copy()
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

######################### return the trained mlp model ########################

tcoil_model = mlp(tcoil_x_train, tcoil_y_train)

################# test the forward inference tcoil mlp model ##################
tcoil_predictions = tcoil_model.predict(tcoil_x_test) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_


tcoil_labels_test_df = pd.DataFrame(data=tcoil_y_test, columns = ['La', 'Ra', 'Lb', 'Rb', 'k'])
tcoil_predictions_df = pd.DataFrame(data=tcoil_predictions, columns = ['La', 'Ra', 'Lb', 'Rb', 'k'])


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

plt.figure('Ra Eq. Ckt vs. Ra ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Ra'], tcoil_predictions_df['Ra'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Ra']),np.max(tcoil_labels_test_df['Ra'])],
            [np.min(tcoil_labels_test_df['Ra']),np.max(tcoil_labels_test_df['Ra'])], 'r--')
axs[0].set(ylabel='Ra = real($Z_{11}$)\nML Prediction ($\Omega$)')
axs[1].plot(tcoil_labels_test_df['Ra'],
            np.abs(tcoil_labels_test_df['Ra'] - tcoil_predictions_df['Ra'])/np.array(tcoil_labels_test_df['Ra']) * 100,
            '^')
axs[1].set(xlabel='Ra = real($Z_{11}$) Eq. Ckt ($\Omega$)',
           ylabel='Deviation (%)')                            
fig.suptitle('True Ra vs. Predicted Ra')
axs[0].grid()
axs[1].grid()

plt.figure('Rb Eq. Ckt vs. Rb ML Predicted')
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['Rb'], tcoil_predictions_df['Rb'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['Rb']),np.max(tcoil_labels_test_df['Rb'])],
            [np.min(tcoil_labels_test_df['Rb']),np.max(tcoil_labels_test_df['Rb'])], 'r--')
axs[0].set(ylabel='Rb = real($Z_{22}$)\nML Prediction ($\Omega$)')
axs[1].plot(tcoil_labels_test_df['Rb'],
            np.abs(tcoil_labels_test_df['Rb'] - tcoil_predictions_df['Rb'])/np.array(tcoil_labels_test_df['Rb']) * 100,
            '^')
axs[1].set(xlabel='Rb = real($Z_{22}$) Eq. Ckt ($\Omega$)',
           ylabel='Deviation (%)')                            
fig.suptitle('True Rb vs. Predicted Rb')
axs[0].grid()
axs[1].grid()

plt.figure('k Eq. Ckt vs. k ML Predicted')
k_flaw = np.where(tcoil_labels_test_df['k']<0.15)
tcoil_labels_test_df['k']=tcoil_labels_test_df['k'].drop(k_flaw)
tcoil_predictions_df['k']=tcoil_predictions_df['k'].drop(k_flaw)                                                
fig, axs = plt.subplots(2)
axs[0].plot(tcoil_labels_test_df['k'], tcoil_predictions_df['k'], '^')
axs[0].plot([np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])],
            [np.min(tcoil_labels_test_df['k']),np.max(tcoil_labels_test_df['k'])], 'r--')
axs[0].set(ylabel='k ML Prediction')
axs[1].plot(tcoil_labels_test_df['k'],
            np.abs(tcoil_labels_test_df['k'] - tcoil_predictions_df['k'])/np.array(tcoil_labels_test_df['k']) * 100,
            '^')
axs[1].set(xlabel='k Eq. Ckt',
           ylabel='Deviation (%)')                            
fig.suptitle('True k vs. Predicted k')
axs[0].grid()
axs[1].grid()

############################# import INN  ###############################

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

############################# GA implementation ###############################


def f1(x):
    # minimize the resistance of Ra + Rb
    x_copy = np.copy(x)
    x_copy[3] = x_copy[3] * 0.25
    #print('Tcoil geometry [L, W, S, N, tap]: {}'.format(x_copy))
    x_copy = scaler_in.transform(x_copy.reshape(1,-1))
    predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
    res = predictions[0][1] + predictions[0][3]
    #La = predictions[0][0]
    #Lb = predictions[0][2]
    #k = predictions[0][4]

    #print('Total res: {} | La: {} | Lb: {} | k: {}'.format(res, La, Lb, k))

    return res

def f2(x):
    # minimize the area of the tcoil
    x_copy = np.copy(x)
    L, W, S, N, tap = x_copy
    L_tot = N * L
    if N % 2 == 0:
        W_tot = (int((N-2)/2)+1)*int((N-2)/2) + int(N/2) # int is taking the lower flooer
    else:
        W_tot = (int(N/2)+1)*int(N/2)     
    S_tot = W_tot - (N-1)

    area = W * (L_tot - W_tot - S_tot)    
    #print('Tcoil total area (um^2): {}'.format(area))
    
    return area

def g1(x):
    # constraint on the deviation of La should be less than 10%  
    
    x_copy = np.copy(x)
    x_copy[3] = x_copy[3] * 0.25
    x_copy = scaler_in.transform(x_copy.reshape(1,-1))
    predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_
    
    La = predictions[0][0]
  
    loss = np.abs(La/1e-10 - La_target/1e-10) - 0.1*La_target/1e-10
    # print(La, loss)
    #print(x)

    return loss

def g2(x):
    # constraint on the deviation of Lb should be less than 10%  
    
    x_copy = np.copy(x)
    x_copy[3] = x_copy[3] * 0.25
    x_copy = scaler_in.transform(x_copy.reshape(1,-1))
    predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_

    Lb = predictions[0][2]

    loss = np.abs(Lb/1e-10 - Lb_target/1e-10) - 0.1*Lb_target/1e-10
    # print(La, loss)
    return loss


def g3(x):
    # constraint on the deviation of k should be less than 10%  
    
    x_copy = np.copy(x)
    x_copy[3] = x_copy[3] * 0.25    
    x_copy = scaler_in.transform(x_copy.reshape(1,-1))
    predictions = tcoil_model.predict(x_copy) * np.sqrt(mean_std_train_y.var_) + mean_std_train_y.mean_

    k = predictions[0][4]
        
    loss = np.abs(k - k_target) - 0.1*k_target
    # print(La, loss)
    return loss

def g4(x):
    # pose constraint for N that has to be smaller than the maximum allowable turns for a given set of L, W ,and 
    
    x_copy = np.copy(x)
    L = x_copy[0]
    W = x_copy[1]
    S = x_copy[2]
    N = x_copy[3] * 0.25
    N_max = int((L/(W+S))/2/0.25)*0.25
    diff = N - N_max

    return diff

def g5(x):
    # pose constraint fot tap number that has to be smaller than 4*N
    
    x_copy = np.copy(x)
    N = x_copy[3] * 0.25
    tap = x_copy[4]
    tap_max = int(4*N)
    diff = tap - tap_max

    return diff

objs = [f1, f2]
constr_ieq = [g1, g2, g3, g4, g5]

mask = ['real', 'real', 'real', 'int', 'int']

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

# from pymoo.util.display import MultiObjectiveDisplay
# class MyDisplay(MultiObjectiveDisplay):
#     def _do(self, problem, evaluator, algorithm):
#         super()._do(problem, evaluator, algorithm)
#         self.output.append('[L, W, S, N, tap]', algorithm.pop.get('X')[0] * np.array([1,1,1,0.25,1]))
#         self.output.append('[res, area (um^2)]', algorithm.pop.get('F')[0])
        

problem = FunctionalProblem(np.shape(tcoil_x_train)[1],
                            objs,
                            #xl = np.array([30, 1.5, 1, 8, 4]), # why N is from 8 - 48? I do this so that I can use integer random number to represent N in terms of n*0,25
                            xl = x_low,
                            #xu = np.array([60, 2, 2, 48, 45]),
                            xu = x_high,
                            constr_ieq = constr_ieq,
                            constr_eq = [],                        
                            parallelization = ('starmap', pool.starmap)
                            )

problem = ConstraintsAsPenaltyProblem(problem, penalty=1e6)

termination = get_termination("n_gen", 100)

algorithm = NSGA2(
            pop_size=100,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True)

    
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
ax3.plot(n_evals, output_history['k'], 'g-', label='k')
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
