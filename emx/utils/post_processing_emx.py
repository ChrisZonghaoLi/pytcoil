# This script is used to extract La, Lb, Ra, Rb, k, Cbr, etc.

import skrf as rf
import numpy as np
import math
import os
import errno
import ast
import csv
import pandas as pd
import yaml
from datetime import datetime
import re

import sys
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
EMX_WORK_DIR = os.environ['EMX_WORK_DIR']
TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']
sys.path.append(PYTCOIL_DIR)
from common.eq_ckt import input_ind_res_asitic, s3p2s2p, t_network_eq_ckt

import matplotlib.pyplot as plt
plt.style.use(style='default')
plt.rcParams['font.family']='serif'

"""
This function summaries La, Lb, Ra, Rb, k, Cbr, and all parasitic of
all t-coil deisgn. 

----------
Parameters
----------
ind_num : int
    total number of t-coil design
middle_tap: int
    port of t-coil that will be grounded for calculating La, Lb, Ra, Rb, k
    using the T-network
    
    1 --[tcoil] --2
           |
           | 3
          ---
           -

Returns
-------
tcoil: dictionary

tcoil = {
    'tcoil0':{
        'La':
        'Lb':
        'Ra':
        'Rb':
        'k':
        'Cbr':
        'Cox_in':
        'Rsub_in':
        'Csub_in':
        'Cox_mid':
        'Rsub_mid':
        'Csub_mid':
        'Rsub_mid':
        'Cox_out':
        'Rsub_out':
        'Csub_out':
        'L':
        'W':
        'S':
        'tap':
        },
    'tcoil1':{
        ...
        },
    ...
    
    }

"""

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'

rx = re.compile(numeric_const_pattern, re.VERBOSE)
 
class PostProcessingEMX():

    def __init__(self):
            
        self.stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
        self.sim_setups = yaml.load(self.stream, yaml.SafeLoader)
        
        self.tcoil_num_new = int(self.sim_setups['tcoil_num_new'])
        self.tcoil_num_old = int(self.sim_setups['tcoil_num_old'])
        self.freq_start = float(self.sim_setups['freq_start'])
        self.freq_stop = float(self.sim_setups['freq_stop'])
        self.freq_step = float(self.sim_setups['freq_step'])
        self.middle_tap = int(self.sim_setups['middle_tap'])
        self.tcoil_dims_name = self.sim_setups['tcoil_dims_name']

        
        self.date = datetime.today().strftime('%Y-%m-%d')
        
    def summary_tcoil(self):
        tcoil = {}
        tcoil_dims=pd.read_csv(f'{TCOIL_DATA_DIR}/train/{self.tcoil_dims_name}.csv', usecols = ['L', 'W', 'S', 'Nin', 'Nout'])     
        for i in range(self.tcoil_num_new):
            print(f"Processing tcoil{i} data.")
            tcoil['tcoil{}'.format(i)] = {}
            tcoil['tcoil{}'.format(i)]['index'] = i
            network = rf.Network(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s3p')
            s_params = network.s
            
            # tcoil s-param
            s11_list = [s_params[i][0][0] for i in range(len(s_params))]
            s12_list = [s_params[i][0][1] for i in range(len(s_params))]
            s13_list = [s_params[i][0][2] for i in range(len(s_params))]
            s21_list = [s_params[i][1][0] for i in range(len(s_params))]
            s22_list = [s_params[i][1][1] for i in range(len(s_params))]
            s23_list = [s_params[i][1][2] for i in range(len(s_params))]
            s31_list = [s_params[i][2][0] for i in range(len(s_params))]
            s32_list = [s_params[i][2][1] for i in range(len(s_params))]
            s33_list = [s_params[i][2][2] for i in range(len(s_params))]
            
            tcoil['tcoil{}'.format(i)]['s11'] = s11_list
            tcoil['tcoil{}'.format(i)]['s12'] = s12_list
            tcoil['tcoil{}'.format(i)]['s13'] = s13_list
            tcoil['tcoil{}'.format(i)]['s21'] = s21_list
            tcoil['tcoil{}'.format(i)]['s22'] = s22_list
            tcoil['tcoil{}'.format(i)]['s23'] = s23_list
            tcoil['tcoil{}'.format(i)]['s31'] = s31_list
            tcoil['tcoil{}'.format(i)]['s32'] = s32_list
            tcoil['tcoil{}'.format(i)]['s33'] = s33_list
            
            # tcoil dimensions
            tcoil['tcoil{}'.format(i)]['L'] = tcoil_dims['L'][i]
            tcoil['tcoil{}'.format(i)]['W'] = tcoil_dims['W'][i]
            tcoil['tcoil{}'.format(i)]['S'] = tcoil_dims['S'][i]
            tcoil['tcoil{}'.format(i)]['Nin'] = tcoil_dims['Nin'][i]
            tcoil['tcoil{}'.format(i)]['Nout'] = tcoil_dims['Nout'][i]
            
            ## La, Lb, Ra, Rb, k ##
            tcoil['tcoil{}'.format(i)]['La'] = t_network_eq_ckt(network, self.middle_tap)['La']
            tcoil['tcoil{}'.format(i)]['Lb'] = t_network_eq_ckt(network, self.middle_tap)['Lb']
            tcoil['tcoil{}'.format(i)]['Ra'] = t_network_eq_ckt(network, self.middle_tap)['Ra']
            tcoil['tcoil{}'.format(i)]['Rb'] = t_network_eq_ckt(network, self.middle_tap)['Rb']
            tcoil['tcoil{}'.format(i)]['Qa'] = t_network_eq_ckt(network, self.middle_tap)['Qa']
            tcoil['tcoil{}'.format(i)]['Qb'] = t_network_eq_ckt(network, self.middle_tap)['Qb']
            # for GF22 k is negative value
            tcoil['tcoil{}'.format(i)]['k'] = -t_network_eq_ckt(network, self.middle_tap)['k']
            
            # SRF
            tcoil['tcoil{}'.format(i)]['fr'] = t_network_eq_ckt(network, self.middle_tap)['fr']
            
         
        return tcoil
        
    
    def save2csv(self, tcoil_results, plot=True, format='csv'):    
    
        index_list = []    
    
        La_list = []
        Lb_list = []
        Ra_list = []
        Rb_list = []
        Qa_list = []
        Qb_list = []
        k_list = [] 
        fr_list = []
        
        #############################
        
        L_list = []
        W_list = []
        S_list = []
        Nin_list = []
        Nout_list = []
        
        
        ##############################
        
        s11_list = []
        s12_list = []
        s13_list = []
        s21_list = []
        s22_list = []
        s23_list = []
        s31_list = []
        s32_list = []
        s33_list = []
        
        
        for i in range(self.tcoil_num_new):
                
            La = tcoil_results['tcoil{}'.format(i)]['La']
            Lb = tcoil_results['tcoil{}'.format(i)]['Lb']
            Ra = tcoil_results['tcoil{}'.format(i)]['Ra']
            Rb = tcoil_results['tcoil{}'.format(i)]['Rb']
            Qa = tcoil_results['tcoil{}'.format(i)]['Qa']
            Qb = tcoil_results['tcoil{}'.format(i)]['Qb']
            k = tcoil_results['tcoil{}'.format(i)]['k']
            fr = tcoil_results['tcoil{}'.format(i)]['fr']
        
            index_list.append(tcoil_results['tcoil{}'.format(i)]['index'])
            
            L_list.append(tcoil_results['tcoil{}'.format(i)]['L'])
            W_list.append(tcoil_results['tcoil{}'.format(i)]['W'])
            S_list.append(tcoil_results['tcoil{}'.format(i)]['S'])
            Nin_list.append(tcoil_results['tcoil{}'.format(i)]['Nin'])
            Nout_list.append(tcoil_results['tcoil{}'.format(i)]['Nout'])
            
            La_list.append(La)
            Lb_list.append(Lb)
            Ra_list.append(Ra)
            Rb_list.append(Rb)
            Qa_list.append(Qa)
            Qb_list.append(Qb)
            k_list.append(k)
            fr_list.append(fr)
            
            s11_list.append(tcoil_results['tcoil{}'.format(i)]['s11'])
            s12_list.append(tcoil_results['tcoil{}'.format(i)]['s12'])
            s13_list.append(tcoil_results['tcoil{}'.format(i)]['s13'])
            s21_list.append(tcoil_results['tcoil{}'.format(i)]['s21'])
            s22_list.append(tcoil_results['tcoil{}'.format(i)]['s22'])
            s23_list.append(tcoil_results['tcoil{}'.format(i)]['s23'])
            s31_list.append(tcoil_results['tcoil{}'.format(i)]['s31'])
            s32_list.append(tcoil_results['tcoil{}'.format(i)]['s32'])
            s33_list.append(tcoil_results['tcoil{}'.format(i)]['s33'])
            
                     
        ######################################################################
        tcoil_data = pd.DataFrame([np.array(index_list).tolist(),
                                   np.array(L_list).tolist(),
                                   np.array(W_list).tolist(),
                                   np.array(S_list).tolist(),
                                   np.array(Nin_list).tolist(),
                                   np.array(Nout_list).tolist(),
                                   
                                   np.array(La_list).tolist(),
                                   np.array(Ra_list).tolist(),
                                   np.array(Qa_list).tolist(),
                                   np.array(Lb_list).tolist(),
                                   np.array(Rb_list).tolist(),
                                   np.array(Qb_list).tolist(),
                                   np.array(k_list).tolist(),
                                   np.array(fr_list).tolist(),
                                   
                                   np.array(s11_list).tolist(),
                                   np.array(s12_list).tolist(),
                                   np.array(s13_list).tolist(),
                                   np.array(s21_list).tolist(),
                                   np.array(s22_list).tolist(),
                                   np.array(s23_list).tolist(),
                                   np.array(s31_list).tolist(),
                                   np.array(s32_list).tolist(),
                                   np.array(s33_list).tolist()]).T
        
        tcoil_data.columns=['index', 'L', 'W', 'S', 'Nin', 'Nout', 'La', 'Ra', 'Qa', 'Lb', 'Rb', 'Qb', 'k', 'fr', 's11', 's12', 's13', 's21', 's22', 's23', 's31', 's32', 's33']
        tcoil_data.to_csv(f'{TCOIL_DATA_DIR}/train/tcoil_{self.freq_start/1e9}-{self.freq_stop/1e9}GHz_{self.tcoil_num_new}_{self.date}.csv',mode='w',header=True)
        
        if format == 'json':
            tcoil_data.to_json(f'{TCOIL_DATA_DIR}/train/tcoil_{self.freq_start/1e9}-{self.freq_stop/1e9}GHz_{self.tcoil_num_new}_{self.date}.json', orient='split')

                
