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

import sys
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
HSPICE_WORK_DIR = os.environ['HSPICE_WORK_DIR']
ASITIC_WORK_DIR = os.environ['ASITIC_WORK_DIR']
TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']
from utils.eq_ckt import input_ind_res_asitic, s3p2s2p, t_network_eq_ckt

import matplotlib.pyplot as plt
plt.style.use(style='default')
plt.rcParams['font.family']='serif'

class PostProcessingASITIC():
            
    """
        freq_design: frequency of interest (in GHz)
        accuracy (in decimal): deviation between eq. ckt. and ASITIC results larger than
                               this value will be dropped
        tcoil_num: # of tcoil designs
        mode: either 'train' or 'test'
        middle_tap: port num for the tcoil middle tap 
        branch: 2-port S params of the eq. ckt. branch used to compare to ASITIC results
        termination: since it is the 2-port S params used to be compared to, the left one 
                     port of the tcoil will be left 'open' by default
        hspcei_ser_opt: True by default; using HSPICE optimization results for series components
        middle_branch: True by default; the 2-pi model is used for tcoil
       
    """

    def __init__(self, mode):
        self.mode = mode
        self.stream = open(f'{PYTCOIL_DIR}/asitic/sim_setup_asitic.yaml','r')
        self.sim_setups = yaml.load(self.stream, yaml.SafeLoader)
        self.tcoil_num_old = int(self.sim_setups['tcoil_num_old'])
        self.tcoil_num_new = int(self.sim_setups['tcoil_num_new'])
        self.tcoil_num_test = int(self.sim_setups['tcoil_num_test'])

        self.freq_start = float(self.sim_setups['freq_start'])
        self.freq_stop = float(self.sim_setups['freq_stop'])
        self.freq_step = float(self.sim_setups['freq_step'])
        self.freq_design = float(self.sim_setups['freq_design'])
        self.hspice_ser_opt = self.sim_setups['hspice_ser_opt']
        self.middle_branch = self.sim_setups['middle_branch']
        self.middle_tap = int(self.sim_setups['middle_tap'])
        
        self.accuracy = float(self.sim_setups['accuracy'])
        self.ind_max = float(self.sim_setups['ind_max'])
        
        self.date = datetime.today().strftime('%Y-%m-%d')
        
        if self.mode == 'test':
            self.tcoil_num_old = 0
            self.tcoil_num_new = self.tcoil_num_test
        elif self.mode == 'train':
            None
        else:
            print('Wrong entry for "mode", should be either "train" or "test".')
        
    def summary_tcoil(self):
        
        """
        This function summaries La, Lb, Ra, Rb, k, Cbr, and all parasitic of
        all t-coil deisgn. 
        
        ----------
        Parameters
        ----------
        tcoil_num : int
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
                'N':
                'tap':
                },
            'tcoil1':{
                ...
                },
            ...
            
            }
        
        """
        
        
        tcoil = {}
        
        for i in range(self.tcoil_num_new):
            network_eq_ckt = rf.Network('{}/{}/tcoil{}/tcoil{}.s3p'.format(HSPICE_WORK_DIR,self.mode,i,i))
            
            tcoil['tcoil{}'.format(i)] = {}
            
            with open('{}/{}/tcoil{}/tcoil{}.sp'.format(HSPICE_WORK_DIR,self.mode,i,i),'r') as tcoil_netlist:
                    lines = []
                    for line in tcoil_netlist.readlines():
                        lines.append(str.split(line))
            
            ## La, Lb, Ra, Rb, k ##
            tcoil['tcoil{}'.format(i)]['La'] = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['La']
            tcoil['tcoil{}'.format(i)]['Lb'] = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Lb']
            tcoil['tcoil{}'.format(i)]['Ra'] = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Ra']
            tcoil['tcoil{}'.format(i)]['Rb'] = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Rb']
            tcoil['tcoil{}'.format(i)]['k'] = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['k']
            
            ## parasitic ##
            if self.middle_branch == True: 
                tcoil['tcoil{}'.format(i)]['Cox_in'] = float(lines[12][3])
                tcoil['tcoil{}'.format(i)]['Rsub_in'] = float(lines[14][3])
                tcoil['tcoil{}'.format(i)]['Csub_in'] = float(lines[13][3])
                
                tcoil['tcoil{}'.format(i)]['Cox_mid'] = float(lines[15][3])
                tcoil['tcoil{}'.format(i)]['Rsub_mid'] = float(lines[17][3])
                tcoil['tcoil{}'.format(i)]['Csub_mid'] = float(lines[16][3]) 
            
                tcoil['tcoil{}'.format(i)]['Cox_out'] = float(lines[22][3])
                tcoil['tcoil{}'.format(i)]['Rsub_out'] = float(lines[24][3])
                tcoil['tcoil{}'.format(i)]['Csub_out'] = float(lines[23][3])
               
            else:
                tcoil['tcoil{}'.format(i)]['Cox_in'] = float(lines[12][3])
                tcoil['tcoil{}'.format(i)]['Rsub_in'] = float(lines[14][3])
                tcoil['tcoil{}'.format(i)]['Csub_in'] = float(lines[13][3])
            
                tcoil['tcoil{}'.format(i)]['Cox_out'] = float(lines[22][3])
                tcoil['tcoil{}'.format(i)]['Rsub_out'] = float(lines[24][3])
                tcoil['tcoil{}'.format(i)]['Csub_out'] = float(lines[23][3])
    
            ## Cbr ##
            # Read Cbr from HSPICE optimization output
            with open('{}/{}/tcoil{}/tcoil{}_ext.ma0'.format(HSPICE_WORK_DIR,self.mode,i,i),'r') as hspice_opt:
                opt_output = []
                for output in hspice_opt.readlines():
                    opt_output.append(str.split(output))
                Cbr = float(opt_output[6][1])
            tcoil['tcoil{}'.format(i)]['Cbr'] = Cbr
            
        ## t-coil geometry info ##      
        if self.mode == 'train':      
            tcoil_dims=pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_dims.csv', usecols = ['L', 'W', 'S', 'N', 'tap'])     
            for i in range(self.tcoil_num_new):
                tcoil['tcoil{}'.format(i)]['L'] = tcoil_dims['L'][i]
                tcoil['tcoil{}'.format(i)]['W'] = tcoil_dims['W'][i]
                tcoil['tcoil{}'.format(i)]['S'] = tcoil_dims['S'][i]
                tcoil['tcoil{}'.format(i)]['N'] = tcoil_dims['N'][i]
                tcoil['tcoil{}'.format(i)]['tap'] = tcoil_dims['tap'][i]
        elif self.mode == 'test':
            tcoil_dims=pd.read_csv(f'{TCOIL_DATA_DIR}/test/tcoil_dims.csv', usecols = ['L', 'W', 'S', 'N', 'tap'])     
            for i in range(self.tcoil_num_old, self.tcoil_num_new):
                tcoil['tcoil{}'.format(i)]['L'] = tcoil_dims['L'][i]
                tcoil['tcoil{}'.format(i)]['W'] = tcoil_dims['W'][i]
                tcoil['tcoil{}'.format(i)]['S'] = tcoil_dims['S'][i]
                tcoil['tcoil{}'.format(i)]['N'] = tcoil_dims['N'][i]
                tcoil['tcoil{}'.format(i)]['tap'] = tcoil_dims['tap'][i]
        else:
            print('Wrong entry for the "mode", should be either "train" or "test".')
        return tcoil



    def rl_eq_ckt_vs_asitic(self, network_asitic_a, network_asitic_b, network_eq_ckt):
        
        La_asitic = input_ind_res_asitic(network_asitic_a)['Lin']
        Lb_asitic = input_ind_res_asitic(network_asitic_b)['Lin']
        Ra_asitic = input_ind_res_asitic(network_asitic_a)['Rin']
        Rb_asitic = input_ind_res_asitic(network_asitic_b)['Rin']
        
        La_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['La']
        Lb_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Lb']
        Ra_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Ra']
        Rb_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Rb']
        
        La_diff_perc = abs(np.array(La_asitic)-np.array(La_eq_ckt))/np.array(La_asitic)
        Lb_diff_perc = abs(np.array(Lb_asitic)-np.array(Lb_eq_ckt))/np.array(Lb_asitic)
        Ra_diff_perc = abs(np.array(Ra_asitic)-np.array(Ra_eq_ckt))/np.array(Ra_asitic)
        Rb_diff_perc = abs(np.array(Rb_asitic)-np.array(Rb_eq_ckt))/np.array(Rb_asitic)
        
        
        return {
                'La_diff_perc': La_diff_perc, 
                'Lb_diff_perc': Lb_diff_perc, 
                'Ra_diff_perc': Ra_diff_perc, 
                'Rb_diff_perc': Rb_diff_perc,
                
                'La_asitic': La_asitic,
                'Lb_asitic': Lb_asitic,
                'Ra_asitic': Ra_asitic,
                'Rb_asitic': Rb_asitic,
                
                'La_eq_ckt': La_eq_ckt,
                'Lb_eq_ckt': Lb_eq_ckt,
                'Ra_eq_ckt': Ra_eq_ckt,
                'Rb_eq_ckt': Rb_eq_ckt
                }



    def s2p_eq_ckt_vs_asitic(self, network_asitic, network_eq_ckt):
        
        
        s2p = network_asitic.s
        
        s11_list = [s2p[i][0][0] for i in range(len(s2p))]
        s12_list = [s2p[i][0][1] for i in range(len(s2p))]
        
        s11_ph = np.angle(s11_list, deg=True)
        s11_mag = np.absolute(s11_list)
        
        s12_ph = np.angle(s12_list, deg=True)
        s12_mag = np.absolute(s12_list)
        
        # when compare 2-port S params, centertap is left floating
        s2p_eq_ckt = s3p2s2p(network_eq_ckt, self.middle_tap, 'open')
        
        s11_list_eq_ckt = [s2p_eq_ckt[i][0][0] for i in range(len(s2p_eq_ckt))]
        s12_list_eq_ckt = [s2p_eq_ckt[i][0][1] for i in range(len(s2p_eq_ckt))]
        
        s11_ph_eq_ckt = np.angle(s11_list_eq_ckt, deg=True)
        s11_mag_eq_ckt = np.absolute(s11_list_eq_ckt)
        
        s12_ph_eq_ckt = np.angle(s12_list_eq_ckt, deg=True)
        s12_mag_eq_ckt = np.absolute(s12_list_eq_ckt)
    
        s11_mag_diff = abs(np.array(s11_mag - s11_mag_eq_ckt))
        s11_ph_diff = abs(np.array(s11_ph - s11_ph_eq_ckt))
    
        s12_mag_diff = abs(np.array(s12_mag - s12_mag_eq_ckt))
        s12_ph_diff = abs(np.array(s12_ph - s12_ph_eq_ckt))
        
        
        return {
                's11_mag_diff': s11_mag_diff, 
                's11_ph_diff': s11_ph_diff, 
                's12_mag_diff':s12_mag_diff, 
                's12_ph_diff': s12_ph_diff,
                
                's11_mag_asitic': s11_mag,
                's11_ph_asitic': s11_ph,
                's11_mag_eq_ckt': s11_mag_eq_ckt,
                's11_ph_eq_ckt': s11_ph_eq_ckt,
                
                's12_mag_asitic': s12_mag,
                's12_ph_asitic': s12_ph,
                's12_mag_eq_ckt': s12_mag_eq_ckt,
                's12_ph_eq_ckt': s12_ph_eq_ckt
                }


    def summary_tcoil_eq_ckt_vs_asitic(self):
        """
        tcoil:
            {
            'tcoil0':{
                ...
                's11_mag_diff':
                's11_ph_diff':
                's12_mag_diff':
                's12_ph_diff':
                'La_diff_perc':
                'Lb_diff_perc':
                'Ra_diff_perc':
                'Rb_diff_perc':
                ...
                },
            'tcoil1':{
                ...
                },
            ...  
             }
        """
               
        
        tcoil = {}
        for i in range(self.tcoil_num_new):
            network_asitic = rf.Network('{}/{}/S_tcoil{}_ab.s2p'.format(ASITIC_WORK_DIR,self.mode,i))
            network_asitic_a = rf.Network('{}/{}/S_tcoil{}_a.s2p'.format(ASITIC_WORK_DIR,self.mode,i))
            network_asitic_b = rf.Network('{}/{}/S_tcoil{}_b.s2p'.format(ASITIC_WORK_DIR,self.mode,i))
            
            network_eq_ckt = rf.Network('{}/{}/tcoil{}/tcoil{}.s3p'.format(HSPICE_WORK_DIR,self.mode,i,i))
            
            tcoil['tcoil{}'.format(i)] = {}
            
            s2p_diff = self.s2p_eq_ckt_vs_asitic(network_asitic, network_eq_ckt)
            
            tcoil['tcoil{}'.format(i)]['s11_mag_diff'] = s2p_diff['s11_mag_diff']
            tcoil['tcoil{}'.format(i)]['s11_ph_diff'] = s2p_diff['s11_ph_diff']
            tcoil['tcoil{}'.format(i)]['s12_mag_diff'] = s2p_diff['s12_mag_diff']
            tcoil['tcoil{}'.format(i)]['s12_ph_diff'] = s2p_diff['s12_ph_diff']
            
            tcoil['tcoil{}'.format(i)]['s11_mag_asitic'] = s2p_diff['s11_mag_asitic']
            tcoil['tcoil{}'.format(i)]['s11_ph_asitic'] = s2p_diff['s11_ph_asitic']
            tcoil['tcoil{}'.format(i)]['s12_mag_asitic'] = s2p_diff['s12_mag_asitic']
            tcoil['tcoil{}'.format(i)]['s12_ph_asitic'] = s2p_diff['s12_ph_asitic']
            
            tcoil['tcoil{}'.format(i)]['s11_mag_eq_ckt'] = s2p_diff['s11_mag_eq_ckt']
            tcoil['tcoil{}'.format(i)]['s11_ph_eq_ckt'] = s2p_diff['s11_ph_eq_ckt']
            tcoil['tcoil{}'.format(i)]['s12_mag_eq_ckt'] = s2p_diff['s12_mag_eq_ckt']
            tcoil['tcoil{}'.format(i)]['s12_ph_eq_ckt'] = s2p_diff['s12_ph_eq_ckt']
        
            rl_diff = self.rl_eq_ckt_vs_asitic(network_asitic_a, network_asitic_b, network_eq_ckt)    
        
            tcoil['tcoil{}'.format(i)]['La_asitic'] = rl_diff['La_asitic']
            tcoil['tcoil{}'.format(i)]['Lb_asitic'] = rl_diff['Lb_asitic']
            tcoil['tcoil{}'.format(i)]['Ra_asitic'] = rl_diff['Ra_asitic']
            tcoil['tcoil{}'.format(i)]['Rb_asitic'] = rl_diff['Rb_asitic']
            
            tcoil['tcoil{}'.format(i)]['La_eq_ckt'] = rl_diff['La_eq_ckt']
            tcoil['tcoil{}'.format(i)]['Lb_eq_ckt'] = rl_diff['Lb_eq_ckt']
            tcoil['tcoil{}'.format(i)]['Ra_eq_ckt'] = rl_diff['Ra_eq_ckt']
            tcoil['tcoil{}'.format(i)]['Rb_eq_ckt'] = rl_diff['Rb_eq_ckt']
            
            tcoil['tcoil{}'.format(i)]['La_diff_perc'] = rl_diff['La_diff_perc']
            tcoil['tcoil{}'.format(i)]['Lb_diff_perc'] = rl_diff['Lb_diff_perc']
            tcoil['tcoil{}'.format(i)]['Ra_diff_perc'] = rl_diff['Ra_diff_perc']
            tcoil['tcoil{}'.format(i)]['Rb_diff_perc'] = rl_diff['Rb_diff_perc']
    
        return tcoil


    def save2csv(self, tcoil_results, comparisons, plot=True):
        """
        
        Parameters
        ----------
        freq_design : float
            frequecy of interest
        accracy : float, optional
            filters out the data that deviate to ASITIC results that are larger
            than this. The default is 0.1. Set it to 1 if you want to save all the 
            data (unfiltered)
    
        Returns
        -------
        None.
    
        """

        
        # translate the frequency of interest into a list slicer index
        slicer = int((self.freq_design-self.freq_start)/self.freq_step+1)
    
        if self.mode == 'train':   
                
            L_list = []
            W_list = []
            S_list = []
            N_list = []
            tap_list = []
            
            La_asitic_list = []
            Lb_asitic_list = []
            Ra_asitic_list = []
            Rb_asitic_list = []
            
            La_eq_ckt_list = []
            Lb_eq_ckt_list = []
            Ra_eq_ckt_list = []
            Rb_eq_ckt_list = []
            
            s11_mag_asitic_list = []
            s11_ph_asitic_list = []
            s12_mag_asitic_list = []
            s12_ph_asitic_list = []
            
            s11_mag_eq_ckt_list = []
            s12_mag_eq_ckt_list = []
            s11_ph_eq_ckt_list = []
            s12_ph_eq_ckt_list = []
            
            s11_mag_diff_list = []
            s12_mag_diff_list = []
            s11_ph_diff_list = []
            s12_ph_diff_list = []
            
            La_diff_perc_list = []
            Ra_diff_perc_list = []
            Lb_diff_perc_list = []
            Rb_diff_perc_list = []
            
            k_eq_ckt_list = []
            
            defected_design = []
            
            
            for i in range(self.tcoil_num_new):
                La_asitic = comparisons[f'tcoil{i}']['La_asitic'] 
                Lb_asitic = comparisons[f'tcoil{i}']['Lb_asitic']
                
                Ra_asitic = comparisons[f'tcoil{i}']['Ra_asitic']
                Rb_asitic = comparisons[f'tcoil{i}']['Rb_asitic']
                
                La_eq_ckt = comparisons[f'tcoil{i}']['La_eq_ckt'] # this is basically tcoil_results['La']
                Lb_eq_ckt = comparisons[f'tcoil{i}']['Lb_eq_ckt'] # this is basically tcoil_results['Lb']
                
                Ra_eq_ckt = comparisons[f'tcoil{i}']['Ra_eq_ckt']
                Rb_eq_ckt = comparisons[f'tcoil{i}']['Rb_eq_ckt']
        
                s11_mag_asitic = comparisons[f'tcoil{i}']['s11_mag_asitic']
                s11_ph_asitic = comparisons[f'tcoil{i}']['s11_ph_asitic']
                
                s12_mag_asitic = comparisons[f'tcoil{i}']['s12_mag_asitic']
                s12_ph_asitic = comparisons[f'tcoil{i}']['s12_ph_asitic']
                
                s11_mag_eq_ckt = comparisons[f'tcoil{i}']['s11_mag_eq_ckt']
                s11_ph_eq_ckt = comparisons[f'tcoil{i}']['s11_ph_eq_ckt']
                
                s12_mag_eq_ckt = comparisons[f'tcoil{i}']['s12_mag_eq_ckt']
                s12_ph_eq_ckt = comparisons[f'tcoil{i}']['s12_ph_eq_ckt']
                
                s11_mag_diff = comparisons[f'tcoil{i}']['s11_mag_diff']
                s11_ph_diff = comparisons[f'tcoil{i}']['s11_ph_diff']
                
                s12_mag_diff = comparisons[f'tcoil{i}']['s12_mag_diff']
                s12_ph_diff = comparisons[f'tcoil{i}']['s12_ph_diff']
                
                La_diff_perc = comparisons[f'tcoil{i}']['La_diff_perc']
                Ra_diff_perc = comparisons[f'tcoil{i}']['Ra_diff_perc']
                
                Lb_diff_perc = comparisons[f'tcoil{i}']['Lb_diff_perc']
                Rb_diff_perc = comparisons[f'tcoil{i}']['Rb_diff_perc']
                
                k_eq_ckt = tcoil_results[f'tcoil{i}']['k']
                
                
                if (
                    max(La_asitic)>self.ind_max or
                    max(Lb_asitic)>self.ind_max 
                 ):
                
                    defected_design.append(i)
                    print(f'tcoil{i} is larger than the specified maximum inductance {self.ind_max} H.')
         
                # filter out defected tcoil designs that failed to be accurately modeled by the eq. ckt.
                elif (max(La_diff_perc) > self.accuracy or 
                    max(Lb_diff_perc) > self.accuracy or 
                    max(Ra_diff_perc) > self.accuracy or 
                    max(Rb_diff_perc) > self.accuracy or
                    any(i < 0 for i in s11_ph_eq_ckt) == True or
                    any(i < 0 for i in La_eq_ckt) == True or
                    any(i < 0 for i in Lb_eq_ckt) == True):
                    
                    defected_design.append(i)
                        
                    print(f'tcoil{i} is not accurately modeled by the eq.ckt.')                      
            
                else:
                    L_list.append(tcoil_results['tcoil{}'.format(i)]['L'])
                    W_list.append(tcoil_results['tcoil{}'.format(i)]['W'])
                    S_list.append(tcoil_results['tcoil{}'.format(i)]['S'])
                    N_list.append(tcoil_results['tcoil{}'.format(i)]['N'])
                    tap_list.append(tcoil_results['tcoil{}'.format(i)]['tap'])
                    
                    La_asitic_list.append(La_asitic) 
                    Lb_asitic_list.append(Lb_asitic)
                    
                    Ra_asitic_list.append(Ra_asitic)
                    Rb_asitic_list.append(Rb_asitic)
                    
                    La_eq_ckt_list.append(La_eq_ckt)
                    Lb_eq_ckt_list.append(Lb_eq_ckt)
                    
                    Ra_eq_ckt_list.append(Ra_eq_ckt)
                    Rb_eq_ckt_list.append(Rb_eq_ckt)
            
                    s11_mag_asitic_list.append(s11_mag_asitic)
                    s11_ph_asitic_list.append(s11_ph_asitic)
                    
                    s12_mag_asitic_list.append(s12_mag_asitic)
                    s12_ph_asitic_list.append(s12_ph_asitic)
                    
                    s11_mag_eq_ckt_list.append(s11_mag_eq_ckt)
                    s11_ph_eq_ckt_list.append(s11_ph_eq_ckt)
                    
                    s12_mag_eq_ckt_list.append(s12_mag_eq_ckt)
                    s12_ph_eq_ckt_list.append(s12_ph_eq_ckt)
                    
                    s11_mag_diff_list.append(s11_mag_diff)
                    s11_ph_diff_list.append(s11_ph_diff)
                    
                    s12_mag_diff_list.append(s12_mag_diff)
                    s12_ph_diff_list.append(s12_ph_diff)
                    
                    La_diff_perc_list.append(La_diff_perc)
                    Ra_diff_perc_list.append(Ra_diff_perc)
                    
                    Lb_diff_perc_list.append(Lb_diff_perc)
                    Rb_diff_perc_list.append(Rb_diff_perc)
                    
                    k_eq_ckt_list.append(k_eq_ckt)
                        
            # the L, R, k of all frequency points of all legal designs will be save in tcoil_LRk.csv
            tcoil_LRk = pd.DataFrame([
                                       np.array(L_list).tolist(),
                                       np.array(W_list).tolist(),
                                       np.array(S_list).tolist(),
                                       np.array(N_list).tolist(),
                                       np.array(tap_list).tolist(),
                                       np.array(La_eq_ckt_list).tolist(),
                                       np.array(Ra_eq_ckt_list).tolist(),
                                       np.array(Lb_eq_ckt_list).tolist(),
                                       np.array(Rb_eq_ckt_list).tolist(),
                                       np.array(k_eq_ckt_list).tolist()]).T
            
            tcoil_LRk.columns=['L', 'W', 'S', 'N', 'tap', 'La', 'Ra', 'Lb', 'Rb', 'k']
            tcoil_LRk.to_csv(f'{TCOIL_DATA_DIR}/train/tcoil_LRk.csv',mode='w',header=True)
                            
            print(f"There are {len(defected_design)} out of {len(tcoil_results)} tcoil designs failed to be accurately modeled by the eq.ckt.")


            with open(f'{PYTCOIL_DIR}/asitic/sim_setup_asitic.yaml','w') as yamlfile:
                self.sim_setups['tcoil_num_defected'] = len(defected_design)
                yaml.dump(self.sim_setups, yamlfile)   
            
            filename = f'{TCOIL_DATA_DIR}/train/tcoil_results_{self.freq_design/1e9}GHz_{self.tcoil_num_new-len(defected_design)}_{self.hspice_ser_opt}_{self.middle_branch}_{self.date}.csv'
            
            with open(filename, 'w') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['L', 'W', 'S', 'N', 'tap', 'La', 'Ra', 'Lb', 'Rb', 'k', 'Cbr'])
                for i in [x for x in range(self.tcoil_num_old, self.tcoil_num_new) if x not in defected_design]:
                    writer.writerow([tcoil_results['tcoil{}'.format(i)]['L'],
                                     tcoil_results['tcoil{}'.format(i)]['W'],
                                     tcoil_results['tcoil{}'.format(i)]['S'],
                                     tcoil_results['tcoil{}'.format(i)]['N'],
                                     tcoil_results['tcoil{}'.format(i)]['tap'],
                                     tcoil_results['tcoil{}'.format(i)]['La'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Ra'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Lb'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Rb'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['k'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Cbr']])
        
            if plot==True:
                s11_mag_diff_rms_freq = (np.sum(np.array(s11_mag_diff_list)**2, axis=0)/len(s11_mag_diff_list))**0.5 # rms value of all design w.r.t. freq_design
                #s11_mag_diff_rms_case = (np.sum(np.array(s11_mag_diff_list)**2, axis=1)/len(s11_mag_diff_list))**0.5 # rms value of one design across its freq_design
                
                s11_ph_diff_rms_freq = (np.sum(np.array(s11_ph_diff_list)**2, axis=0)/len(s11_ph_diff_list))**0.5
                #s11_ph_diff_rms_case = (np.sum(np.array(s11_ph_diff_list)**2, axis=1)/len(s11_ph_diff_list))**0.5
                
                s12_mag_diff_rms_freq = (np.sum(np.array(s12_mag_diff_list)**2, axis=0)/len(s12_mag_diff_list))**0.5
                #s12_mag_diff_rms_case = (np.sum(np.array(s12_mag_diff_list)**2, axis=1)/len(s12_mag_diff_list))**0.5
                
                s12_ph_diff_rms_freq = (np.sum(np.array(s12_ph_diff_list)**2, axis=0)/len(s12_ph_diff_list))**0.5
                #s12_ph_diff_rms_case = (np.sum(np.array(s12_ph_diff_list)**2, axis=1)/len(s12_ph_diff_list))**0.5
                
                plt.figure('$S_{11}$ mag rms w.r.t. Frequency')
                plt.plot(s11_mag_diff_rms_freq)
                plt.grid('True')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('$S_{11}$ mag RMSE')
                
                plt.figure('$S_{11}$ ph rms w.r.t. Frequency')
                plt.plot(s11_ph_diff_rms_freq)
                plt.grid('True')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('$S_{11}$ ph RMSE')
                
                plt.figure('$S_{12}$ mag rms w.r.t. Frequency')
                plt.plot(s12_mag_diff_rms_freq)
                plt.grid('True')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('$S_{12}$ mag RMSE')
                
                plt.figure('$S_{12}$ ph rms w.r.t. Frequency')
                plt.plot(s12_ph_diff_rms_freq)
                plt.grid('True')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('$S_{12}$ ph RMSE')
                
                #######################################################################
            
                plt.figure('La ASITIC vs. Eq. Ckt')
                plt.plot(La_asitic_list, La_eq_ckt_list, '^')
                plt.xlabel('imag($Z_{11}$/$\omega$) ASITIC (H)')
                plt.ylabel('imag($Z_{11}$/$\omega$) Eq. Ckt. (H)')
                plt.title('La ASITIC vs. Eq. Ckt')
                plt.grid(True)
                
                plt.figure('Lb ASITIC vs. Eq. Ckt')
                plt.plot(Lb_asitic_list, Lb_eq_ckt_list, '^')
                plt.xlabel('imag($Z_{22}$/$\omega$) ASITIC (H)')
                plt.ylabel('imag($Z_{22}$/$\omega$) Eq. Ckt. (H)')
                plt.title('Lb ASITIC vs. Eq. Ckt')
                plt.grid(True)
                
                plt.figure('Ra ASITIC vs. Eq. Ckt')
                plt.plot(Ra_asitic_list, Ra_eq_ckt_list, '^')
                plt.xlabel('real($Z_{11}$) ASITIC ($\Omega$)')
                plt.ylabel('real($Z_{11}$) Eq. Ckt. ($\Omega$)')
                plt.title('Ra ASITIC vs. Eq. Ckt')
                plt.grid(True)
                
                plt.figure('Rb ASITIC vs. Eq. Ckt')
                plt.plot(Rb_asitic_list, Rb_eq_ckt_list, '^')
                plt.xlabel('real($Z_{22}$) ASITIC ($\Omega$)')
                plt.ylabel('real($Z_{22}$) Eq. Ckt. ($\Omega$)')
                plt.title('Rb ASITIC vs. Eq. Ckt')
                plt.grid(True)
                
                #######################################################################
            
                plt.figure('mag(S11) ASITIC vs. Eq. Ckt.')
                plt.plot(s11_mag_asitic_list, s11_mag_eq_ckt_list, '^')
                plt.xlabel('mag(S11) ASITIC')
                plt.ylabel('mag(S11) Eq. Ckt.')
                plt.title('mag(S11) ASITIC vs. Eq. Ckt.')
                plt.grid(True)
                
                plt.figure('mag(S12) ASITIC vs. Eq. Ckt.')
                plt.plot(s12_mag_asitic_list, s12_mag_eq_ckt_list, '^')
                plt.xlabel('mag(S12) ASITIC')
                plt.ylabel('mag(S12) Eq. Ckt.')
                plt.title('mag(S12) ASITIC vs. Eq. Ckt.')
                plt.grid(True)
                
                plt.figure('ph(S11) ASITIC vs. Eq. Ckt.')
                plt.plot(s11_ph_asitic_list, s11_ph_eq_ckt_list, '^')
                plt.xlabel('ph(S11) ASITIC (deg)')
                plt.ylabel('ph(S11) Eq. Ckt. (deg)')
                plt.title('ph(S11) ASITIC vs. Eq. Ckt.')
                plt.grid(True)
                
                plt.figure('ph(S12) ASITIC vs. Eq. Ckt.')
                plt.plot(s12_ph_asitic_list, s12_ph_eq_ckt_list, '^')
                plt.xlabel('ph(S12) ASITIC (deg)')
                plt.ylabel('ph(S12) Eq. Ckt. (deg)')
                plt.title('ph(S12) ASITIC vs. Eq. Ckt.')
                plt.grid(True)
                
                #######################################################################
        
        
        else: #self.mode == 'test':
            filename = f'{TCOIL_DATA_DIR}/test/tcoil_dims.csv'
            with open(filename,'w') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['L', 'W', 'S', 'N', 'tap', 'La', 'Ra', 'Lb', 'Rb', 'k', 'Cbr'])
                for i in range(self.tcoil_num_old, self.tcoil_num_new):
                    writer.writerow([tcoil_results['tcoil{}'.format(i)]['L'],
                                     tcoil_results['tcoil{}'.format(i)]['W'],
                                     tcoil_results['tcoil{}'.format(i)]['S'],
                                     tcoil_results['tcoil{}'.format(i)]['N'],
                                     tcoil_results['tcoil{}'.format(i)]['tap'],
                                     tcoil_results['tcoil{}'.format(i)]['La'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Ra'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Lb'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Rb'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['k'][slicer],
                                     tcoil_results['tcoil{}'.format(i)]['Cbr']])
                
