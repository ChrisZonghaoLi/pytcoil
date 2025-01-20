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
        self.freq_stop = float(self.sim_setups['freq_start'])
        self.freq_step = float(self.sim_setups['freq_step'])
        self.freq_design = float(self.sim_setups['freq_design'])
        self.ind_max = float(self.sim_setups['ind_max'])
        self.accuracy = float(self.sim_setups['accuracy'])
        self.middle_tap = int(self.sim_setups['middle_tap'])
        
        self.date = datetime.today().strftime('%Y-%m-%d')
        
    def summary_tcoil(self):
        tcoil = {}
        tcoil_dims=pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_dims.csv', usecols = ['L', 'W', 'S', 'Nin', 'Nout'])     
        for i in range(self.tcoil_num_new):
            print(i)
            tcoil['tcoil{}'.format(i)] = {}
            tcoil['tcoil{}'.format(i)]['index'] = i
            network = rf.Network(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s3p')
            s_params = network.s
            z_params = network.z
            
            # tcoil s-param
            s11_list = [s_params[i][0][0] for i in range(len(s_params))]
            s12_list = [s_params[i][0][1] for i in range(len(s_params))]
            s13_list = [s_params[i][0][2] for i in range(len(s_params))]
            s21_list = s12_list
            s22_list = [s_params[i][1][1] for i in range(len(s_params))]
            s23_list = [s_params[i][1][2] for i in range(len(s_params))]
            s31_list = s13_list
            s32_list = s23_list
            s33_list = [s_params[i][2][2] for i in range(len(s_params))]
            
            z11_list = [z_params[i][0][0] for i in range(len(z_params))]
            z12_list = [z_params[i][0][1] for i in range(len(z_params))]
            z13_list = [z_params[i][0][2] for i in range(len(z_params))]
            z21_list = z12_list
            z22_list = [z_params[i][1][1] for i in range(len(z_params))]
            z23_list = [z_params[i][1][2] for i in range(len(z_params))]
            z31_list = z13_list
            z32_list = z23_list
            z33_list = [z_params[i][2][2] for i in range(len(z_params))]
                
            tcoil['tcoil{}'.format(i)]['s11'] = s11_list
            tcoil['tcoil{}'.format(i)]['s12'] = s12_list
            tcoil['tcoil{}'.format(i)]['s13'] = s13_list
            tcoil['tcoil{}'.format(i)]['s21'] = s21_list
            tcoil['tcoil{}'.format(i)]['s22'] = s22_list
            tcoil['tcoil{}'.format(i)]['s23'] = s23_list
            tcoil['tcoil{}'.format(i)]['s31'] = s31_list
            tcoil['tcoil{}'.format(i)]['s32'] = s32_list
            tcoil['tcoil{}'.format(i)]['s33'] = s33_list
            
            tcoil['tcoil{}'.format(i)]['z11'] = z11_list
            tcoil['tcoil{}'.format(i)]['z12'] = z12_list
            tcoil['tcoil{}'.format(i)]['z13'] = z13_list
            tcoil['tcoil{}'.format(i)]['z21'] = z21_list
            tcoil['tcoil{}'.format(i)]['z22'] = z22_list
            tcoil['tcoil{}'.format(i)]['z23'] = z23_list
            tcoil['tcoil{}'.format(i)]['z31'] = z31_list
            tcoil['tcoil{}'.format(i)]['z32'] = z32_list
            tcoil['tcoil{}'.format(i)]['z33'] = z33_list
            
            # tcoil dimensions
            tcoil['tcoil{}'.format(i)]['L'] = tcoil_dims['L'][i]
            tcoil['tcoil{}'.format(i)]['W'] = tcoil_dims['W'][i]
            tcoil['tcoil{}'.format(i)]['S'] = tcoil_dims['S'][i]
            tcoil['tcoil{}'.format(i)]['Nin'] = tcoil_dims['Nin'][i]
            tcoil['tcoil{}'.format(i)]['Nout'] = tcoil_dims['Nout'][i]
            
            
            with open(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.scs','r') as tcoil_netlist:
                    lines = []
                    for line in tcoil_netlist.readlines():
                        lines.append(str.split(line))
            
            ## La, Lb, Ra, Rb, k ##
            tcoil['tcoil{}'.format(i)]['La'] = t_network_eq_ckt(network, self.middle_tap)['La']
            tcoil['tcoil{}'.format(i)]['Lb'] = t_network_eq_ckt(network, self.middle_tap)['Lb']
            tcoil['tcoil{}'.format(i)]['Ra'] = t_network_eq_ckt(network, self.middle_tap)['Ra']
            tcoil['tcoil{}'.format(i)]['Rb'] = t_network_eq_ckt(network, self.middle_tap)['Rb']
            tcoil['tcoil{}'.format(i)]['Qa'] = t_network_eq_ckt(network, self.middle_tap)['Qa']
            tcoil['tcoil{}'.format(i)]['Qb'] = t_network_eq_ckt(network, self.middle_tap)['Qb']
            # for GF22 k is negative value
            tcoil['tcoil{}'.format(i)]['k'] = -t_network_eq_ckt(network, self.middle_tap)['k']
            
            # Las and Lbs are different from La and Lb: they are fixed values and are the
            # inductance in the equivalent circuit; see paper "Frequency-Independent 
            # Asymmetric Double-pi Equivalent Circuit for On-Chip Spiral Inductors: 
            # Physicas-Based Modeilling and Parameter Extraction" Fig. 7.
            # Below follows the parameter name convention except Co = Cbr
            Ls1 = float(rx.findall(lines[10][0])[1]) * float(rx.findall(lines[11][0])[1]) # in H, prime series ind; l1_sect1
            Ls2 = float(rx.findall(lines[12][0])[1]) * float(rx.findall(lines[13][0])[1]) # in H, second seires ind; l2_sect1
            Rs1 = float(rx.findall(lines[14][0])[1]) # prime series res; r1_sect1
            Rs2 = float(rx.findall(lines[15][0])[1]) # second series res; r2_sect1
            
            Ls11 = float(rx.findall(lines[16][0])[1]) * float(lines[17][0].split('=')[1].split('*')[0]) # prime parallel ind skin effect; l1_sect2
            Ls22 = float(rx.findall(lines[18][0])[1]) * float(lines[19][0].split('=')[1].split('*')[0]) # second parallel ind skin effect; l2_sect2
            Rs11 = float(rx.findall(lines[20][0])[1]) # prime parallel res skin effect; r1_sect2
            Rs22 = float(rx.findall(lines[21][0])[1]) # second parallel res skin effect; r2_sect2
            
            tcoil['tcoil{}'.format(i)]['Ls1'] = Ls1
            tcoil['tcoil{}'.format(i)]['Rs1'] = Rs1
            tcoil['tcoil{}'.format(i)]['Ls2'] = Ls2
            tcoil['tcoil{}'.format(i)]['Rs2'] = Rs2
            tcoil['tcoil{}'.format(i)]['Ls11'] = Ls11
            tcoil['tcoil{}'.format(i)]['Rs11'] = Rs11
            tcoil['tcoil{}'.format(i)]['Ls22'] = Ls22
            tcoil['tcoil{}'.format(i)]['Rs22'] = Rs22
            
            xsc1 = float(rx.findall(lines[25][0])[2])
            sc1 = float(rx.findall(lines[26][0])[2])
            
            xsc2 = xsc1 + float(rx.findall(lines[22][0])[1])**2
            sc2 = 1/(xsc2)**0.5
            
            xsc3 = xsc2 + float(rx.findall(lines[23][0])[1])**2
            sc3 = 1/(xsc3)**0.5
            
            xsc4 = xsc3 + float(rx.findall(lines[24][0])[1])**2
            sc4 = 1/(xsc4)**0.5

            k12 = float(rx.findall(lines[22][0])[1]) * sc2 # coupling factor between Ls1 and Ls2; k12_ks
            k111 = float(rx.findall(lines[23][0])[1]) * sc3 # coupling factor between Ls1 and Ls11; k13_ks
            k122 = float(rx.findall(lines[24][0])[1]) * sc4 # coupling factor between Ls1 and Ls22; k14_ks
            k211 = (float(rx.findall(lines[22][0])[1]) * float(rx.findall(lines[23][0])[1]) + 
                    float(rx.findall(lines[22][0])[1])) * sc2 * sc3 # coupling factor between Ls2 and Ls11; k23_ks
            k222 = (float(rx.findall(lines[22][0])[1]) * float(rx.findall(lines[24][0])[1]) + 
                    float(rx.findall(lines[23][0])[1])) * sc2 * sc4 # coupling factor between Ls2 and Ls22; k24_ks
            k1122 = (float(rx.findall(lines[23][0])[1]) * float(rx.findall(lines[24][0])[1]) + float(rx.findall(lines[22][0])[1]) * float(rx.findall(lines[23][0])[1]) +
                    float(rx.findall(lines[22][0])[1])) * sc2 * sc4 # coupling factor between Ls11 and Ls22; k34_ks
            
            Rc = float(rx.findall(lines[39][0])[1]) # rc
            
            tcoil['tcoil{}'.format(i)]['k12'] = k12
            tcoil['tcoil{}'.format(i)]['k111'] = k111
            tcoil['tcoil{}'.format(i)]['k122'] = k122
            tcoil['tcoil{}'.format(i)]['k211'] = k211
            tcoil['tcoil{}'.format(i)]['k222'] = k222
            tcoil['tcoil{}'.format(i)]['k1122'] = k1122
            tcoil['tcoil{}'.format(i)]['Rc'] = Rc            
            
            Cp1 = float(rx.findall(lines[42][0])[1]) * float(rx.findall(lines[43][0])[1]) # in F; c13
            Cp2 = float(rx.findall(lines[44][0])[1]) * float(rx.findall(lines[45][0])[1]) # in F; c23
            Cbr = float(rx.findall(lines[40][0])[1]) * float(rx.findall(lines[41][0])[1]) # in F; c12
            Cox1 = float(rx.findall(lines[46][0])[1]) * float(rx.findall(lines[55][0])[1]) # in F, input Cox; c_1_sub
            Cox2 = float(rx.findall(lines[49][0])[1]) * float(rx.findall(lines[58][0])[1]) # in F, endtap Cox; c_2_sub
            Cox3 = float(rx.findall(lines[52][0])[1]) * float(rx.findall(lines[61][0])[1]) # in F, centertap Cox; c_3_sub
            Csub1 = float(rx.findall(lines[48][0])[1]) * float(rx.findall(lines[57][0])[1]) # in F, input Csub; cs_1_sub
            Csub2 = float(rx.findall(lines[51][0])[1]) * float(rx.findall(lines[60][0])[1]) # in F, endtap Csub; cs_2_sub
            Csub3 = float(rx.findall(lines[54][0])[1]) * float(rx.findall(lines[63][0])[1]) # in F, centertap Csub; cs_3_sub
            Rsub1 = float(rx.findall(lines[47][0])[1]) * float(rx.findall(lines[56][0])[1]) # in F, input Rsub; re_1_sub
            Rsub2 = float(rx.findall(lines[50][0])[1]) * float(rx.findall(lines[59][0])[1]) # in F, endtap Rsub; re_2_sub
            Rsub3 = float(rx.findall(lines[53][0])[1]) * float(rx.findall(lines[62][0])[1]) # in F, centertap Rsub; re_3_sub
            
            Rx12 = float(rx.findall(lines[64][0])[1]) * float(rx.findall(lines[66][0])[1]) # input/endtap substrate cross res; rx_1_2_sub
            Cx12 = float(rx.findall(lines[65][0])[1]) * float(rx.findall(lines[67][0])[1]) # input/endtap substrate cross cap; cx_1_2_sub
            Rx13 = float(rx.findall(lines[68][0])[1]) * float(rx.findall(lines[70][0])[1]) # input/centertap substrate cross res; rx_1_3_sub
            Cx13 = float(rx.findall(lines[69][0])[1]) * float(rx.findall(lines[71][0])[1]) # input/centertap substrate cross cap; cx_1_3_sub
            Rx23 = float(rx.findall(lines[72][0])[1]) * float(rx.findall(lines[74][0])[1]) # input/centertap substrate cross res; rx_2_3_sub
            Cx23 = float(rx.findall(lines[73][0])[1]) * float(rx.findall(lines[75][0])[1]) # input/centertap substrate cross cap; cx_2_3_sub
            
            tcoil['tcoil{}'.format(i)]['Cp1'] = Cp1
            tcoil['tcoil{}'.format(i)]['Cp2'] = Cp2
            tcoil['tcoil{}'.format(i)]['Rx12'] = Rx12
            tcoil['tcoil{}'.format(i)]['Cx12'] = Cx12
            tcoil['tcoil{}'.format(i)]['Rx13'] = Rx13
            tcoil['tcoil{}'.format(i)]['Cx13'] = Cx13
            tcoil['tcoil{}'.format(i)]['Rx23'] = Rx23
            tcoil['tcoil{}'.format(i)]['Cx23'] = Cx23
            
            tcoil['tcoil{}'.format(i)]['Cbr'] = Cbr
            tcoil['tcoil{}'.format(i)]['Cox_in'] = Cox1
            tcoil['tcoil{}'.format(i)]['Cox_mid'] = Cox3
            tcoil['tcoil{}'.format(i)]['Cox_out'] = Cox2
            tcoil['tcoil{}'.format(i)]['Csub_in'] = Csub1
            tcoil['tcoil{}'.format(i)]['Csub_mid'] = Csub3
            tcoil['tcoil{}'.format(i)]['Csub_out'] = Csub2
            tcoil['tcoil{}'.format(i)]['Rsub_in'] = Rsub1
            tcoil['tcoil{}'.format(i)]['Rsub_mid'] = Rsub3
            tcoil['tcoil{}'.format(i)]['Rsub_out'] = Rsub2
        
            # Approximate resonant frequency SRF
            # alpha = Cp2/Cp1
            # chi = 1/alpha
            # fr = 1/(2*math.pi*np.sqrt( 0.25*Ls1*Cp1*(1+chi)**2 + 2*Cbr*Ls1*(1+chi) ))
            # tcoil['tcoil{}'.format(i)]['fr'] = fr
            
         
        return tcoil
        
    
    def save2csv(self, tcoil_results, plot=True, filter = True):    
    
        index_list = []    
    
        La_list = []
        La_eq_ckt_list = []
        Lb_list = []
        Lb_eq_ckt_list = []
        Ra_list = []
        Ra_eq_ckt_list = []
        Rb_list = []
        Rb_eq_ckt_list = []
        Qa_list = []
        Qa_eq_ckt_list = []
        Qb_list = []
        Qb_eq_ckt_list = []
        k_list = []
        k_eq_ckt_list = []
        
        L_list = []
        W_list = []
        S_list = []
        Nin_list = []
        Nout_list = []
        
        # These are for equivalent circuits####
        Ls1_list = []
        Rs1_list = []
        Ls11_list = []
        Rs11_list = []
        Ls2_list = []
        Rs2_list = []
        Ls22_list = []
        Rs22_list = []
        
        k12_list = []
        k111_list = []
        k122_list = []
        k211_list = []
        k222_list = []
        k1122_list = []
        
        Rc_list = []
        
        Cp1_list = []
        Cp2_list = []
        Cbr_list = []
        Cox1_list = []
        Cox2_list = []
        Cox3_list = []
        Csub1_list = []
        Csub2_list = []
        Csub3_list = []
        Rsub1_list = []
        Rsub2_list = []
        Rsub3_list = []
        
        Rx12_list = []
        Cx12_list = []
        Rx13_list = []
        Cx13_list = []
        Rx23_list = []
        Cx23_list = []
        
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
        
        z11_list = []
        z12_list = []
        z13_list = []
        z21_list = []
        z22_list = []
        z23_list = []
        z31_list = []
        z32_list = []
        z33_list = []
        
        defected_design = []
        
        # slicer will translate the freq you want to the index of the list
        slicer = int((self.freq_design-self.freq_start)/self.freq_step+1)
        
        for i in range(self.tcoil_num_new):
            network_eq_ckt = rf.Network(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}_mod.s3p')
        
            index = tcoil_results['tcoil{}'.format(i)]['index'] 
        
            La = tcoil_results['tcoil{}'.format(i)]['La']
            Lb = tcoil_results['tcoil{}'.format(i)]['Lb']
            Ra = tcoil_results['tcoil{}'.format(i)]['Ra']
            Rb = tcoil_results['tcoil{}'.format(i)]['Rb']
            Qa = tcoil_results['tcoil{}'.format(i)]['Qa']
            Qb = tcoil_results['tcoil{}'.format(i)]['Qb']
            k = tcoil_results['tcoil{}'.format(i)]['k']
        
            La_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['La']
            Lb_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Lb']
            Ra_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Ra']
            Rb_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Rb']
            Qa_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Qa']
            Qb_eq_ckt = t_network_eq_ckt(network_eq_ckt, self.middle_tap)['Qb']
            # for GF22 k is negative value
            k_eq_ckt = -t_network_eq_ckt(network_eq_ckt, self.middle_tap)['k']
            
            if filter == True:
                if (
                    La[slicer]>self.ind_max or
                    Lb[slicer]>self.ind_max 
                     ):
                    
                    defected_design.append(i)
                    print(f'tcoil{i} is larger than the specified maximum inductance {self.ind_max} H.')
            
                elif (
                    abs(La[slicer]-La_eq_ckt[slicer])/La[slicer]>self.accuracy or
                    abs(Lb[slicer]-Lb_eq_ckt[slicer])/Lb[slicer]>self.accuracy or
                    abs(Ra[slicer]-Ra_eq_ckt[slicer])/Ra[slicer]>self.accuracy or
                    abs(Rb[slicer]-Rb_eq_ckt[slicer])/Rb[slicer]>self.accuracy or
                    abs(k[slicer]-k_eq_ckt[slicer])/k[slicer]>self.accuracy
                        ):
                    
                    defected_design.append(i)
                    print(f'tcoil{i} is not accurately modeled by ModelGen within the given {self.accuracy} error tolerance.')
                    
                else:
                    index_list.append(tcoil_results['tcoil{}'.format(i)]['index'])
                    
                    L_list.append(tcoil_results['tcoil{}'.format(i)]['L'])
                    W_list.append(tcoil_results['tcoil{}'.format(i)]['W'])
                    S_list.append(tcoil_results['tcoil{}'.format(i)]['S'])
                    Nin_list.append(tcoil_results['tcoil{}'.format(i)]['Nin'])
                    Nout_list.append(tcoil_results['tcoil{}'.format(i)]['Nout'])
                    
                    La_list.append(La)
                    La_eq_ckt_list.append(La_eq_ckt)
                    Lb_list.append(Lb)
                    Lb_eq_ckt_list.append(Lb_eq_ckt)
                    Ra_list.append(Ra)
                    Ra_eq_ckt_list.append(Ra_eq_ckt)
                    Rb_list.append(Rb)
                    Rb_eq_ckt_list.append(Rb_eq_ckt)
                    Qa_list.append(Qa)
                    Qa_eq_ckt_list.append(Qa_eq_ckt)
                    Qb_list.append(Qb)
                    Qb_eq_ckt_list.append(Qb_eq_ckt)
                    k_list.append(k)
                    k_eq_ckt_list.append(k_eq_ckt)
                    
                    # equivalent circuits
                    Ls1_list.append(tcoil_results['tcoil{}'.format(i)]['Ls1'])
                    Rs1_list.append(tcoil_results['tcoil{}'.format(i)]['Rs1'])
                    Ls11_list.append(tcoil_results['tcoil{}'.format(i)]['Ls11'])
                    Rs11_list.append(tcoil_results['tcoil{}'.format(i)]['Rs11'])
                    
                    Ls2_list.append(tcoil_results['tcoil{}'.format(i)]['Ls2'])
                    Rs2_list.append(tcoil_results['tcoil{}'.format(i)]['Rs2'])
                    Ls22_list.append(tcoil_results['tcoil{}'.format(i)]['Ls22'])
                    Rs22_list.append(tcoil_results['tcoil{}'.format(i)]['Rs22'])
                    
                    k12_list.append(tcoil_results['tcoil{}'.format(i)]['k12'])
                    k111_list.append(tcoil_results['tcoil{}'.format(i)]['k111'])
                    k122_list.append(tcoil_results['tcoil{}'.format(i)]['k122'])
                    k211_list.append(tcoil_results['tcoil{}'.format(i)]['k211'])
                    k222_list.append(tcoil_results['tcoil{}'.format(i)]['k222'])
                    k1122_list.append(tcoil_results['tcoil{}'.format(i)]['k1122'])
                    
                    Rc_list.append(tcoil_results['tcoil{}'.format(i)]['Rc'])
                    
                    Cp1_list.append(tcoil_results['tcoil{}'.format(i)]['Cp1'])
                    Cp2_list.append(tcoil_results['tcoil{}'.format(i)]['Cp2'])
                    Cbr_list.append(tcoil_results['tcoil{}'.format(i)]['Cbr'])
                    Cox1_list.append(tcoil_results['tcoil{}'.format(i)]['Cox_in'])
                    Cox3_list.append(tcoil_results['tcoil{}'.format(i)]['Cox_mid'])
                    Cox2_list.append(tcoil_results['tcoil{}'.format(i)]['Cox_out'])
                    Csub1_list.append(tcoil_results['tcoil{}'.format(i)]['Csub_in'])
                    Csub3_list.append(tcoil_results['tcoil{}'.format(i)]['Csub_mid'])
                    Csub2_list.append(tcoil_results['tcoil{}'.format(i)]['Csub_out'])
                    Rsub1_list.append(tcoil_results['tcoil{}'.format(i)]['Rsub_in'])
                    Rsub3_list.append(tcoil_results['tcoil{}'.format(i)]['Rsub_mid'])   
                    Rsub2_list.append(tcoil_results['tcoil{}'.format(i)]['Rsub_out'])
                    
                    Rx12_list.append(tcoil_results['tcoil{}'.format(i)]['Rx12'])
                    Cx12_list.append(tcoil_results['tcoil{}'.format(i)]['Cx12'])
                    Rx13_list.append(tcoil_results['tcoil{}'.format(i)]['Rx13'])
                    Cx13_list.append(tcoil_results['tcoil{}'.format(i)]['Cx13'])
                    Rx23_list.append(tcoil_results['tcoil{}'.format(i)]['Rx23'])
                    Cx23_list.append(tcoil_results['tcoil{}'.format(i)]['Cx23'])
    
                    s11_list.append(tcoil_results['tcoil{}'.format(i)]['s11'])
                    s12_list.append(tcoil_results['tcoil{}'.format(i)]['s12'])
                    s13_list.append(tcoil_results['tcoil{}'.format(i)]['s13'])
                    s22_list.append(tcoil_results['tcoil{}'.format(i)]['s22'])
                    s23_list.append(tcoil_results['tcoil{}'.format(i)]['s23'])
                    s33_list.append(tcoil_results['tcoil{}'.format(i)]['s33'])
                    
                    z11_list.append(tcoil_results['tcoil{}'.format(i)]['z11'])
                    z12_list.append(tcoil_results['tcoil{}'.format(i)]['z12'])
                    z13_list.append(tcoil_results['tcoil{}'.format(i)]['z13'])
                    z22_list.append(tcoil_results['tcoil{}'.format(i)]['z22'])
                    z23_list.append(tcoil_results['tcoil{}'.format(i)]['z23'])
                    z33_list.append(tcoil_results['tcoil{}'.format(i)]['z33'])

            else:
                index_list.append(tcoil_results['tcoil{}'.format(i)]['index'])
                
                L_list.append(tcoil_results['tcoil{}'.format(i)]['L'])
                W_list.append(tcoil_results['tcoil{}'.format(i)]['W'])
                S_list.append(tcoil_results['tcoil{}'.format(i)]['S'])
                Nin_list.append(tcoil_results['tcoil{}'.format(i)]['Nin'])
                Nout_list.append(tcoil_results['tcoil{}'.format(i)]['Nout'])
                
                La_list.append(La)
                La_eq_ckt_list.append(La_eq_ckt)
                Lb_list.append(Lb)
                Lb_eq_ckt_list.append(Lb_eq_ckt)
                Ra_list.append(Ra)
                Ra_eq_ckt_list.append(Ra_eq_ckt)
                Rb_list.append(Rb)
                Rb_eq_ckt_list.append(Rb_eq_ckt)
                Qa_list.append(Qa)
                Qa_eq_ckt_list.append(Qa_eq_ckt)
                Qb_list.append(Qb)
                Qb_eq_ckt_list.append(Qb_eq_ckt)
                k_list.append(k)
                k_eq_ckt_list.append(k_eq_ckt)
                
                # equivalent circuits
                Ls1_list.append(tcoil_results['tcoil{}'.format(i)]['Ls1'])
                Rs1_list.append(tcoil_results['tcoil{}'.format(i)]['Rs1'])
                Ls11_list.append(tcoil_results['tcoil{}'.format(i)]['Ls11'])
                Rs11_list.append(tcoil_results['tcoil{}'.format(i)]['Rs11'])
                
                Ls2_list.append(tcoil_results['tcoil{}'.format(i)]['Ls2'])
                Rs2_list.append(tcoil_results['tcoil{}'.format(i)]['Rs2'])
                Ls22_list.append(tcoil_results['tcoil{}'.format(i)]['Ls22'])
                Rs22_list.append(tcoil_results['tcoil{}'.format(i)]['Rs22'])
                
                k12_list.append(tcoil_results['tcoil{}'.format(i)]['k12'])
                k111_list.append(tcoil_results['tcoil{}'.format(i)]['k111'])
                k122_list.append(tcoil_results['tcoil{}'.format(i)]['k122'])
                k211_list.append(tcoil_results['tcoil{}'.format(i)]['k211'])
                k222_list.append(tcoil_results['tcoil{}'.format(i)]['k222'])
                k1122_list.append(tcoil_results['tcoil{}'.format(i)]['k1122'])
                
                Rc_list.append(tcoil_results['tcoil{}'.format(i)]['Rc'])
                
                Cp1_list.append(tcoil_results['tcoil{}'.format(i)]['Cp1'])
                Cp2_list.append(tcoil_results['tcoil{}'.format(i)]['Cp2'])
                Cbr_list.append(tcoil_results['tcoil{}'.format(i)]['Cbr'])
                Cox1_list.append(tcoil_results['tcoil{}'.format(i)]['Cox_in'])
                Cox3_list.append(tcoil_results['tcoil{}'.format(i)]['Cox_mid'])
                Cox2_list.append(tcoil_results['tcoil{}'.format(i)]['Cox_out'])
                Csub1_list.append(tcoil_results['tcoil{}'.format(i)]['Csub_in'])
                Csub3_list.append(tcoil_results['tcoil{}'.format(i)]['Csub_mid'])
                Csub2_list.append(tcoil_results['tcoil{}'.format(i)]['Csub_out'])
                Rsub1_list.append(tcoil_results['tcoil{}'.format(i)]['Rsub_in'])
                Rsub3_list.append(tcoil_results['tcoil{}'.format(i)]['Rsub_mid'])   
                Rsub2_list.append(tcoil_results['tcoil{}'.format(i)]['Rsub_out'])
                
                Rx12_list.append(tcoil_results['tcoil{}'.format(i)]['Rx12'])
                Cx12_list.append(tcoil_results['tcoil{}'.format(i)]['Cx12'])
                Rx13_list.append(tcoil_results['tcoil{}'.format(i)]['Rx13'])
                Cx13_list.append(tcoil_results['tcoil{}'.format(i)]['Cx13'])
                Rx23_list.append(tcoil_results['tcoil{}'.format(i)]['Rx23'])
                Cx23_list.append(tcoil_results['tcoil{}'.format(i)]['Cx23'])

                s11_list.append(tcoil_results['tcoil{}'.format(i)]['s11'])
                s12_list.append(tcoil_results['tcoil{}'.format(i)]['s12'])
                s13_list.append(tcoil_results['tcoil{}'.format(i)]['s13'])
                s22_list.append(tcoil_results['tcoil{}'.format(i)]['s22'])
                s23_list.append(tcoil_results['tcoil{}'.format(i)]['s23'])
                s33_list.append(tcoil_results['tcoil{}'.format(i)]['s33'])
                
                z11_list.append(tcoil_results['tcoil{}'.format(i)]['z11'])
                z12_list.append(tcoil_results['tcoil{}'.format(i)]['z12'])
                z13_list.append(tcoil_results['tcoil{}'.format(i)]['z13'])
                z22_list.append(tcoil_results['tcoil{}'.format(i)]['z22'])
                z23_list.append(tcoil_results['tcoil{}'.format(i)]['z23'])
                z33_list.append(tcoil_results['tcoil{}'.format(i)]['z33'])

        ######################################################################
        
        with open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','w') as yamlfile:
                self.sim_setups['tcoil_num_defected'] = len(defected_design)
                yaml.dump(self.sim_setups, yamlfile) 
                
        
        
       ######################################################################
        # z parameters will be save in tcoil_Z.csv
        tcoil_S = pd.DataFrame([np.array(index_list).tolist(),
                                   np.array(L_list).tolist(),
                                   np.array(W_list).tolist(),
                                   np.array(S_list).tolist(),
                                   np.array(Nin_list).tolist(),
                                   np.array(Nout_list).tolist(),
                                   np.array(z11_list).tolist(),
                                   np.array(z12_list).tolist(),
                                   np.array(z13_list).tolist(),
                                   np.array(z22_list).tolist(),
                                   np.array(z23_list).tolist(),
                                   np.array(z33_list).tolist()]).T
        
        tcoil_S.columns=['index', 'L', 'W', 'S', 'Nin', 'Nout', 'z11', 'z12', 'z13', 'z22', 'z23', 'z33']
        tcoil_S.to_csv(f'{TCOIL_DATA_DIR}/train/tcoil_Z_{self.freq_start/1e9}-{self.freq_stop/1e9}GHz_{self.ind_max}_{self.tcoil_num_new-len(defected_design)}_{self.date}.csv',mode='w',header=True)
        
        
        ######################################################################
        # S parameters will be save in tcoil_S.csv
        tcoil_S = pd.DataFrame([np.array(index_list).tolist(),
                                   np.array(L_list).tolist(),
                                   np.array(W_list).tolist(),
                                   np.array(S_list).tolist(),
                                   np.array(Nin_list).tolist(),
                                   np.array(Nout_list).tolist(),
                                   np.array(s11_list).tolist(),
                                   np.array(s12_list).tolist(),
                                   np.array(s13_list).tolist(),
                                   np.array(s22_list).tolist(),
                                   np.array(s23_list).tolist(),
                                   np.array(s33_list).tolist()]).T
        
        tcoil_S.columns=['index', 'L', 'W', 'S', 'Nin', 'Nout', 's11', 's12', 's13', 's22', 's23', 's33']
        tcoil_S.to_csv(f'{TCOIL_DATA_DIR}/train/tcoil_S_{self.freq_start/1e9}-{self.freq_stop/1e9}GHz_{self.ind_max}_{self.tcoil_num_new-len(defected_design)}_{self.date}.csv',mode='w',header=True)
        
        ######################################################################
        # the L, R, Q, k of all frequency points of all legal designs will be save in tcoil_LRQk.csv
        tcoil_LRQk = pd.DataFrame([np.array(index_list).tolist(),
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
                                   np.array(k_list).tolist()]).T
        
        tcoil_LRQk.columns=['index', 'L', 'W', 'S', 'Nin', 'Nout', 'La', 'Ra', 'Qa', 'Lb', 'Rb', 'Qb', 'k']
        tcoil_LRQk.to_csv(f'{TCOIL_DATA_DIR}/train/tcoil_LRQk_{self.freq_start/1e9}-{self.freq_stop/1e9}GHz_{self.ind_max}_{self.tcoil_num_new-len(defected_design)}_{self.date}.csv',mode='w',header=True)
        
        ######################################################################
        # record all equivalent circuit components
        tcoil_eq_ckt = pd.DataFrame([np.array(index_list).tolist(),
                                   np.array(L_list).tolist(),
                                   np.array(W_list).tolist(),
                                   np.array(S_list).tolist(),
                                   np.array(Nin_list).tolist(),
                                   np.array(Nout_list).tolist(),
                                   
                                   np.array(Ls1_list).tolist(),
                                   np.array(Rs1_list).tolist(),
                                   np.array(Ls11_list).tolist(),
                                   np.array(Rs11_list).tolist(),
                                   
                                   np.array(Ls2_list).tolist(),
                                   np.array(Rs2_list).tolist(),
                                   np.array(Ls22_list).tolist(),
                                   np.array(Rs22_list).tolist(),
                                   
                                   np.array(k12_list).tolist(),
                                   np.array(k111_list).tolist(),
                                   np.array(k122_list).tolist(),
                                   np.array(k211_list).tolist(),
                                   np.array(k222_list).tolist(),
                                   np.array(k1122_list).tolist(),
                                   
                                   np.array(Rc_list).tolist(),
                                   
                                   np.array(Cp1_list).tolist(),
                                   np.array(Cp2_list).tolist(),
                                   np.array(Cbr_list).tolist(),
                                   np.array(Cox1_list).tolist(),
                                   np.array(Cox3_list).tolist(),
                                   np.array(Cox2_list).tolist(),
                                   np.array(Csub1_list).tolist(),                                   
                                   np.array(Csub3_list).tolist(),
                                   np.array(Csub2_list).tolist(),
                                   np.array(Rsub1_list).tolist(),
                                   np.array(Rsub3_list).tolist(),
                                   np.array(Rsub2_list).tolist(),
                                   
                                   np.array(Rx12_list).tolist(),
                                   np.array(Cx12_list).tolist(),
                                   np.array(Rx13_list).tolist(),
                                   np.array(Cx13_list).tolist(),
                                   np.array(Rx23_list).tolist(),
                                   np.array(Cx23_list).tolist()
                                   ]).T
        
        tcoil_eq_ckt.columns=['index', 'L', 'W', 'S', 'Nin', 'Nout', 
                            'Ls1', 'Rs1', 'Ls11', 'Rs11', 'Ls2', 'Rs2', 'Ls22', 'Rs22',
                            'k12', 'k111', 'k122', 'k211', 'k222', 'k1122',
                            'Rc',
                            'Cp1', 'Cp2', 'Cbr', 
                            'Cox_in', 'Cox_mid', 'Cox_out',
                            'Csub_in', 'Csub_mid', 'Csub_out',
                            'Rsub_in', 'Rsub_mid', 'Rsub_out',
                            'Rx12', 'Cx12', 'Rx13', 'Cx13', 'Rx23', 'Cx23']
        
        tcoil_eq_ckt.to_csv(f'{TCOIL_DATA_DIR}/train/tcoil_eq_ckt_{self.freq_start/1e9}-{self.freq_stop/1e9}GHz_{self.ind_max}_{self.tcoil_num_new-len(defected_design)}_{self.date}.csv',mode='w',header=True)
        
        
        ######################################################################
        # tcoil behavior at the frequency of interest
        
        filename = f'{TCOIL_DATA_DIR}/train/tcoil_results_{self.freq_design/1e9}GHz_{self.ind_max}_{self.tcoil_num_new-len(defected_design)}_{self.date}.csv'
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        

        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['L', 'W', 'S', 'N', 'tap', 'La', 'Ra', 'Lb', 'Rb', 'k', 'Cbr'])
            writer.writerow(['index', 'L','W','S','Nin','Nout','La','Ra','Qa','Lb','Rb','Qb','k','Cbr','Cox_in',
                             'Cox_mid','Cox_out','Rsub_in','Rsub_mid','Rsub_out','Csub_in','Csub_mid','Csub_out'])
            for i in [x for x in range(len(tcoil_results)) if x not in defected_design]:
                writer.writerow([tcoil_results['tcoil{}'.format(i)]['index'],
                                 tcoil_results['tcoil{}'.format(i)]['L'],
                                 tcoil_results['tcoil{}'.format(i)]['W'],
                                 tcoil_results['tcoil{}'.format(i)]['S'],
                                 tcoil_results['tcoil{}'.format(i)]['Nin'],
                                 tcoil_results['tcoil{}'.format(i)]['Nout'],
                                 tcoil_results['tcoil{}'.format(i)]['La'][slicer],
                                 tcoil_results['tcoil{}'.format(i)]['Ra'][slicer],
                                 tcoil_results['tcoil{}'.format(i)]['Qa'][slicer],
                                 tcoil_results['tcoil{}'.format(i)]['Lb'][slicer],
                                 tcoil_results['tcoil{}'.format(i)]['Rb'][slicer],
                                 tcoil_results['tcoil{}'.format(i)]['Qb'][slicer],
                                 tcoil_results['tcoil{}'.format(i)]['k'][slicer],
                                 #tcoil_results['tcoil{}'.format(i)]['fr'],
                                 tcoil_results['tcoil{}'.format(i)]['Cbr'],
                                 tcoil_results['tcoil{}'.format(i)]['Cox_in'],
                                 tcoil_results['tcoil{}'.format(i)]['Cox_mid'],
                                 tcoil_results['tcoil{}'.format(i)]['Cox_out'],
                                 tcoil_results['tcoil{}'.format(i)]['Rsub_in'],
                                 tcoil_results['tcoil{}'.format(i)]['Rsub_mid'],
                                 tcoil_results['tcoil{}'.format(i)]['Rsub_out'],
                                 tcoil_results['tcoil{}'.format(i)]['Csub_in'],
                                 tcoil_results['tcoil{}'.format(i)]['Csub_mid'],
                                 tcoil_results['tcoil{}'.format(i)]['Csub_out']]
                                )
    
        
        if plot == True:
            f = network_eq_ckt.f
            La_mpe = np.mean(abs(np.array(La_list)-np.array(La_eq_ckt_list))/np.array(La_list), axis=0)
            Lb_mpe = np.mean(abs(np.array(Lb_list)-np.array(Lb_eq_ckt_list))/np.array(Lb_list), axis=0)
            Ra_mpe = np.mean(abs(np.array(Ra_list)-np.array(Ra_eq_ckt_list))/np.array(Ra_list), axis=0)
            Rb_mpe = np.mean(abs(np.array(Rb_list)-np.array(Rb_eq_ckt_list))/np.array(Rb_list), axis=0)
            k_mpe = np.mean(abs(np.array(k_list)-np.array(k_eq_ckt_list))/abs(np.array(k_list)), axis=0)
            
            plt.figure('La Lb EMX vs. ModelGen eq. ckt. MPE (%) w.r.t. frequency')
            plt.plot(f, La_mpe * 100, 'r')
            plt.plot(f, Lb_mpe * 100, 'b')
            plt.grid('True')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('MPE (%)')
            plt.legend(['La','Lb'])
            plt.title('La Lb EMX vs. ModelGen eq. ckt. MPE (%) w.r.t. frequency')
            plt.show()
            
            plt.figure('Ra Rb EMX vs. ModelGen eq. ckt. MPE (%) w.r.t. frequency')
            plt.plot(f, Ra_mpe * 100, 'r')
            plt.plot(f, Rb_mpe * 100, 'b')
            plt.grid('True')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('MPE (%)')
            plt.legend(['Ra','Rb'])
            plt.title('Ra Rb EMX vs. ModelGen eq. ckt. MPE (%) w.r.t. frequency')
            plt.show()
            
            plt.figure('k EMX vs. ModelGen eq. ckt. MPE (%) w.r.t. frequency')
            plt.plot(f, k_mpe * 100, 'r')
            plt.grid('True')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('MPE (%)')
            plt.title('k EMX vs. ModelGen eq. ckt. MPE (%) w.r.t. frequency')
            plt.show()
        else:
            None
                
