# This script is used to generate the script for ASITIC to run simulation

import sys
import os
import yaml
import pandas as pd
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
ASITIC_WORK_DIR = os.environ['ASITIC_WORK_DIR']
TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']

stream = open(f'{PYTCOIL_DIR}/asitic/sim_setup_asitic.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)

freq_start = float(sim_setups['freq_start'])
freq_stop = float(sim_setups['freq_stop'])
freq_step = float(sim_setups['freq_step'])
tcoil_num_old = int(sim_setups['tcoil_num_old'])
tcoil_num_new = int(sim_setups['tcoil_num_new'])
tcoil_num_test = int(sim_setups['tcoil_num_test'])

def asitic_script(mode):
    """
    Generate ASITIC scripts of inductor designs for ASITIC to run
    """
    
    global tcoil_num_old
    global tcoil_num_new

    try:
        mode == 'train' or mode == 'test'
    except:
        print('Wrong entry for "mode", should be either "train" or "test".')
 
    if mode == 'test':
        tcoil_num_old = 0
        tcoil_num_new = tcoil_num_test
        
    tcoil_dims=pd.read_csv(f'{TCOIL_DATA_DIR}/train/tcoil_dims.csv', usecols = ['L', 'W', 'S', 'N', 'tap'])  
        
    with open(f'{ASITIC_WORK_DIR}/{mode}/testbench','w') as out:
        for i in range(tcoil_num_old,tcoil_num_new):
            line1 = 'sq name={} len={} w={} s={} n={} xorg=20 yorg=20 metal=m2 exit=m1 \n'.format(
                'ind{}'.format(i),
                tcoil_dims['L'][i], 
                tcoil_dims['W'][i], 
                tcoil_dims['S'][i], 
                tcoil_dims['N'][i])
            line2 = '2portx ind{} ind{} {} {} {} S true false S_tcoil{}_ab.s2p \n'.format(i, i, freq_start/1e9, freq_stop/1e9, freq_step/1e9, i)
            line3 = 'split ind{} {} ind{}_b \n'.format(i, int(tcoil_dims['tap'][i-tcoil_num_old]), i)
            line4 = 'rename ind{} ind{}_a \n'.format(i, i)
            line5 = '2portx ind{}_a ind{}_a {} {} {} S true false S_tcoil{}_a.s2p \n'.format(i, i, freq_start/1e9, freq_stop/1e9, freq_step/1e9, i)
            line6 = '2portx ind{}_b ind{}_b {} {} {} S true false S_tcoil{}_b.s2p \n'.format(i, i, freq_start/1e9, freq_stop/1e9, freq_step/1e9, i)
            line7 = 'del ind{}_a \n'.format(i)
            line8 = 'del ind{}_b \n'.format(i)
            line9 = '\n'
            out.writelines([line1, line2, line3, line4, line5,
                            line6, line7, line8, line9])
            
    print('** ASITIC script is generated. **')
    
    return tcoil_dims


def asitic_del_s2p(mode):
    """
    Delete old ASITIC .s2p files
    """
    dir_name = f'{ASITIC_WORK_DIR}/{mode}'
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith('.s2p'):
            os.remove(os.path.join(dir_name, item))
            print('*** Old asitic .s2p files for {} are deleted. **'.format(item))
        else:
            pass


def s2p_corrector(mode):
    """
    Since when ASITIC generate .s2p file, line 27 is mistaken, we need to change this line
    to '# HZ S MA R 50'
    """
    
    global tcoil_num_old
    global tcoil_num_new
    
    if mode == 'test':
        tcoil_num_old = 0
        tcoil_num_new = tcoil_num_test

    for i in range(tcoil_num_old,tcoil_num_new):
        print(i)
        s2p_a = open(f'{ASITIC_WORK_DIR}/{mode}/S_tcoil{i}_a.s2p','r')
        list_of_lines_a = s2p_a.readlines()
        list_of_lines_a[24] = '# HZ S MA R 50 \n'
        s2p_a = open(f'{ASITIC_WORK_DIR}/{mode}/S_tcoil{i}_a.s2p','w')
        s2p_a.writelines(list_of_lines_a)
        s2p_a.close()
        
        s2p_b = open(f'{ASITIC_WORK_DIR}/{mode}/S_tcoil{i}_b.s2p','r')
        list_of_lines_b = s2p_b.readlines()
        list_of_lines_b[24] = '# HZ S MA R 50 \n'
        s2p_b = open(f'{ASITIC_WORK_DIR}/{mode}/S_tcoil{i}_b.s2p','w')
        s2p_b.writelines(list_of_lines_b)
        s2p_b.close()
    
        s2p_ab = open(f'{ASITIC_WORK_DIR}/{mode}/S_tcoil{i}_ab.s2p','r')
        list_of_lines_ab = s2p_ab.readlines()
        list_of_lines_ab[24] = '# HZ S MA R 50 \n'
        s2p_ab = open(f'{ASITIC_WORK_DIR}/{mode}/S_tcoil{i}_ab.s2p','w')
        s2p_ab.writelines(list_of_lines_ab)
        s2p_ab.close()

    print('** All s2p files are corrected. **')