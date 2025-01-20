# This script is used to generate the dimension of a square inductor for
# ASITIC input


import random
import pandas as pd
import os
import errno
import yaml
import itertools

from skopt.sampler import Lhs
from skopt.space import Space



# PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
# stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
# sim_setups = yaml.load(stream, yaml.SafeLoader)
# tcoil_num_old = 0#sim_setups['tcoil_num_old']
# tcoil_num_new = 100#sim_setups['tcoil_num_new']
# freq_design = float(sim_setups['freq_design'])

# tcoil_dim_dict = {'L':[],
#                 'W':[],
#                 'S':[],
#                 'Nin':[],
#                 'Nout':[]}

# # since W and S have narrow discreted range, having it randomly selected for
# # massive sample size is not very different compared to LHS
# W_space = [2.4, 4.2, 5]
# W_list = []
# S_list = []

# counter0 = 0 # counts how many W=2.4 or 4.2
# counter1 = 0 # counts how many W=5
# for i in range(tcoil_num_new-tcoil_num_old):
#     W = random.choice(W_space)
#     if W==2.4:
#         counter0 = counter0 + 1
#         S = 1.2
#         S_list.append(S)
#     elif W==4.2:
#         counter0 = counter0 + 1
#         S = 1.44
#         S_list.append(S)
#     else:
#         counter1 = counter1 + 1
#         S = 1.44
#         S_list.append(S)
        
#     W_list.append(W)


# lhs = Lhs(lhs_type='classic', criterion=None)
    
# L_space0 = Space([(32, 81)]) # W=2.4 or W=4.2
# L_list0 = lhs.generate(L_space0.dimensions, n_samples=counter0)
# L_list0 = [L_list0[i][0] for i in range(counter0)]

# L_space1 = Space([(45, 81)]) # W=5
# L_list1 = lhs.generate(L_space1.dimensions, n_samples=counter1)
# L_list1 = [L_list1[i][0] for i in range(counter1)]

# Nout_space = Space([(4, 13)])
# N_list = lhs.generate(Nout_space.dimensions, n_samples=tcoil_num_new-tcoil_num_old)
# N_list = [N_list[i][0] for i in range(tcoil_num_new-tcoil_num_old)]

# # they are in increasing order
# L_list = L_list0 + L_list1
# W_list = sorted(W_list)
# S_list = sorted(S_list)


def tcoil_generator_asitic(sampling='random'):
    """
    Generate square inductor dimensions randomly within the given bound.
    ----------
    Parameters
    ----------

    L_range : list
        lower and upper bound of inductor length in um (int only)
    W_range : list
        lower and upper bound of width of metal trace in um (int only)
    S_range : list
        lower and upper bound of spacing between metal trace in um 
        (should be at least larger than 0.1 and only 1 decimal number)
    tcoil_num: int
        number of inductors you want
        
    

    Returns
    -------
    ind_list : list
        randomly generate tcoil_num amount of different inductor dimension [L, W, S]

    """
    
    PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
    stream = open(f'{PYTCOIL_DIR}/asitic/sim_setup_asitic.yaml','r')
    sim_setups = yaml.load(stream, yaml.SafeLoader)
    tcoil_num_new = sim_setups['tcoil_num_new']
    tcoil_num_old = sim_setups['tcoil_num_old']
    L_range = sim_setups['L_range']
    S_range = sim_setups['S_range']
    W_range = sim_setups['W_range']   
    
    L_low, L_up = L_range
    W_low, W_up = W_range
    S_low, S_up = S_range
    
    tcoil_dim_dict = {'L':[],
                'W':[],
                'S':[],
                'N':[],
                'tap':[]}
    
    if sampling == 'random':
        L_list = [L/10 + L_low for L in range(int(10*(L_up-L_low)))]
        W_list = [W/10 + W_low for W in range(int(10*(W_up-W_low)))]
        S_list = [S/10 + S_low for S in range(int(10*(S_up-S_low)))]
    elif sampling == 'lhs':
        lhs = Lhs(lhs_type='classic', criterion=None)
        LWS_space = Space([(float(L_low), float(L_up)),
               (float(W_low), float(W_up)), 
               (float(S_low), float(S_up))])
        LWS_list = lhs.generate(LWS_space.dimensions, n_samples=tcoil_num_new-tcoil_num_old)

        L_list = [round(LWS_list[i][0],1) for i in range(len(LWS_list))]
        W_list = [round(LWS_list[i][1],1) for i in range(len(LWS_list))]
        S_list = [round(LWS_list[i][2],1) for i in range(len(LWS_list))]
    else:
        print("Wrong sampling mode; should be either 'random' or 'lhs'.")


    for i in range(tcoil_num_new-tcoil_num_old):    
        if sampling == 'random':  
            L = random.choice(L_list)
            W = random.choice(W_list)
            S = random.choice(S_list)
        elif sampling == 'lhs':
            L = L_list[i]
            W = W_list[i]
            S = S_list[i]
        else:
            print("Wrong sampling mode; should be either 'random' or 'lhs'.")
        
        N_max = int((L/(W+S))/2/0.25)*0.25
        
        # since N and tap really dependes on L, W, and S, having N and tap
        # to be LHS sampled is not really taking its advantages
        N = random.choice([0.25*x for x in range(8, int(N_max/0.25))])
        
        # we make sure at least two full turns, worst case each La, Lb has one turn
        #print(N)
        tap_max = int(4*N)
        #print(tap_max)
        # and make sure at least each inductor has one turn
        tap = random.choice([x for x in range(4,tap_max-3)])
        
        tcoil_dim_dict['L'].append(L)
        tcoil_dim_dict['W'].append(W)
        tcoil_dim_dict['S'].append(S) 
        tcoil_dim_dict['N'].append(N)
        tcoil_dim_dict['tap'].append(tap)
        
    tcoil_dim_pd = pd.DataFrame.from_dict(tcoil_dim_dict)
    
    TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']
        
    filename_dim = f'{TCOIL_DATA_DIR}/train/tcoil_dims.csv'
    if not os.path.exists(os.path.dirname(filename_dim)):
        try:
            os.makedirs(os.path.dirname(filename_dim))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    if os.path.isfile(filename_dim): # if the dim file already exists
        if os.stat(filename_dim).st_size == 0: # the csv is empty:
            tcoil_dim_pd.to_csv(filename_dim, mode='w', header=True)
        else:
            tcoil_dim_pd.to_csv(filename_dim, mode='a', header=False) # not empty, append
    else: # file does not exist, write data to the file
        tcoil_dim_pd.to_csv(filename_dim, mode='w', header=True)
        
    return tcoil_dim_pd


def tcoil_generator_gf22(sampling='random'):
    PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
    stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
    sim_setups = yaml.load(stream, yaml.SafeLoader)
    tcoil_num_old = int(sim_setups['tcoil_num_old'])
    tcoil_num_new = int(sim_setups['tcoil_num_new'])
    L_range = sim_setups['L_range']
    L_low, L_up = L_range
    
    # L_low minimum is 32u, and L_up maximum is 80u
    
tcoil_dim_dict = {'L':[],
                'W':[],
                'S':[],
                'Nin':[],
                'Nout':[]}
    
    # since input geometries are discretized, combinations are finite
W_list = [2.4, 4.2, 5]
L_list = [i for i in range(32, 81)]
Nin_list = [i for i in range(5, 25)]
Nout_list = [i for i in range(4, 13)]
geometry_combs = list(itertools.product(*[L_list, W_list, Nin_list, Nout_list]))
defection = []
for i in range(len(geometry_combs)):
    if geometry_combs[i][1] == 5: # W = 5um
        geometry_combs[i] = geometry_combs[i] + (1.44,) # add 'S'
        if geometry_combs[i][0] <= 44:
            defection.append(i)
        elif geometry_combs[i][0] >= 45 and geometry_combs[i][0] <= 48:
            if geometry_combs[i][2] == 8 or geometry_combs[i][2] >= 12:
                defection.append(i)
        elif geometry_combs[i][0] >= 49 and geometry_combs[i][0] <= 54:
            if geometry_combs[i][2] == 8 or geometry_combs[i][2] >= 14:
                defection.append(i)
        elif geometry_combs[i][0] >= 55 and geometry_combs[i][0] <= 57:
            if geometry_combs[i][2] >= 14:
                defection.append(i)
        elif geometry_combs[i][0] >= 58 and geometry_combs[i][0] <= 64:
            if geometry_combs[i][2] >= 15:
                defection.append(i)
        else:
            None
            
    elif geometry_combs[i][1] == 2.4: # W = 2.4um
        geometry_combs[i] = geometry_combs[i] + (1.2,) # add 'S'
        if geometry_combs[i][0] <= 35 and geometry_combs[i][0] >= 32:
            if geometry_combs[i][2] == 8 or geometry_combs[i][2] >= 15:
                defection.append(i)
        elif geometry_combs[i][0] <= 37 and geometry_combs[i][0] >= 36:
            if geometry_combs[i][2] == 8:
                defection.append(i)
        else:
            None
            
    else:  # W = 4.2um
        geometry_combs[i] = geometry_combs[i] + (1.44,) # add 'S'
        if geometry_combs[i][0] <= 33 and geometry_combs[i][0] >= 32:
            if geometry_combs[i][2] != 5:
                defection.append(i)
        elif geometry_combs[i][0] <= 37 and geometry_combs[i][0] >= 34:
            if geometry_combs[i][2] != 5 or geometry_combs[i][2] != 6:
                defection.append(i)
        elif geometry_combs[i][0] <= 40 and geometry_combs[i][0] >= 38:
            if geometry_combs[i][2] == 8 or geometry_combs[i][2] >= 11:
                defection.append(i)
        elif geometry_combs[i][0] <= 45 and geometry_combs[i][0] >= 41:
            if geometry_combs[i][2] == 8 or geometry_combs[i][2] >= 12:
                defection.append(i)
        elif geometry_combs[i][0] <= 49 and geometry_combs[i][0] >= 46:
            if geometry_combs[i][2] == 8 or geometry_combs[i][2] >= 15:
                defection.append(i)
        elif geometry_combs[i][0] <= 51 and geometry_combs[i][0] >= 50:
            if geometry_combs[i][2] >= 15:
                defection.append(i)
        elif geometry_combs[i][0] <= 56 and geometry_combs[i][0] >= 52:
            if geometry_combs[i][2] >= 16:
                defection.append(i)
        else:
            None
       
       
geometry_combs = [i for j, i in enumerate(geometry_combs) if j not in defection] 
random.shuffle(geometry_combs)           

tcoil_dim_dict['L'] = [geometry_combs[i][0] for i in range(len(geometry_combs))]
tcoil_dim_dict['W'] = [geometry_combs[i][1] for i in range(len(geometry_combs))]
tcoil_dim_dict['Nin'] = [geometry_combs[i][2] for i in range(len(geometry_combs))]
tcoil_dim_dict['Nout'] = [geometry_combs[i][3] for i in range(len(geometry_combs))]
tcoil_dim_dict['S'] = [geometry_combs[i][4] for i in range(len(geometry_combs))]

    for W in W_list:
        for W == 2.4:
            S = 1.2
            L_list = [i for i in range(32, 81)]
            for L in L_list:
                if L>=32 and L<=35:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 15)]
                elif L>=36 and L<=37:
                    Nin_list = [i for i in range(5, 8)] + [i for i in range(9, 25)]
                else:
                    Nin_list = [i for i in range(5, 25)]
            
    
    if sampling == 'random':
        for i in range(tcoil_num_old, tcoil_num_new):
            
            # W
            W_list = [2.4, 4.2, 5]
            W = random.choice(W_list)
            
            # S
            if W == 2.4:
                S = 1.2
            else:
                S = 1.44    
            
            # L
            if W == 2.4 or W == 4.2:
                L_list = [i for i in range(L_low, L_up+1)]
            else: # W=5
                if L_low <= 45:
                    L_list = [i for i in range(45, L_up+1)]
                else:
                    L_list = [i for i in range(L_low, L_up+1)]
                    
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
                    
            tcoil_dim_dict['L'].append(L)
            tcoil_dim_dict['W'].append(W)
            tcoil_dim_dict['S'].append(S) 
            tcoil_dim_dict['Nin'].append(Nin)
            tcoil_dim_dict['Nout'].append(Nout)
        
    elif sampling == 'lhs':
         # since W and S have narrow discreted range, having it randomly selected for
         # massive sample size is not very different compared to LHS
         W_space = [2.4, 4.2, 5]
         W_list = []
         S_list = []
         
         counter0 = 0 # counts how many W=2.4 or 4.2
         counter1 = 0 # counts how many W=5
         for i in range(tcoil_num_new-tcoil_num_old):
             W = random.choice(W_space)
             if W==2.4:
                 counter0 = counter0 + 1
                 S = 1.2
                 S_list.append(S)
             elif W==4.2:
                 counter0 = counter0 + 1
                 S = 1.44
                 S_list.append(S)
             else:
                 counter1 = counter1 + 1
                 S = 1.44
                 S_list.append(S)
                 
             W_list.append(W)
         
         
         lhs = Lhs(lhs_type='classic', criterion=None)
             
         L_space0 = Space([(32, 81)]) # W=2.4 or W=4.2
         L_list0 = lhs.generate(L_space0.dimensions, n_samples=counter0)
         L_list0 = [L_list0[i][0] for i in range(counter0)]
         
         L_space1 = Space([(45, 81)]) # W=5
         L_list1 = lhs.generate(L_space1.dimensions, n_samples=counter1)
         L_list1 = [L_list1[i][0] for i in range(counter1)]
         
         Nout_space = Space([(4, 13)])
         Nout_list = lhs.generate(Nout_space.dimensions, n_samples=tcoil_num_new-tcoil_num_old)
         Nout_list = [Nout_list[i][0] for i in range(tcoil_num_new-tcoil_num_old)]
         
         # they are in increasing order, so they match the dependency
         L_list = L_list0 + L_list1
         W_list = sorted(W_list)
         S_list = sorted(S_list)
         
         tcoil_dim_dict['L'] = L_list
         tcoil_dim_dict['W'] = W_list
         tcoil_dim_dict['S'] = S_list
         tcoil_dim_dict['Nout'] = Nout_list
         
         # trouble is the Nin...
         for i in range(tcoil_num_old, tcoil_num_new):
             # W
             W = W_list[i]
                         
             # L
             L = L_list[i]
             
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
                     
             # I left it as randomly selected since its sampling space is too
             # dependent on W and L
             
             Nin = random.choice(Nin_list)  
             tcoil_dim_dict['Nin'].append(Nin)

    else:
         print("Wrong sampling mode; should be either 'random' or 'lhs'.")
      
    
    tcoil_dim_pd = pd.DataFrame.from_dict(tcoil_dim_dict)
    
    TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']
    
    filename_dim = f'{TCOIL_DATA_DIR}/train/tcoil_dims.csv'
    if not os.path.exists(os.path.dirname(filename_dim)):
        try:
            os.makedirs(os.path.dirname(filename_dim))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    if os.path.isfile(filename_dim): # if the dim file already exists
        if os.stat(filename_dim).st_size == 0: # the csv is empty:
            tcoil_dim_pd.to_csv(filename_dim, mode='w', header=True)
        else:
            tcoil_dim_pd.to_csv(filename_dim, mode='a', header=False) # not empty, append
    else: # file does not exist, write data to the file
        tcoil_dim_pd.to_csv(filename_dim, mode='w', header=True)
  

    return tcoil_dim_pd