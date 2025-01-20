'''
    main0.py:
    This script is used to generate "runemx.sh", "runspectre.sh", and "runmodelgen.sh"
    shell schripts and SKILL wrapper "TcoilLayoutAutomation.il"
'''

import os
import errno
import yaml
from utils import emx_utils


###############################################################################
# declare some environment variable
###############################################################################
try:
    PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
    print('All environment variables load sucessfully!')
except RuntimeError:
    print('Environment variables loading fails; reload your "env_setup.sh".')

    
# load simulation setting
stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
tcoil_num_old = sim_setups['tcoil_num_old']
tcoil_num_new = sim_setups['tcoil_num_new']
freq_start = sim_setups['freq_start']
freq_stop = sim_setups['freq_stop']
freq_step = sim_setups['freq_step']
tcoil_dims_name = sim_setups['tcoil_dims_name']

##############################################################################
# SKILL script for streaming layout
##############################################################################
import sys
sys.path.append(PYTCOIL_DIR)
from common import sq_ind

#geometry_combs = sq_ind.tcoil_generator_gf22(skip_L = 2, skip_Nin=2, skip_Nout=2)
emx_utils.tcoil_layout_skill(tcoil_num_old=tcoil_num_old, 
                              tcoil_num_new=tcoil_num_new, tcoil_dims_name=tcoil_dims_name)

###############################################################################
# generate playback netlist for running SPECTRE on EMX simulation data
# AND
# generate playback netlist for running SPECTRE on ModelGen eq. ckt. data 
###############################################################################   
emx_utils.playback_netlist(tcoil_num_old=tcoil_num_old, 
                           tcoil_num_new=tcoil_num_new, 
                           freq_start=freq_start, 
                           freq_stop=freq_stop, 
                           freq_step=freq_step)

###############################################################################
# shell script for runnung EMX
###############################################################################
emx_utils.emx_script(tcoil_num_old=tcoil_num_old, 
                     tcoil_num_new=tcoil_num_new, 
                     freq_start=freq_start, 
                     freq_stop=freq_stop, 
                     freq_step=freq_step)

###############################################################################
# shell script for runnung Cadence SPECTRE on EMX simulation data
###############################################################################
emx_utils.spectre_sim_script(tcoil_num_old=tcoil_num_old,
                             tcoil_num_new=tcoil_num_new)

###############################################################################
# shell script for runnung ModelGen
###############################################################################
emx_utils.modelgen_script(tcoil_num_old=tcoil_num_old,
                          tcoil_num_new=tcoil_num_new)

###############################################################################
# shell script for runnung Cadence SPECTRE on ModelGen eq. ckt. data
###############################################################################
emx_utils.spectre_mod_script(tcoil_num_old=tcoil_num_old,
                             tcoil_num_new=tcoil_num_new)


     





