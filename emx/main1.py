'''
    main1.py:
    This script is used to post-process the EMX output data
'''
import os
import yaml
import datetime
import pathlib
from utils import emx_utils
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
EMX_WORK_DIR = os.environ['EMX_WORK_DIR']
from utils.post_processing_emx import PostProcessingEMX

# load simulation setting
stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
tcoil_num_old = sim_setups['tcoil_num_old']
tcoil_num_new = sim_setups['tcoil_num_new']
freq_start = sim_setups['freq_start']
freq_stop = sim_setups['freq_stop']
freq_step = sim_setups['freq_step']

# in case due to the license connection issue some design simulations were missed
emx_defection = []
modelgen_defection = []
spectre_mod_defection = []
spectre_sim_defection = emx_defection 

for i in range(tcoil_num_old, tcoil_num_new):
    print(i)
    emx_file = pathlib.Path(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s3p')
    # modelgen_file = pathlib.Path(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.scs')
    # spectre_mod_file = pathlib.Path(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}_mod.s3p')

    emx_file_old = pathlib.Path(f'{EMX_WORK_DIR}/tcoil_tcoil{tcoil_num_old}.work/tcoil{tcoil_num_old}.s3p')
    # modelgen_file_old = pathlib.Path(f'{EMX_WORK_DIR}/tcoil_tcoil{tcoil_num_old}.work/tcoil{tcoil_num_old}.scs')
    # spectre_mod_file_old = pathlib.Path(f'{EMX_WORK_DIR}/tcoil_tcoil{tcoil_num_old}.work/tcoil{tcoil_num_old}_mod.s3p')
    &
    emx_mtime = datetime.datetime.fromtimestamp(emx_file.stat().st_mtime)
    # modelgen_mtime = datetime.datetime.fromtimestamp(modelgen_file.stat().st_mtime)
    # spectre_mod_mtime = datetime.datetime.fromtimestamp(spectre_mod_file.stat().st_mtime)

    emx_mtime_old = datetime.datetime.fromtimestamp(emx_file_old.stat().st_mtime)
    # modelgen_mtime_old = datetime.datetime.fromtimestamp(modelgen_file_old.stat().st_mtime)
    # spectre_mod_mtime_old = datetime.datetime.fromtimestamp(spectre_mod_file_old.stat().st_mtime)
    
    if emx_file.exists() == False: #or int(''.join(str(emx_mtime.date()).split('-'))) < int(''.join(str(emx_mtime_old.date()).split('-'))): # .s3p did not modified before May 27 mean they were not simulated by EMX
        emx_defection.append(i)
    # if modelgen_file.exists() == False:# or int(''.join(str(modelgen_mtime.date()).split('-'))) < int(''.join(str(spectre_mod_mtime_old.date()).split('-'))):
    #     modelgen_defection.append(i)
    # if spectre_mod_file.exists() == False:# or int(''.join(str(spectre_mod_mtime.date()).split('-'))) < int(''.join(str(spectre_mod_mtime_old.date()).split('-'))):  
    #     spectre_mod_defection.append(i)


if len(emx_defection)!=0:
    emx_utils.emx_script(freq_start=freq_start, 
                         freq_stop=freq_stop, 
                         freq_step=freq_step, 
                         repair_mode=True, 
                         repair_list=emx_defection)
    
    # emx_utils.spectre_sim_script( 
    #                      repair_mode=True, 
    #                      repair_list=spectre_sim_defection)

# if len(modelgen_defection)!=0:
#     emx_utils.modelgen_script(
#                          repair_mode=True, 
#                          repair_list=modelgen_defection)
    
# if len(spectre_mod_defection)!=0:
#     emx_utils.spectre_mod_script( 
#                          repair_mode=True, 
#                          repair_list=spectre_mod_defection)


# if len(emx_defection)==0 and len(modelgen_defection)==0 and len(spectre_mod_defection)==0 :  
if len(emx_defection)==0 and len(modelgen_defection)==0 :  

    # tcoil results summary
    tcoil_results = PostProcessingEMX().summary_tcoil()
    
    # # save dataset to csv and plot the MPE (mean % error) 
    PostProcessingEMX().save2csv(tcoil_results,format='csv')



