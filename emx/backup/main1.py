'''
    main1.py:
    This script is used to post-process the EMX output data
'''
import os
import sys
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
EMX_WORK_DIR = os.environ['EMX_WORK_DIR']
sys.path.append(PYTCOIL_DIR)
from utils.post_processing_emx import PostProcessingEMX

# in case due to the license connection issue some designs were missed
emx_defection = []
modelgen_defection = []
spectre_mod_defection = []
for i in range(10000, 15000):
    emx_file = os.path.isfile(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s3p')
    modelgen_file = os.path.isfile(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.scs')
    spectre_mod_file = os.path.isfile(f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}_mod.s3p')
    if emx_file == False:
        emx_defection.append(i)
    elif modelgen_file == False:
        modelgen_defection.append(i)
    elif spectre_mod_file == False:  
        spectre_mod_defection.append(i)
    else:
        None

# tcoil results summary
tcoil_results = PostProcessingEMX().summary_tcoil()

# # save dataset to csv and plot the MPE (mean % error) 
PostProcessingEMX().save2csv(tcoil_results)



