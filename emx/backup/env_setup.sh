#!/bin/tcsh 

# Invoke Anaconda virtual environment for running pytcoil
conda activate tcoil

# Source EMX and Cadence virtuoso and spectre
source /CMC/tools/CSHRCs/Cadence.Integrand.10.01
source /CMC/tools/CSHRCs/Cadence.SPECTRE.19.10.496
source /CMC/tools/CSHRCs/Cadence.ICADVM18.10

# set some environment variables
setenv CDS_WORK_DIR /autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b
setenv LM_LICENSE_FILE $CDS_LIC_FILE # load the environment variable of EMX license to Cadence license file
setenv EMX_WORK_DIR $CDS_WORK_DIR/EMX_work
setenv PROCESS_FILE /CMC/kits/gf22/22FDX-EXT/V1.0_2.0b/Emagnetic/EMX/10M_2Mx_5Cx_1Jx_2Qx_LBthick/22fdsoi_10M_2Mx_5Cx_1Jx_2Qx_LBthick_nominal_detailed.encrypted.proc
setenv PYTCOIL_DIR /fs1/eecg/tcc/lizongh2/TCoil_ML/pytcoil
setenv MY_SKILL_DIR $CDS_WORK_DIR/myskill
setenv TCOIL_DATA_DIR /fs1/eecg/tcc/lizongh2/TCoil_ML/data/gf22