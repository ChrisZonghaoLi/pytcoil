#!/bin/tcsh
conda activate tcoil

source /CMC/tools/CSHRCs/Synopsys.2013.03
source /CMC/tools/CSHRCs/Synopsys.Hspice.2013 

# set some environment variables
setenv ASITIC_WORK_DIR /fs1/eecg/tcc/lizongh2/TCoil_ML/asitic
setenv PYTCOIL_DIR /fs1/eecg/tcc/lizongh2/TCoil_ML/pytcoil
setenv TCOIL_DATA_DIR /fs1/eecg/tcc/lizongh2/TCoil_ML/data/7nm
setenv HSPICE_WORK_DIR /autofs/fs1.ece/fs1.eecg.tcc/lizongh2/TCoil_ML/hspice


