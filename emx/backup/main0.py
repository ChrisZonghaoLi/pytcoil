'''
    main0.py:
    This script is used to generate "runemx.sh", "runspectre.sh", and "runmodelgen.sh"
    shell schripts and SKILL wrapper "TcoilLayoutAutomation.il"
'''

import os
import errno
import yaml


###############################################################################
# declare some environment variable
###############################################################################
try:
    EMX_INSTALL_HOME = os.environ['EMX_INSTALL_HOME']
    MODELGEN_INSTALL_HOME = os.environ['MODELGEN_INSTALL_HOME']
    EMX_WORK_DIR = os.environ['EMX_WORK_DIR']
    PROCESS_FILE = os.environ['PROCESS_FILE']
    PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
    MY_SKILL_DIR = os.environ['MY_SKILL_DIR']
    print('All environment variables load sucessfully!')
except RuntimeError:
    print('Environment variables loading fails.')
    # print('EMX and ModelGen environement variables did not source before running this script.')
    # EMX_INSTALL_HOME = '/CMC/tools/cadence/INTEGRAND10.01.000_lnx86/emx64-5.11'
    # MODELGEN_INSTALL_HOME = '/CMC/tools/cadence/INTEGRAND10.01.000_lnx86/modelgen64-2.23'
    # EMX_WORK_DIR = '/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/EMX_work'
    # PROCESS_FILE = '/CMC/kits/gf22/22FDX-EXT/V1.0_2.0b/Emagnetic/EMX/10M_2Mx_5Cx_1Jx_2Qx_LBthick/22fdsoi_10M_2Mx_5Cx_1Jx_2Qx_LBthick_nominal_detailed.encrypted.proc'
    # PYTCOIL_DIR = '/fs1/eecg/tcc/lizongh2/TCoil_ML/pytcoil'
    # MY_SKILL_DIR = '/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/myskill'
    # print('All environment variables load sucessfully!')
    

# load simulation setting
stream = open(f'{PYTCOIL_DIR}/emx/sim_setup_emx.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
tcoil_num_old = sim_setups['tcoil_num_old']
tcoil_num_new = sim_setups['tcoil_num_new']
freq_start = sim_setups['freq_start']
freq_stop = sim_setups['freq_stop']
freq_step = sim_setups['freq_step']


EMX_SETTINGS = f"""-e 1 -t 1 -v 0.5 --3d=* -p 'P000=P1' -p 'P001=P2' -p 'P002=P3' -i P000 -i P001 -i P002  --sweep {freq_start} {freq_stop} --sweep-stepsize {freq_step} --verbose=3 --print-command-line -l 2 --dump-connectivity --quasistatic --dump-connectivity --parallel=0 --threads-per-cpu=2 --simultaneous-frequencies=0 --key=EMXkey --format=psf"""
MODELGEN_SETTINGS = "--type=complex_tcoil -i 20 --global-interp"

###############################################################################
# shell script for runnung EMX
###############################################################################
with open(EMX_WORK_DIR + '/runemx.sh','w') as out:
    for i in range(tcoil_num_old,tcoil_num_new):
        RAW = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.raw'
        RAW_Y = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.raw/tcoil{i}.y'
        TOUCHSTONE = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s%dp'
        OUTPUT_FILE_SETTINGS = f'-s {RAW} -y {RAW}  --format=matlab -y {RAW_Y} --format=touchstone -s {TOUCHSTONE}'
        
        line1 = f'cd {EMX_WORK_DIR}/tcoil_tcoil{i}.work \n'
        line2 = 'echo Interface date: 11-May-2020 1>&2 \n'
        line3 = f'{EMX_INSTALL_HOME}/emx {EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.gds tcoil_master {PROCESS_FILE} {EMX_SETTINGS} {OUTPUT_FILE_SETTINGS}'
        line4 = '\n\n'
        
        out.writelines([line1, line2, line3, line4])

###############################################################################
# shell script for runnung Cadence SPECTRE on EMX simulation data
###############################################################################
with open(EMX_WORK_DIR + '/runspectre_sim.sh','w') as out:
    for i in range(tcoil_num_old,tcoil_num_new):
        RAW_SIM = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}_sim.raw'
        PLAYBACK_SIM_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/playback_tcoil{i}_sim.scs'
        
        line1 = f'cd {EMX_WORK_DIR}/tcoil_tcoil{i}.work \n' 
        line2 = f'spectre -f psfascii -r {RAW_SIM} {PLAYBACK_SIM_SCS} > /dev/null 2>&1'
        line3 = '\n\n'
        
        out.writelines([line1, line2, line3])
        
###############################################################################
# shell script for runnung Cadence SPECTRE on ModelGen eq. ckt. data
###############################################################################
with open(EMX_WORK_DIR + '/runspectre_mod.sh','w') as out:
    for i in range(tcoil_num_old,tcoil_num_new):
        RAW_MOD = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}_mod.raw'
        PLAYBACK_MOD_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/playback_tcoil{i}_mod.scs'
        
        line1 = f'cd {EMX_WORK_DIR}/tcoil_tcoil{i}.work \n' 
        line2 = f'spectre -f psfascii -r {RAW_MOD} {PLAYBACK_MOD_SCS} > /dev/null 2>&1'
        line3 = '\n\n'
        
        out.writelines([line1, line2, line3])

###############################################################################
# shell script for runnung ModelGen
###############################################################################
with open(EMX_WORK_DIR + '/runmodelgen.sh','w') as out:
    for i in range(tcoil_num_old,tcoil_num_new):
        TCOIL_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.scs'
        FROM_MODELGEN_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/from_modelgen.scs'
        SPICE = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.sp'
        RAW_Y = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.raw/tcoil{i}.y'
        
        line1 = f'cd {EMX_WORK_DIR}/tcoil_tcoil{i}.work \n' 
        line2 = 'echo Interface date: 11-May-2020 1>&2 \n'
        line3 = f'{MODELGEN_INSTALL_HOME}/modelgen --name=tcoil_master {MODELGEN_SETTINGS} --spectre {TCOIL_SCS} --unreordered-spectre {FROM_MODELGEN_SCS} --spice {SPICE} {RAW_Y}'
        line4 = '\n\n'
        
        out.writelines([line1, line2, line3, line4])
     
###############################################################################
# generate playback netlist for running SPECTRE on EMX simulation data
###############################################################################        
for i in range(tcoil_num_old,tcoil_num_new):
    PLAYBACK_SIM_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/playback_tcoil{i}_sim.scs'
    if not os.path.exists(os.path.dirname(PLAYBACK_SIM_SCS)):
        try:
            os.makedirs(os.path.dirname(PLAYBACK_SIM_SCS))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    with open(PLAYBACK_SIM_SCS,'w') as out:
        lines = []
        lines.append('simulator lang=spectre \n')
        lines.append('subckt for_import(p0 p1 p2 gnd) \n')
        lines.append(f'model spar nport file="{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s3p" \n')
        lines.append('xsp ( p0 gnd p1 gnd p2 gnd) spar \n')
        lines.append('ends for_import \n')
        lines.append('v_1_1 (p_1_1 0) vsource mag=-1 \n')
        lines.append('v_1_2 (p_1_2 0) vsource mag=0 \n')
        lines.append('v_1_3 (p_1_3 0) vsource mag=0 \n')
        lines.append('x_1 (p_1_1 p_1_2 p_1_3 0) for_import \n')
        lines.append('save v_1_1:p v_1_2:p v_1_3:p \n')
        lines.append('v_2_1 (p_2_1 0) vsource mag=0 \n')          
        lines.append('v_2_2 (p_2_2 0) vsource mag=-1 \n')
        lines.append('v_2_3 (p_2_3 0) vsource mag=0 \n')
        lines.append('x_2 (p_2_1 p_2_2 p_2_3 0) for_import \n')
        lines.append('save v_2_1:p v_2_2:p v_2_3:p \n')
        lines.append('v_3_1 (p_3_1 0) vsource mag=0 \n')
        lines.append('v_3_2 (p_3_2 0) vsource mag=0 \n')
        lines.append('v_3_3 (p_3_3 0) vsource mag=-1 \n')
        lines.append('x_3 (p_3_1 p_3_2 p_3_3 0) for_import \n')
        lines.append('save v_3_1:p v_3_2:p v_3_3:p \n')
        lines.append(f'Y ac start={freq_start} stop={freq_stop} step={freq_step} \n')
        lines.append('xsp (p_1 p_2 p_3 0) for_import \n')
        lines.append('port1 (p_1 0) port \n')
        lines.append('port2 (p_2 0) port \n')
        lines.append('port3 (p_3 0) port \n')
        lines.append(f'S sp start={freq_start} stop={freq_stop} step={freq_step} ports=[ port1 port2 port3] \n')
                     
        out.writelines(lines)
        
###############################################################################     
# generate playback netlist for running SPECTRE on ModelGen eq. ckt. data   
###############################################################################     
for i in range(tcoil_num_old,tcoil_num_new):
    PLAYBACK_SIM_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/playback_tcoil{i}_mod.scs'
    if not os.path.exists(os.path.dirname(PLAYBACK_SIM_SCS)):
        try:
            os.makedirs(os.path.dirname(PLAYBACK_SIM_SCS))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    with open(PLAYBACK_SIM_SCS,'w') as out:
        lines = []
        lines.append('simulator lang=spectre \n')
        lines.append(f'include "{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.scs" \n')
        lines.append('subckt for_import(p0 p1 p2 gnd) \n')
        lines.append('xmod ( p0 p1 p2 gnd) tcoil_master \n')
        lines.append('ends for_import \n')
        lines.append('v_1_1 (p_1_1 0) vsource mag=-1 \n')
        lines.append('v_1_2 (p_1_2 0) vsource mag=0 \n')
        lines.append('v_1_3 (p_1_3 0) vsource mag=0 \n')
        lines.append('x_1 (p_1_1 p_1_2 p_1_3 0) for_import \n')
        lines.append('save v_1_1:p v_1_2:p v_1_3:p \n')
        lines.append('v_2_1 (p_2_1 0) vsource mag=0 \n')          
        lines.append('v_2_2 (p_2_2 0) vsource mag=-1 \n')
        lines.append('v_2_3 (p_2_3 0) vsource mag=0 \n')
        lines.append('x_2 (p_2_1 p_2_2 p_2_3 0) for_import \n')
        lines.append('save v_2_1:p v_2_2:p v_2_3:p \n')
        lines.append('v_3_1 (p_3_1 0) vsource mag=0 \n')
        lines.append('v_3_2 (p_3_2 0) vsource mag=0 \n')
        lines.append('v_3_3 (p_3_3 0) vsource mag=-1 \n')
        lines.append('x_3 (p_3_1 p_3_2 p_3_3 0) for_import \n')
        lines.append('save v_3_1:p v_3_2:p v_3_3:p \n')
        lines.append(f'Y ac start={freq_start} stop={freq_stop} step={freq_step} \n')
        lines.append('xsp (p_1 p_2 p_3 0) for_import \n')
        lines.append('port1 (p_1 0) port \n')
        lines.append('port2 (p_2 0) port \n')
        lines.append('port3 (p_3 0) port \n')
        lines.append(f'S sp file="tcoil{i}_mod.s3p" datafmt=touchstone datatype=realimag start={freq_start} stop={freq_stop} step={freq_step} ports=[ port1 port2 port3] \n')
                     
        out.writelines(lines)

###############################################################################
# SKILL script for runnung EMX
###############################################################################
import sys
sys.path.append(PYTCOIL_DIR)
from common import sq_ind

tcoil_dim_pd = sq_ind.tcoil_generator_gf22()

with open(MY_SKILL_DIR + '/TcoilLayoutAutomation.il','w') as out:
    line1 = 'load("{}/CCSchangeParam.il") \n'.format(MY_SKILL_DIR)
    line2 = 'load("{}/MovePinsToInstances.il") \n'.format(MY_SKILL_DIR)
    line3 = 'load("{}/StreamOutGDS.il") \n'.format(MY_SKILL_DIR)
    line4 = '\n\n'

    out.writelines([line1, line2, line3, line4])

with open(MY_SKILL_DIR + '/TcoilLayoutAutomation.il','a') as out:
    for i in range(tcoil_num_old, tcoil_num_new):
        fileDir = EMX_WORK_DIR + f'/tcoil_tcoil{i}.work'
        
        line1 = 'cv_schematic = dbOpenCellViewByType("tcoil" "tcoil_master" "schematic") \n'
        ##
        # put codes for changing tcoil instance parameters here using CCSchangeParam( libname cellname viewname findInstCellName paramName newValue )
        
        # adjust input stub position to the default value 60%
        line2 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "pneo" "60" ) \n'
        # adjust centertap stub position to the default postion "bot"
        line3 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "ctapLocation" "bot" ) \n'
        
        line4 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "od" "{}u" ) \n'.format(tcoil_dim_pd['L'][i-tcoil_num_old])
        line5 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "w" "{}u" ) \n'.format(tcoil_dim_pd['W'][i-tcoil_num_old])
        line6 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "neo" "{}" ) \n'.format(tcoil_dim_pd['Nout'][i-tcoil_num_old])
        # for gf22 tcoil the "S" is determined by the value for W, the PCell will auto pick the value so no need to set it 
        # the same is for centertap/endtap stub width that will be automatically adjusted by the PCell
        line7 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "nei" "{}" ) \n'.format(tcoil_dim_pd['Nin'][i-tcoil_num_old])
        ##
        
        line8 = 'lxGenFromSource( cv_schematic ) \n' # generate layout for the tcoil
        line9 = 'cv_layout = dbOpenCellViewByType("tcoil" "tcoil_master" "layout") \n'
        line10 = 'MovePinsToInstances( cv_schematic cv_layout ) \n' # move pins to the instance's terminals then add labels to the ports
        line11 = 'StreamOutGDS("tcoil" "tcoil_master" "layout" "{}" "tcoil{}.gds") \n'.format(fileDir,i)
        line12 = '\n\n'
        
        
        out.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12])

