'''

    This script is used to generate the Shell and SKILL script that is a wrapper of
    all SKILL functions needed to update tcoil parameters and stream out the
    GDS layout files that are EMX-ready
    
    P1 ("input") -- tcoil -- P2 ("endtap")
                      |
                      |
                      P3 ("centertap")
           
'''

import os
import errno
import pandas as pd

###############################################################################
# declare some environment variable
###############################################################################
try:
    EMX_INSTALL_HOME = os.environ['EMX_INSTALL_HOME']
    MODELGEN_INSTALL_HOME = EMX_INSTALL_HOME #os.environ['MODELGEN_INSTALL_HOME']
    EMX_WORK_DIR = os.environ['EMX_WORK_DIR']
    PROCESS_FILE = os.environ['PROCESS_FILE']
    PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
    TCOIL_DATA_DIR = os.environ['TCOIL_DATA_DIR']
    MY_SKILL_DIR = os.environ['MY_SKILL_DIR']
    print('All environment variables load sucessfully!')
except RuntimeError:
    print('Environment variables loading fails; reload your "env_setup.sh".')


def emx_script(freq_start, freq_stop, freq_step, tcoil_num_old=0, tcoil_num_new=0, repair_mode = False, repair_list = None):
###############################################################################
# shell script for runnung EMX
###############################################################################
    EMX_SETTINGS = f"""-e 1 -t 1 -v 0.5 --3d=* -p 'P000=P1' -p 'P001=P2' -p 'P002=P3' -i P000 -i P001 -i P002  --sweep {freq_start} {freq_stop} --sweep-stepsize {freq_step} --verbose=3 --print-command-line -l 2 --dump-connectivity --quasistatic --dump-connectivity --parallel=0 --simultaneous-frequencies=0 --key=EMXkey --format=psf"""
    
    if repair_mode == False:
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
                
    else: # meaning that some simulations were not done due to license connection failure...
        with open(EMX_WORK_DIR + '/runemx.sh','w') as out:
            for i in repair_list:
                RAW = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.raw'
                RAW_Y = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.raw/tcoil{i}.y'
                TOUCHSTONE = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.s%dp'
                OUTPUT_FILE_SETTINGS = f'-s {RAW} -y {RAW}  --format=matlab -y {RAW_Y} --format=touchstone -s {TOUCHSTONE}'
                
                line1 = f'cd {EMX_WORK_DIR}/tcoil_tcoil{i}.work \n'
                line2 = 'echo Interface date: 11-May-2020 1>&2 \n'
                line3 = f'{EMX_INSTALL_HOME}/emx {EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.gds tcoil_master {PROCESS_FILE} {EMX_SETTINGS} {OUTPUT_FILE_SETTINGS}'
                line4 = '\n\n'
                
                out.writelines([line1, line2, line3, line4])
    

def modelgen_script(tcoil_num_old=0, tcoil_num_new=0, repair_mode = False, repair_list = None):
###############################################################################
# shell script for runnung ModelGen
###############################################################################
    MODELGEN_SETTINGS = "--type=complex_tcoil -i 20 --global-interp"
    
    if repair_mode == False:
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

    else:
        with open(EMX_WORK_DIR + '/runmodelgen.sh','w') as out:
            for i in repair_list:
                TCOIL_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.scs'
                FROM_MODELGEN_SCS = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/from_modelgen.scs'
                SPICE = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.sp'
                RAW_Y = f'{EMX_WORK_DIR}/tcoil_tcoil{i}.work/tcoil{i}.raw/tcoil{i}.y'
                
                line1 = f'cd {EMX_WORK_DIR}/tcoil_tcoil{i}.work \n' 
                line2 = 'echo Interface date: 11-May-2020 1>&2 \n'
                line3 = f'{MODELGEN_INSTALL_HOME}/modelgen --name=tcoil_master {MODELGEN_SETTINGS} --spectre {TCOIL_SCS} --unreordered-spectre {FROM_MODELGEN_SCS} --spice {SPICE} {RAW_Y}'
                line4 = '\n\n'
                
                out.writelines([line1, line2, line3, line4])


def playback_netlist(freq_start, freq_stop, freq_step, tcoil_num_old=0, tcoil_num_new=0):
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


def spectre_sim_script(tcoil_num_old=0, tcoil_num_new=0, repair_mode = False, repair_list = None):
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


def spectre_mod_script(tcoil_num_old=0, tcoil_num_new=0, repair_mode = False, repair_list = None):
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


def tcoil_layout_skill(tcoil_num_old=0, tcoil_num_new=0, tcoil_dims_name='tcoil_dims_new'):
###############################################################################
# SKILL script for runnung EMX
###############################################################################
    tcoil_dims=pd.read_csv(f'{TCOIL_DATA_DIR}/train/{tcoil_dims_name}.csv', usecols = ['L', 'W', 'S', 'Nin', 'Nout'])      

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
            
            line4 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "od" "{}u" ) \n'.format(tcoil_dims['L'][i])
            line5 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "w" "{}u" ) \n'.format(tcoil_dims['W'][i])
            line6 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "neo" "{}" ) \n'.format(tcoil_dims['Nout'][i])
            # for gf22 tcoil the "S" is determined by the value for W, the PCell will auto pick the value so no need to set it 
            # the same is for centertap/endtap stub width that will be automatically adjusted by the PCell
            line7 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "nei" "{}" ) \n'.format(tcoil_dims['Nin'][i])
            ##
            
            line8 = 'lxGenFromSource( cv_schematic ) \n' # generate layout for the tcoil
            line9 = 'cv_layout = dbOpenCellViewByType("tcoil" "tcoil_master" "layout") \n'
            line10 = 'MovePinsToInstances( cv_schematic cv_layout ) \n' # move pins to the instance's terminals then add labels to the ports
            line11 = 'StreamOutGDS("tcoil" "tcoil_master" "layout" "{}" "tcoil{}.gds") \n'.format(fileDir,i)
            line12 = '\n\n'
            
        
            out.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12])
    




