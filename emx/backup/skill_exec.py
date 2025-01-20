'''

    This script is used to generate the SKILL script that is a wrapper of
    all SKILL functions needed to update tcoil parameters and stream out the
    GDS layout files that are EMX-ready
    
    P1 ("input") -- tcoil -- P2 ("endtap")
                      |
                      |
                      P3 ("centertap")
           
'''

from common import sq_ind

tcoil_num = 10

tcoil_dim_pd = sq_ind.tcoil_generator_gf22(tcoil_num)


MY_SKILL_DIR = '/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/myskill'
EMX_WORK_DIR = '/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/EMX_work'

# SKILL script for runnung EMX

with open(MY_SKILL_DIR + '/TcoilLayoutAutomation.il','w') as out:
    line1 = 'load("{}/CCSchangeParam.il") \n'.format(MY_SKILL_DIR)
    line2 = 'load("{}/MovePinsToInstances.il") \n'.format(MY_SKILL_DIR)
    line3 = 'load("{}/StreamOutGDS.il") \n'.format(MY_SKILL_DIR)
    line4 = '\n\n'

    out.writelines([line1, line2, line3, line4])

with open(MY_SKILL_DIR + '/TcoilLayoutAutomation.il','a') as out:
    for i in range(tcoil_num):
        fileDir = EMX_WORK_DIR + '/tcoil_tcoil{}.work'.format(i)
        
        line1 = 'cv_schematic = dbOpenCellViewByType("tcoil" "tcoil_master" "schematic") \n'
        ##
        # put codes for changing tcoil instance parameters here using CCSchangeParam( libname cellname viewname findInstCellName paramName newValue )
        
        # adjust input stub position to the default value 60%
        line2 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "pneo" "60" ) \n'
        # adjust centertap stub position to the default postion "bot"
        line3 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "ctapLocation" "bot" ) \n'
        
        line4 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "od" "{}u" ) \n'.format(tcoil_dim_pd['L'][i])
        line5 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "w" "{}u" ) \n'.format(tcoil_dim_pd['W'][i])
        line6 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "neo" "{}" ) \n'.format(tcoil_dim_pd['Nout'][i])
        # for gf22 tcoil the "S" is determined by the value for W, the PCell will auto pick the value so no need to set it 
        # the same is for centertap/endtap stub width that will be automatically adjusted by the PCell
        line7 = 'CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "nei" "{}" ) \n'.format(tcoil_dim_pd['Nin'][i])
        ##
        
        line8 = 'lxGenFromSource( cv_schematic ) \n' # generate layout for the tcoil
        line9 = 'cv_layout = dbOpenCellViewByType("tcoil" "tcoil_master" "layout") \n'
        line10 = 'MovePinsToInstances( cv_schematic cv_layout ) \n' # move pins to the instance's terminals then add labels to the ports
        line11 = 'StreamOutGDS("tcoil" "tcoil_master" "layout" "{}" "tcoil{}.gds") \n'.format(fileDir,i)
        line12 = '\n\n'
        
        
        out.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12])



