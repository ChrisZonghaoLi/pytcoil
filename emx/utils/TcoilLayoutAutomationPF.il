load("/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/myskill/CCSchangeParam.il") 
load("/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/myskill/MovePinsToInstances.il") 
load("/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/myskill/StreamOutGDS.il") 


cv_schematic = dbOpenCellViewByType("tcoil" "tcoil_master" "schematic") 
CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "pneo" "60" ) 
CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "ctapLocation" "bot" ) 
CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "od" "32.0u" ) 
CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "w" "2.4u" ) 
CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "neo" "9.0" ) 
CCSchangeParam( "tcoil" "tcoil_master" "schematic" "tcoil3_mmw" "nei" "12.0" ) 
lxGenFromSource( cv_schematic ) 
cv_layout = dbOpenCellViewByType("tcoil" "tcoil_master" "layout") 
MovePinsToInstances( cv_schematic cv_layout ) 
StreamOutGDS("tcoil" "tcoil_master" "layout" "/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/tcoil_esd/tcoil_pf/emx/tcoil_pf0" "tcoil_pf0.gds") 


