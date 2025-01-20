/*
This script is used to stream out the GDS file from the layout
*/


procedure( StreamOutGDS(libName cell cellName fileDir fileName)

	let( ()
		xstSetField("library" libName)
		xstSetField("topCell" cell)
		xstSetField("view" cellName)
		xstSetField("runDir" fileDir) ;"/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/EMX_work/tcoil_tcoil1.work"
		xstSetField("logFile" strcat(fileDir "/" "strmOut.log")) ;"/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/EMX_work/tcoil_tcoil1.work/strmOut.log"
		xstSetField("hierDepth" "32")
		xstSetField("strmFile" strcat(fileDir "/" fileName)) ; "/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/gf22x-1020b/EMX_work/tcoil_tcoil1.work/tcoil1.gds"
		xstSetField("layerMap" "/autofs/fs1.ece/fs1.vrg.CMC/kits/gf22/22FDX-EXT/V1.0_2.0b/DesignEnv/VirtuosoOA/libs/cmos22fdsoi_tech_10M_2Mx_5Cx_1Jx_2Qx_LB/cmos22fdsoi_tech.layermap")
		xstSetField("objectMap" "/autofs/fs1.ece/fs1.vrg.CMC/kits/gf22/22FDX-EXT/V1.0_2.0b/DesignEnv/VirtuosoOA/libs/cmos22fdsoi_tech_10M_2Mx_5Cx_1Jx_2Qx_LB/cmos22fdsoi_tech.objectmap")
		xstSetField("case" "preserve")
		xstSetField("convertDot" "node")
		xstSetField("pinAttNum" "1")
		xstSetField("enableColoring" "true")
		xstSetField("virtuoalMemory" "true")
		xstSetField("showCompletionMsgBox" "false")
		xstOutDoTranslate()
	)

)
