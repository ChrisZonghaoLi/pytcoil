;------------------ From here ------------------------
/*******************************************************************************
*  DISCLAIMER: The following code is provided for Cadence customers to use at  *
*   their own risk. The code may require modification to satisfy the           *
*   requirements of any user. The code and any modifications to the code may   *
*   not be compatible with current or future versions of Cadence products.     *
*   THE CODE IS PROVIDED "AS IS" AND WITH NO WARRANTIES, INCLUDING WITHOUT     *
*   LIMITATION ANY EXPRESS WARRANTIES OR IMPLIED WARRANTIES OF MERCHANTABILITY *
*   OR FITNESS FOR A PARTICULAR USE.                                           *
*******************************************************************************/
;; FileName: CCSUpdateCDF.il
;;
;; This procedure is used to change the value of specified CDF parameter
;; of all the instances of a particular cell. Works for the open schematic. 
;;
;; Usage :      CCSUpdateCDF("cellName" "cdf_param_name" "new_value")
;; Example:     : CCSUpdateCDF("pmos1v" "w" "0.4u"),  CCSUpdateCDF("pmos1v" "model" "gpdk090_pmos1v_mod")

procedure(CCSUpdateCDF(CELL PARAM VALUE)
foreach(inst geGetEditCellView()~>instances
if(inst~>cellName == CELL then
   foreach(param cdfGetInstCDF(inst)~>parameters
     if(param->name == PARAM then
         param->value = VALUE );if
    );foreach
 );if
);foreach
);procedure
;--------------- Till here ----------------------------
