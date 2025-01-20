;===============================================================================
;   File Name      : CCSchangeParam.il
;   Date Created  : 9-16-2008
;   Group      : Custom IC, Cadence Design Systems Inc.
;   Language      : SKILL
;
;   Last Modified : 9-16-2008
;   Revision No.  : 1
;
;   Description   : This code will find an instance CDF parameter
;                    for a given instance cellName and change to a new value
;
;   How to Run      : In CIW load the file CCSchangeParam.il:
;                    e.g. load("CCSchangeParam.il")
;                    CCSchangeParam("testLib" "testCell" "schematic" "cap" "model" "CAP") 

;                                            CCSchangeParam("ether" "PLL_VCO_250MHZ_INV" "schematic" "nmos2v" "model" "gpdk090_nmos2v_mod")
;
;            where testLib    - library name
;                  testCell   - cell name
;                  schematic  - view name
;                  cap        - instance cellName
;                  model      - parameter name
;                  CAP        - new value
;
;   Tested in      : IC612, IC617
;   SKILL Lint Score (Max=100) : 100
;   SKILL Audit IC50: Passed, no private functions were found.
;===============================================================================

/*******************************************************************************
*   DISCLAIMER: The following code is provided for Cadence customers to use at  *
*   their own risk. The code may require modification to satisfy the           *
*   requirements of any user. The code and any modifications to the code may   *
*   not be compatible with current or future versions of Cadence products.     *
*   THE CODE IS PROVIDED "AS IS" AND WITH NO WARRANTIES, INCLUDING WITHOUT     *
*   LIMITATION ANY EXPRESS WARRANTIES OR IMPLIED WARRANTIES OF MERCHANTABILITY *
*   OR FITNESS FOR A PARTICULAR USE.                           *
*******************************************************************************/
procedure( CCSchangeParam( libname cellname viewname findInstCellName paramName newValue "tttttg")
let( (cv)
   cv = dbOpenCellViewByType(libname cellname viewname nil "a")
   ; find instance param
   foreach(inst cv~>instances
      if( inst~>cellName == findInstCellName    then
     printf("Inst %s  %s\n" inst~>name inst~>cellName)
     foreach(param cdfGetInstCDF(inst)~>parameters
        if( param~>name == paramName      then
          printf("           %s [old]: %L\n" param~>name param~>value)
          param~>value = newValue
          printf("           %s [new]: %L\n" param~>name param~>value)
        );if
     ) ;foreach
      );if
   );foreach
   dbSave(cv)
   dbClose(cv)
) ;let
); end procedure CCSchangeParam
/********************/
