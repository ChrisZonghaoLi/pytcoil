/* 
This script will automatically place the terminal pins to the corresponding net and also
putting the labels.

ref: https://community.cadence.com/cadence_technology_forums/f/custom-ic-skill/36600/automatically-placement-of-pins-on-nets

*/

procedure(MovePinsToInstances(cv_schematic cv_layout)
	let( (term pin net childTerm pinFigLocation location instTerm pinFig ins_names params values term_names)
	;cv_layout = dbOpenCellViewByType("tcoil" "tcoil0" "layout")
	foreach(term cv_layout~>terminals
	; iterate over all the pins for this terminal
		foreach(pin term~>pins
			pinFig=pin~>fig
			net=term~>net
			; only do this if there aren't more than one inst term
			unless(cdr(net~>instTerms)
				instTerm=car(net~>instTerms)
				childTerm=instTerm~>term
				childPinFig=car(instTerm~>term~>pins)~>fig
				; location within the instance master
				location=centerBox(childPinFig~>bBox)
				; transform to the top level coordinate system
				location=dbTransformPoint(location instTerm~>inst~>transform)
				pinFigLocation=centerBox(pinFig~>bBox)
				; calculate the overall transform to move the pin to the new location
				;transform=dbConcatTransform(list(-xCoord(pinFigLocation):-yCoord(pinFigLocation) "R0" 1) location)
				transform=dbConcatTransform(list(-xCoord(pinFigLocation):-yCoord(pinFigLocation) "R0" 1) list(xCoord(location):yCoord(location) "R0" 1))
				dbMoveFig(pinFig cv_layout transform)
    		) ; unless
   		) ; foreach
  	) ; foreach

	; Putting labels for the pin

	ins_names = cv_schematic~>instances~>name

	; check if tcoil instance is there
	cond(
			( substring(car(ins_names) -2 1)=="I" tcoil_db = car(cv_schematic~>instances) )
			( substring(car(ins_names) -2 1)!="I" break())
		) ; cond
		
	params = cdfGetInstCDF(tcoil_db) ; get dbId of the CDF parameters of tcoil
	values = params~>parameters~>value ; CDF parameter values of tcoil

	term_names=cv_layout~>terminals~>name; terminal names of the tcoil

	neo=atoi(caddr(cddddr(cddddr(values)))) ; number of segments of outer turns (int)

	cond(
		; check the first pin
		(car(term_names)=="endtap" dbCreateLabel(cv_layout '("JA" "pin") centerBox(caar(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P2" "centerCenter" "R0" "euroStyle" 1))
		(car(term_names)=="centertap" dbCreateLabel(cv_layout '("JA" "pin") centerBox(caar(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P3" "centerCenter" "R0" "euroStyle" 1))
		; when outer turns Nout ("neo") >= 9, input stub will be on layer QB, when <= 8, it will be on layer QA
		(car(term_names)=="input" && neo>=9 dbCreateLabel(cv_layout '("QB" "pin") centerBox(caar(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P1" "centerCenter" "R0" "euroStyle" 1))
		(car(term_names)=="input" && neo<=8 dbCreateLabel(cv_layout '("QA" "pin") centerBox(caar(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P1" "centerCenter" "R0" "euroStyle" 1))
	) ; cond
	
	cond(
		; check the second pin
		(cadr(term_names)=="endtap" dbCreateLabel(cv_layout '("JA" "pin") centerBox(caadr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P2" "centerCenter" "R0" "euroStyle" 1))
		(cadr(term_names)=="centertap" dbCreateLabel(cv_layout '("JA" "pin") centerBox(caadr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P3" "centerCenter" "R0" "euroStyle" 1))
		(cadr(term_names)=="input" && neo>=9 dbCreateLabel(cv_layout '("QB" "pin") centerBox(caadr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P1" "centerCenter" "R0" "euroStyle" 1))
		(cadr(term_names)=="input" && neo<=8 dbCreateLabel(cv_layout '("QA" "pin") centerBox(caadr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P1" "centerCenter" "R0" "euroStyle" 1))	
	) ; cond

	cond(
		; check the third pin
		(caddr(term_names)=="endtap" dbCreateLabel(cv_layout '("JA" "pin") centerBox(caaddr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P2" "centerCenter" "R0" "euroStyle" 1))
		(caddr(term_names)=="centertap" dbCreateLabel(cv_layout '("JA" "pin") centerBox(caaddr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P3" "centerCenter" "R0" "euroStyle" 1))
		(caddr(term_names)=="input" && neo>=9 dbCreateLabel(cv_layout '("QB" "pin") centerBox(caaddr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P1" "centerCenter" "R0" "euroStyle" 1))
		(caddr(term_names)=="input" && neo<=8 dbCreateLabel(cv_layout '("QA" "pin") centerBox(caaddr(cv_layout~>terminals~>net~>pins~>fig~>bBox)) "P1" "centerCenter" "R0" "euroStyle" 1))
	) ; cond
		
 	) ; let
) ; procedure
