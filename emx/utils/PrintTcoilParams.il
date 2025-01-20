procedure( PrintTcoilParams(cv)

	let( (names tcoil_db params values param_names)
	names = cv~>instances~>name
	
	cond(
		( substring(car(names) -2 1)=="I" tcoil_db = car(cv~>instances) )
		( substring(car(names) -2 1)!="I" break())
	) ; cond
	
	params = cdfGetInstCDF(tcoil_db)
	values = params~>parameters~>value
	param_names = params~>parameters~>name
	
	printf("Tcoil instance CDF parameter names: %L" param_names)
	printf("\n")
	printf("Tcoil instance CDF parameter values: %L" values)
	) ;let

) ; procedure
