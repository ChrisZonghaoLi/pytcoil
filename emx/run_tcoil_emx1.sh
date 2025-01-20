#!/bin/tcsh 

cd $EMX_WORK_DIR

./runemx.sh

./runspectre_sim.sh

./runmodelgen.sh

./runspectre_mod.sh

#python $PYTCOIL_DIR/emx/main1.py
