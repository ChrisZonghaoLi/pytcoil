#!/bin/tcsh 

# source the environment variables
source env_setup.sh

# run 'main0.py' to generate all shell scripts and SKILL wrapper used to run EMX simulations
python $PYTCOIL_DIR/emx/main0.py

# go to your Cadence directory and run virtuoso
cd $CDS_WORK_DIR
startCds -t gf22x-1020b

cd $PYTCOIL_DIR/emx
# as an option, you can go to the layout of "tcoil_master" to setup Express Pcell
# which saves time of opening a pcell

# then, go to CIW and enter load("./myskill/TcoilLayoutAutomation.il")

