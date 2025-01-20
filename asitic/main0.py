import sys
import os
import yaml
PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
sys.path.append(PYTCOIL_DIR)
from utils import asitic_utils
from common import sq_ind

# generate tcoils
tcoil_dims = sq_ind.tcoil_generator_asitic()

stream = open(f'{PYTCOIL_DIR}/asitic/sim_setup_asitic.yaml','r')
sim_setups = yaml.load(stream, yaml.SafeLoader)
tcoil_num_old = sim_setups['tcoil_num_old']
if tcoil_num_old == 0:
# Delete old asitic .s2p files
    asitic_utils.asitic_del_s2p(mode='train')

# generate ASITIC script
asitic_utils.asitic_script(mode='train')

# Next, run ASITIC
