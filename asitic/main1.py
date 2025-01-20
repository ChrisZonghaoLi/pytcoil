import sys
import os
import ast

PYTCOIL_DIR = os.environ['PYTCOIL_DIR']
sys.path.append(PYTCOIL_DIR)
from utils.post_processing_asitic import PostProcessingASITIC
from utils import asitic_utils
from utils.hspice_utils import HspiceProcessing
#from utils.hspice_utils import hspice_del_sp

mode = 'train'

# After ASITIC is finished, correct a typo in all ASITIC generated .s2p files
asitic_utils.s2p_corrector(mode=mode)


# Delete all old HSPICE files
#hspice.hspice_del_sp(ind_num=ind_num, mode=mode)

tcoil_designs = HspiceProcessing(mode=mode)

# Extract series component of inductor a and b of each t-coil in case you want
# to use HSPICE to extract series components
tcoil_designs.hspice_tcoil_a_b_ext()

# Next, run HSPICE to extract Cbr and Kab
tcoil_designs.hspice_tcoil_ext() 

# Next, run HSPICE to simulate final t-coil netlists
tcoil_designs.hspice_tcoil() 

print('** All HSPICE simulations are done! **')

# # Now, finally, post-process the data
# tcoil_designs_pp = PostProcessingASITIC(mode=mode)

# comparison = tcoil_designs_pp.summary_tcoil_eq_ckt_vs_asitic()
# summary = tcoil_designs_pp.summary_tcoil()

# tcoil_designs_pp.save2csv(summary, comparison, plot=True)
