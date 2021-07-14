from .model import *

import os

# Get directory of vplanet_inference 
PATH 	   = os.path.realpath(__file__)
VPI_DIR    = os.path.split(os.path.split(PATH)[0])[0]
INFILE_DIR = os.path.join(VPI_DIR, "infile")