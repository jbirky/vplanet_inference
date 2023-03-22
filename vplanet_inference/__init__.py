import os

from .model import *
from .analysis import *
from .parameters import *
# from .model import VplanetModel
# from .analysis import AnalyzeVplanetModel
# from .parameters import VplanetParameters
# from vplanet_inference.model import *
# from vplanet_inference.analysis import AnalyzeVplanetModel
# from vplanet_inference.parameters import VplanetParameters

# Get directory of vplanet_inference 
PATH 	   = os.path.realpath(__file__)
VPI_DIR    = os.path.split(os.path.split(PATH)[0])[0]
INFILE_DIR = os.path.join(VPI_DIR, "infiles")
TEMP_DIR   = os.path.join(VPI_DIR, "template")