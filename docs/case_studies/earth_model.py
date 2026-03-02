import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import numpy as np
import pandas as pd
import scipy
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
from functools import partial
import corner
import multiprocessing as mp

import logging
import warnings

# Suppress specific error messages
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')


# ========================================================
# Configure vplanet forward model
# ========================================================

inparams = {
            "earth.d40KPowerMan": u.kg * u.m**2 / u.s**3,          
            "earth.d40KPowerCore": u.kg * u.m**2 / u.s**3,          
            # "earth.d232ThPowerMan": u.kg * u.m**2 / u.s**3,         
            # "earth.d232ThPowerCore": u.kg * u.m**2 / u.s**3,         
            # "earth.d238UPowerMan": u.kg * u.m**2 / u.s**3,           
            # "earth.d238UPowerCore": u.kg * u.m**2 / u.s**3,      
            # "earth.d235UPowerMan": u.kg * u.m**2 / u.s**3,           
            # "earth.d235UPowerCore": u.kg * u.m**2 / u.s**3,
            "earth.dTMan": u.K, 
            "earth.dTCore": u.K,  
            "earth.dEruptEff": None,
            "earth.dDTChiRef": None,
            "earth.dViscRef": None,
            "earth.dViscJumpMan": None,
            "earth.dActViscMan": u.m**2 / u.s,
            # "earth.dTrefLind": None,
            # "earth.dDLind": None, 
            # "earth.dViscMeltPhis": None, 
}

inlabels_dict = {
            "earth.d40KPowerMan": r"$^{40}$K Mantle Power",          
            "earth.d40KPowerCore": r"$^{40}$K Core Power",          
            "earth.d232ThPowerMan": r"$^{232}$Th Mantle Power",         
            "earth.d232ThPowerCore": r"$^{232}$Th Core Power",         
            "earth.d238UPowerMan": r"$^{238}$U Mantle Power",           
            "earth.d238UPowerCore": r"$^{238}$U Core Power",      
            "earth.d235UPowerMan": r"$^{235}$U Mantle Power",           
            "earth.d235UPowerCore": r"$^{235}$U Core Power",
            "earth.dTMan": r"Mantle Temp", 
            "earth.dTCore": r"Core Temp",
            "earth.dViscJumpMan": r"Mantle Viscosity Jump",
            "earth.dActViscMan": r"Mantle Activation Viscosity",
            "earth.dTrefLind": r"Lind Temp",
            "earth.dEruptEff": r"Erupt Efficiency",
            "earth.dDTChiRef": r"DT Chi Reference",
            "earth.dViscMeltPhis": r"Visc Melt Phis",
            "earth.dViscRef": r"Visc Reference",
            "earth.dDLind": r"Lind Depth",
}
inlabels = [inlabels_dict[key] for key in inparams.keys()]

# Data: (mean, stdev)
prior_data = [(None, None) for _ in range(len(inparams))]

rad_params = {
            "earth.d40KPowerMan": 3.615780e13,         
            "earth.d40KPowerCore": 3.385730e13,         
            "earth.d232ThPowerMan": 6.515750e12,         
            "earth.d232ThPowerCore": 1.454100e11,         
            "earth.d238UPowerMan": 1.166910e13,           
            "earth.d238UPowerCore": 1.203810e11,      
            "earth.d235UPowerMan": 2.024900e13 ,           
            "earth.d235UPowerCore": 5.015860e11,  
}
# Prior bounds
bounds_dict = {
            "earth.d40KPowerMan": [0.8*rad_params["earth.d40KPowerMan"], 1.5*rad_params["earth.d40KPowerMan"]],          
            "earth.d40KPowerCore": [0.8*rad_params["earth.d40KPowerCore"], 1.5*rad_params["earth.d40KPowerCore"]],         
            "earth.d232ThPowerMan": [0.8*rad_params["earth.d232ThPowerMan"], 1.2*rad_params["earth.d232ThPowerMan"]],        
            "earth.d232ThPowerCore": [0.8*rad_params["earth.d232ThPowerCore"], 1.2*rad_params["earth.d232ThPowerCore"]],        
            "earth.d238UPowerMan": [0.8*rad_params["earth.d238UPowerMan"], 1.2*rad_params["earth.d238UPowerMan"]],          
            "earth.d238UPowerCore": [0.8*rad_params["earth.d238UPowerCore"], 1.2*rad_params["earth.d238UPowerCore"]],     
            "earth.d235UPowerMan": [0.8*rad_params["earth.d235UPowerMan"], 1.2*rad_params["earth.d235UPowerMan"]],           
            "earth.d235UPowerCore": [0.8*rad_params["earth.d235UPowerCore"], 1.2*rad_params["earth.d235UPowerCore"]],
            "earth.dTMan": [2500, 3000], 
            "earth.dTCore": [5800, 6800],
            "earth.dViscJumpMan": [1.1, 2.4],
            "earth.dActViscMan": [2.5e5, 3.1e5],
            "earth.dTrefLind": [4800, 5800],
            "earth.dEruptEff": [0.05, 0.15],
            "earth.dDTChiRef": [0, 0.001],
            "earth.dViscMeltPhis": [0.7, 0.9],  
            "earth.dViscRef": [4e7, 9e8],
            "earth.dDLind": [6500, 7500],
}
bounds = [bounds_dict[key] for key in inparams.keys()]

prior_sampler = partial(ut.prior_sampler_normal, prior_data=prior_data, bounds=bounds)

# ========================================================
# Configure outputs and observational constraints
# ========================================================

outparams_unordered = {
            "final.earth.FMeltUMan": None, 
            "final.earth.HflowCMB": None,   
            "final.earth.HflowUMan": None,  
            # "final.earth.MeltMassFluxMan": None,
            # "final.earth.MagPauseRad": None,
            # "final.earth.MagMom": None,
            "final.earth.RIC": None,                           
            "final.earth.TCMB": None,                           
            "final.earth.TUMan": None,                          
            "final.earth.ViscUMan": u.m**2 / u.s,
            "final.earth.ViscLMan": u.m**2 / u.s,
}
# NOTE: outparams need to be in alphabetical order for vplanet_inference (i should fix this internally in vpi, but for now just do it here)
outparams = dict(sorted(outparams_unordered.items()))

TW_TO_SI = (u.TW).to(u.kg * u.m**2 / u.s**3)
outparams_data = {
            "final.earth.FMeltUMan": [0.06, 0.04],
            "final.earth.HflowCMB": [11.0 * TW_TO_SI, 6.0 * TW_TO_SI],   
            "final.earth.HflowUMan": [38.0 * TW_TO_SI, 3.0 * TW_TO_SI],  
            "final.earth.MeltMassFluxMan": [1.3e6, 0.8e6],
            "final.earth.MagPauseRad": [9.1*const.R_earth.si.value, 0.91*const.R_earth.si.value],
            "final.earth.MagMom": [80e21, 8e21],
            "final.earth.RIC": [1224.1e3, 1e3],                           
            "final.earth.TCMB": [4000, 200],                          
            "final.earth.TUMan": [1587, 34],                         
            "final.earth.ViscUMan": [2.275e18, 2.27e18],
            "final.earth.ViscLMan": [1.5e18, 1.4e18],
}
like_data = np.array([list(outparams_data[key]) for key in outparams.keys()])
outlabels = [x.strip("final.earth.") for x in outparams.keys()]

# ========================================================
# Configure likelihood
# ========================================================

def lnlike(theta, data):
    mdl = vpm_final.run_model(theta, remove=True)
    lnl = -0.5 * np.sum(((mdl - data.T[0])/data.T[1])**2)
    return lnl

# ========================================================

vpm_evol = vpi.VplanetModel(inparams, inpath="../infiles/", outparams=outparams, verbose=True, timesteps=1e7*u.yr)
vpm_final = vpi.VplanetModel(inparams, inpath="../infiles/", outparams=outparams, verbose=False)
            
fiducial_dict = {
            "earth.d40KPowerMan": 3.615780e13,         
            "earth.d40KPowerCore": 3.385730e13,         
            "earth.d232ThPowerMan": 6.515750e12,         
            "earth.d232ThPowerCore": 1.454100e11,         
            "earth.d238UPowerMan": 1.166910e13,           
            "earth.d238UPowerCore": 1.203810e11,      
            "earth.d235UPowerMan": 2.024900e13,           
            "earth.d235UPowerCore": 5.015860e11,
            "earth.dTMan": 3000,
            "earth.dTCore": 6500,
            "earth.dViscJumpMan": 2.4,
            "earth.dActViscMan": 3e5,
            "earth.dTrefLind": 5451.6,
            "earth.dEruptEff": 0.10,
            "earth.dDTChiRef": 0,
            "earth.dViscMeltPhis": 0.8,    
            "earth.dViscRef": 6e7,
            "earth.dDLind": 7000,
}
fiducial = {key: fiducial_dict[key] for key in inparams.keys()}

if __name__ == "__main__":
    theta_test = np.array(list(fiducial.values()))
    evol = vpm_evol.run_model(theta_test, remove=False, outsubpath="fiducial")
    print("fiducial likelihood:", lnlike(theta_test))