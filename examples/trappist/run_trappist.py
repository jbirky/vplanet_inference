import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import numpy as np
from scipy.stats import norm
from functools import partial
import scipy
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
font = {'family' : 'normal',
        'weight' : 'light'}
rc('font', **font)

# ========================================================
# Configure vplanet forward model
# ========================================================

inpath = os.path.join(vpi.INFILE_DIR, "stellar")
infile_list = ["vpl.in", "star.in"]

inparams  = ["star.dMass",          # mass [Msun]
             "star.dSatXUVFrac",    # fsat
             "star.dSatXUVTime",    # tsat [yr]
             "vpl.dStopTime",       # age [yr]
             "star.dXUVBeta"]       # beta

outparams = ["final.star.Luminosity",
             "final.star.LXUVStellar"]

factor = np.array([1, 1, 1e9, 1e9, -1])

def fsat_conversion(fsat):
    return (10 ** fsat) 

conversions = {1:fsat_conversion}

vpm = vpi.VplanetModel(inparams, inpath=inpath, infile_list=infile_list, 
                       factor=factor, conversions=conversions)

# ========================================================
# Observational constraints
# ========================================================

prior_data = [(None, None),     # mass [Msun]
              (-2.92, 0.26),    # log(fsat) 
              (None, None),     # tsat [Gyr]
              (7.6, 2.2),       # age [Gyr]
              (-1.18, 0.31)]    # beta

# like_data = np.array([[5.22e-4, 0.19e-4],   # Lbol [Lsun]
#                       [3.9e-7, 0.5e-7]])    # Lxuv [Lsun]

like_data = np.array([[5.22e-4, 0.19e-4],   # Lbol [Lsun]
                      [7.5e-4, 1.5e-4]])    # Lxuv/Lbol

# Prior bounds
bounds = [(0.07, 0.11),        
          (-5.0, -1.0),
          (0.1, 12.0),
          (0.1, 12.0),
          (-2.0, 0.0)]

# ========================================================
# Configure prior 
# ========================================================

# Prior - emcee format
lnprior = partial(ut.lnprior_normal, bounds=bounds, data=prior_data)

# Prior - dynesty format
prior_transform = partial(ut.prior_transform_normal, bounds=bounds, data=prior_data)

# ========================================================
# Configure likelihood
# ========================================================

# vpm.initialize_bayes(data=like_data, bounds=bounds, outparams=outparams)

def lnlike(theta):
    out = vpm.run_model(theta, outparams=outparams)
    mdl = np.array([out[0], out[1]/out[0]])
    lnl = -0.5 * np.sum(((mdl - like_data.T[0])/like_data.T[1])**2)
    return lnl

def lnpost(theta):
    return lnlike(theta) + lnprior(theta)

# ========================================================
# Run alabi
# ========================================================

kernel = "ExpSquaredKernel"

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

sm = SurrogateModel(fn=lnpost, bounds=bounds, savedir=f"results/{kernel}", labels=labels)
sm.init_samples(ntrain=100, ntest=100)
# sm.init_samples(train_file="initial_training_sample.npz", test_file="initial_test_sample.npz")
sm.init_gp(kernel=kernel, fit_amp=True, fit_mean=True, white_noise=-15)
sm.active_train(niter=500, algorithm="bape", gp_opt_freq=20)
sm.plot(plots=["gp_all"])

# from alabi.cache_utils import load_model_cache
# sm = load_model_cache(f"results/{kernel}/surrogate_model.pkl")

sm.run_emcee(lnprior=lnprior, nwalkers=20, nsteps=2e4, opt_init=False)
sm.plot(plots=["emcee_corner"])

sampler_kwargs = {} #{'bound': 'single'}
run_kwargs = {'dlogz_init': 0.05, 'wt_kwargs': {'pfrac': 1.0}}
sm.run_dynesty(ptform=prior_transform, mode='dynamic', multi_proc=False, 
               sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
sm.plot(plots=["dynesty_all"])