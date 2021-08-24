import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import numpy as np
from functools import partial
import scipy
import os

os.nice(10)

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

# Prior sampler - alabi format
ps = partial(ut.prior_sampler_normal, prior_data=prior_data, bounds=bounds)

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

sm = SurrogateModel(fn=lnpost, bounds=bounds, prior_sampler=ps, savedir=f"results/{kernel}", labels=labels)
sm.init_samples(ntrain=100, ntest=100, reload=True, scale=True)
sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)
sm.active_train(niter=500, algorithm="bape", gp_opt_freq=10)
sm.plot(plots=["gp_all"])

sm = alabi.cache_utils.load_model_cache(f"results/{kernel}/surrogate_model.pkl")

sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=5e4, opt_init=False)
sm.plot(plots=["emcee_corner"])

sm.run_dynesty(ptform=prior_transform, mode='dynamic')
sm.plot(plots=["dynesty_all"])