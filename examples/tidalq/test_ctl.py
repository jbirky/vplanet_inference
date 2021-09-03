import vplanet_inference as vpi
import numpy as np
import scipy
import os
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)


inpath = os.path.join(vpi.INFILE_DIR, "stellar_eqtide/ctl")

inparams = {"primary.dMass": u.Msun, 
            "secondary.dMass": u.Msun, 
            "primary.dRotPeriod": u.day, 
            "secondary.dRotPeriod": u.day, 
            "primary.dTidalTau": u.dex(u.s), 
            "secondary.dTidalTau": u.dex(u.s), 
            "primary.dObliquity": u.deg, 
            "secondary.dObliquity": u.deg, 
            "secondary.dEcc": u.dimensionless_unscaled, 
            "secondary.dOrbPeriod": u.day,
            "vpl.dStopTime": u.Gyr}

outparams = {"final.primary.Radius": u.Rsun, 
             "final.secondary.Radius": u.Rsun,
             "final.primary.Luminosity": u.Lsun, 
             "final.secondary.Luminosity": u.Lsun,
             "final.primary.RotPer": u.day, 
             "final.secondary.RotPer": u.day,
             "final.secondary.OrbPeriod": u.day,
             "final.secondary.Eccentricity": u.dimensionless_unscaled}

vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams)

theta = np.array([1.08, 1.07, 5.0, 5.0, -1.0, -1.0, 10., 10., .5, 6.0, 2.48])

output = vpm.run_model(theta)