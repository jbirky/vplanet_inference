import alabi
from alabi import utility as ut
from alabi import SurrogateModel
import numpy as np
from functools import partial
from astropy import units as u

# note: make sure this script is in the same directory as the johnstone_model.py file
# which should be at /research/yupra/username/
from johnstone_model import StellarEvolutionModel


# ========================================================
# CHANGE TO THE DATA FOR YOUR STAR
# ========================================================

# Observational data
# star_name = "Trappist-1"
# mass_data = np.array([0.08, 0.007]) * u.Msun
# Lbol_data = np.array([5.22e-4, 0.19e-4]) * u.Lsun
# Lxuv_data = np.array([1.0e-4, 0.1e-4]) * Lbol_data 
# Prot_data = np.array([3.295, 0.003]) * u.day
# age_data = np.array([7.6, 2.2]) * u.Gyr

# star_name = "GJ_3470"
# mass_data = np.array([0.51, 0.06]) * u.Msun
# Lbol_data = np.array([0.029, 0.002]) * u.Lsun
# Lxray_data = np.array([4.43e27, 7.88e26]) * u.erg/u.s
# Prot_data = np.array([21.54, 0.49]) * u.day

star_name = "55_Cnc"
mass_data = np.array([0.905, 0.015]) * u.Msun
Lbol_data = np.array([0.582, 0.014]) * u.Lsun
Lxray_data = np.array([6.05e26, 5.23e25]) * u.erg / u.s
Prot_data = np.array([38.8, 0.05]) * u.day
age_data = np.array([10.2, 2.5]) * u.Gyr

# ========================================================
# Configure the model
# ========================================================

# the rest of the script from here on you can keep the same

# prior data for input parameters
# Data: (mean, stdev)
prior_data = [(mass_data[0].value, mass_data[1].value),        # mass [Msun]
              (None, None),         # Prot initial [days] 
              (None, None),         # age [Gyr]
              (-0.135, 0.030),      # beta1
              (-1.889, 0.079),      # beta2
              (0.0605, 0.00331),    # Rsat
              (5.135e-4, 3.320e-5)] # RXsat

# Prior bounds (min and max) for each input parameter
# let the mass bounds be 5 sigma away from the mean
sigma_factor = 5

grid_min = 0.07
grid_max = 1.4
min_mass = max(mass_data[0].value - sigma_factor * mass_data[1].value, grid_min)
max_mass = min(mass_data[0].value + sigma_factor * mass_data[1].value, grid_max)

min_beta1 = prior_data[3][0] - sigma_factor * prior_data[3][1]
max_beta1 = prior_data[3][0] + sigma_factor * prior_data[3][1]
min_beta2 = prior_data[4][0] - sigma_factor * prior_data[4][1]
max_beta2 = prior_data[4][0] + sigma_factor * prior_data[4][1]
min_Rsat = prior_data[5][0] - sigma_factor * prior_data[5][1]
max_Rsat = prior_data[5][0] + sigma_factor * prior_data[5][1]
min_RXsat = prior_data[6][0] - sigma_factor * prior_data[6][1]
max_RXsat = prior_data[6][0] + sigma_factor * prior_data[6][1]

bounds = [(min_mass, max_mass),         # mass [Msun]        
          (0.1, 12.0),                  # Prot initial [days]
          (0.5, 5.0),                  # age [Gyr]
          (min_beta1, max_beta1),       # beta1
          (min_beta2, max_beta2),       # beta2
          (min_Rsat, max_Rsat),         # Rsat
          (min_RXsat, max_RXsat)]       # RXsat

# Initialize stellar evolution model 
# We will try multiple configurations using different combinations of data
# but for now we will use the Lbol and Lxray data ()

model1 = StellarEvolutionModel(star_name=star_name,
                               Lbol_data=Lbol_data, 
                               Lxray_data=Lxray_data)

model2 = StellarEvolutionModel(star_name=star_name,
                               Lbol_data=Lbol_data, 
                               Prot_data=Prot_data)

model3 = StellarEvolutionModel(star_name=star_name,
                               Lbol_data=Lbol_data, 
                               Lxray_data=Lxray_data,
                               Prot_data=Prot_data)

model4 = StellarEvolutionModel(star_name=star_name,
                               Lxray_data=Lxray_data)

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

def lnlike(theta):
    _ = model.LXUV_model(theta)
    chi2_array = model.compute_chi_squared_fit()
    lnl = -0.5 * np.sum(chi2_array)
    print("chi2", chi2_array)
    print("lnlike: ", lnl)
    return lnl

# def lnpost(theta):
#     return lnlike(theta) + lnprior(theta)

# ========================================================
# Run alabi
# ========================================================

# change these to run different models
test = "model1"
model = eval(test)

save_dir = f"results/{star_name}/{test}/"

kernel = "ExpSquaredKernel"

# labels for input parameters
labels = [r"$m_{\star}$ [M$_{\odot}$]", 
          r"$P_{\rm rot,i}$ [days]", 
          r"$t_{\rm age}$ [Gyr]", 
          r"$\beta_1$", 
          r"$\beta_2$", 
          r"$R_{\rm sat}$", 
          r"$R_{X,\rm sat}$"]

if __name__ == "__main__":

    # Initialize the surrogate model
    sm = SurrogateModel(fn=lnlike, bounds=bounds, prior_sampler=ps, 
                        savedir=save_dir, cache=True,
                        labels=labels, scale=None, ncore=22)

    # Compute an initial training sample and train the GP
    sm.init_samples(ntrain=200, ntest=100, reload=False)
    sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)

    # Train the GP using the active learning algorithm
    sm.active_train(niter=500, algorithm="bape", gp_opt_freq=10)
    sm.plot(plots=["gp_all"])

    # # Reload the saved model and run MCMC
    # sm = alabi.cache_utils.load_model_cache(save_dir)
    # sm.savedir = save_dir

    # sm.active_train(niter=500, algorithm="bape", gp_opt_freq=10)
    # sm.plot(plots=["gp_all"])

    # MCMC with emcee
    sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=int(1e5), opt_init=False)
    sm.plot(plots=["emcee_corner"])

    # MCMC with dynesty
    sm.run_dynesty(ptform=prior_transform, mode='dynamic')
    sm.plot(plots=["dynesty_all"])