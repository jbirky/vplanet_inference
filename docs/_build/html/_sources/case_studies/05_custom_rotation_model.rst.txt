Custom Rotation-Activity-XUV Model
===================================

Fitting stellar evolution to observational constraints using ``johnstone_model.py``
and ``run_alabi.py``.

This tutorial walks through using a custom stellar XUV luminosity model — implemented on top
of VPLanet's ``stellar`` module — and the ``alabi`` surrogate MCMC framework to infer stellar
properties from present-day observations of luminosity, X-ray emission, rotation period, and age.

**What we cover:**

1. :ref:`johnstone-model` — the rotation-activity relation and its 7 free parameters
2. :ref:`model-setup` — configuring ``StellarEvolutionModel`` with observational data
3. :ref:`single-eval` — running a single forward model and computing chi-squared
4. :ref:`sweeps` — parallel parameter sweeps and evolution plots
5. :ref:`new-star` — the only section to edit in ``run_alabi.py`` for a new star
6. :ref:`constraints` — testing different combinations of observational constraints
7. :ref:`prior-likelihood` — prior and likelihood functions for ``alabi``
8. :ref:`alabi-run` — the full surrogate MCMC pipeline
9. :ref:`reload` — reloading a saved run to continue training or run MCMC

----

**Files:**

- ``docs/case_studies/johnstone_model.py`` — ``StellarEvolutionModel`` class
- ``docs/case_studies/run_alabi.py`` — inference script; copy and edit for your star

**Dependencies:** ``vplanet_inference``, ``alabi``, ``numpy``, ``astropy``, ``matplotlib``,
``emcee`` or ``dynesty``

**References:**

- `Johnstone et al. (2021) <https://doi.org/10.3847/1538-4357/abcc02>`_ — rotation-activity-XUV model
- `King & Wheatley (2021) <https://doi.org/10.1093/mnras/stab1564>`_ — EUV estimation from X-ray
- `Birky et al. (2021) <https://doi.org/10.3847/1538-4357/abded5>`_ — ``alabi`` surrogate method

----

.. _johnstone-model:

1. The Johnstone XUV Model
--------------------------

Standard VPLanet stellar evolution tracks bolometric luminosity, radius, and rotation period
using the Baraffe stellar models and a magnetic braking prescription. Its built-in XUV model
is parameterized by ``dSatXUVFrac``, ``dSatXUVTime``, and ``dXUVBeta``.

``StellarEvolutionModel`` replaces that with the **Johnstone et al. (2021)** formulation,
which ties X-ray emission to the Rossby number:

.. math::

   R_X = \frac{L_X}{L_{\rm bol}} =
   \begin{cases}
   C_1 \, Ro^{\beta_1} & Ro < Ro_{\rm sat} \quad \text{(saturated)} \\
   C_2 \, Ro^{\beta_2} & Ro \geq Ro_{\rm sat} \quad \text{(unsaturated)}
   \end{cases}

where :math:`C_1 = R_{X,\rm sat} / Ro_{\rm sat}^{\beta_1}` and
:math:`C_2 = R_{X,\rm sat} / Ro_{\rm sat}^{\beta_2}` enforce continuity at :math:`Ro_{\rm sat}`.

The Rossby number from VPLanet is rescaled by :math:`0.95 / 2.11` to match the Johnstone et al.
(2021) convective turnover time convention. EUV luminosity is then estimated from X-ray
luminosity via the King & Wheatley (2021) empirical relation, and total XUV is:

.. math::

   L_{\rm XUV} = L_X + L_{\rm EUV}

**Free parameters** — 7 total:

.. list-table::
   :widths: 15 15 50
   :header-rows: 1

   * - Parameter
     - Symbol
     - Description
   * - ``mstar``
     - :math:`m_\star`
     - Stellar mass [:math:`M_\odot`]
   * - ``prot``
     - :math:`P_{\rm rot,i}`
     - Initial rotation period [days]
   * - ``age``
     - :math:`t_{\rm age}`
     - Stellar age [Gyr] — the simulation stop time
   * - ``beta1``
     - :math:`\beta_1`
     - Power-law slope in the saturated regime
   * - ``beta2``
     - :math:`\beta_2`
     - Power-law slope in the unsaturated regime
   * - ``Rosat``
     - :math:`Ro_{\rm sat}`
     - Rossby number at the saturation boundary
   * - ``RXsat``
     - :math:`R_{X,\rm sat}`
     - X-ray saturation fraction :math:`L_X/L_{\rm bol}`

The last four parameters are the Johnstone et al. (2021) activity law coefficients.
They are inferred jointly with the stellar parameters, propagating activity-model
uncertainty into the posterior.

.. _model-setup:

2. ``StellarEvolutionModel`` Setup
------------------------------------

Initialize the class with any combination of observational constraints.
Each constraint you pass will be included in the chi-squared fit; omitted ones are ignored.

.. code-block:: python

   import numpy as np
   from astropy import units as u
   from johnstone_model import StellarEvolutionModel

   # Observational data: [mean, std] with astropy units
   Lbol_data = np.array([5.22e-4, 0.19e-4]) * u.Lsun
   Lxuv_data = np.array([1.0e-4,  0.1e-4])  * Lbol_data   # relative to Lbol
   Prot_data = np.array([3.295,   0.003])   * u.day
   age_data  = np.array([7.6,     2.2])     * u.Gyr

   model = StellarEvolutionModel(
       star_name="Trappist-1",
       Lbol_data=Lbol_data,
       Lxuv_data=Lxuv_data,
       Prot_data=Prot_data,
       age_data=age_data,
   )

Internally ``__init__`` creates a ``vpi.VplanetModel`` instance (``self.vpm``) pointing
to the bundled ``infiles/stellar/`` directory.

**VPLanet input parameters** substituted at each call:

.. list-table::
   :widths: 25 15 40
   :header-rows: 1

   * - VPLanet key
     - Unit
     - Role
   * - ``star.dMass``
     - :math:`M_\odot`
     - Stellar mass
   * - ``star.dRotPeriod``
     - days
     - Initial rotation period
   * - ``vpl.dStopTime``
     - Gyr
     - Simulation duration = stellar age

**VPLanet output parameters** collected from the ``.forward`` time-series file:

.. list-table::
   :widths: 30 15 35
   :header-rows: 1

   * - VPLanet key
     - Unit
     - Role
   * - ``final.star.Luminosity``
     - :math:`L_\odot`
     - Bolometric luminosity over time
   * - ``final.star.Radius``
     - :math:`R_\odot`
     - Stellar radius over time
   * - ``final.star.RotPer``
     - days
     - Rotation period over time
   * - ``final.star.RossbyNumber``
     - —
     - Rossby number (rescaled internally)

The model starts at ``time_init=5e6 yr`` and outputs every ``timesteps=1e6 yr``,
so evolution arrays span the full age at 1 Myr resolution.

.. note::

   ``run_alabi.py`` must be placed in the same directory as ``johnstone_model.py``
   because ``johnstone_model.py`` uses the relative path ``../infiles/stellar/`` to
   locate the VPLanet input files.

.. _single-eval:

3. Single Model Evaluation
---------------------------

``LXUV_model(theta)`` runs one complete forward model.
``theta`` is a 7-element array in the order ``[mstar, prot, age, beta1, beta2, Rosat, RXsat]``.

.. code-block:: python

   # Johnstone+2021 best-fit activity parameters
   j21 = {
       "beta1": (-0.135, 0.030),
       "beta2": (-1.889, 0.079),
       "Rosat": (0.0605, 0.00331),
       "RXsat": (5.135e-4, 3.320e-5),
   }

   theta = np.array([
       0.09,              # mstar [Msun]
       0.5,               # prot_initial [days]
       7.6,               # age [Gyr]
       j21["beta1"][0],
       j21["beta2"][0],
       j21["Rosat"][0],
       j21["RXsat"][0],
   ])

   evol = model.LXUV_model(theta)

The returned dictionary ``evol`` contains time-series arrays (with astropy units):

.. list-table::
   :widths: 40 40
   :header-rows: 1

   * - Key
     - Description
   * - ``"Time"``
     - Time since formation [Gyr]
   * - ``"final.star.Luminosity"``
     - Bolometric luminosity [:math:`L_\odot`]
   * - ``"final.star.Radius"``
     - Stellar radius [:math:`R_\odot`]
   * - ``"final.star.RotPer"``
     - Rotation period [days]
   * - ``"final.star.RossbyNumberScaled"``
     - Rossby number (Johnstone convention)
   * - ``"final.star.RX"``
     - X-ray activity fraction :math:`R_X = L_X / L_{\rm bol}`
   * - ``"final.star.LXRAY"``
     - X-ray luminosity
   * - ``"final.star.LEUV"``
     - EUV luminosity (King & Wheatley 2021)
   * - ``"final.star.LXUV"``
     - Total XUV luminosity

After calling ``LXUV_model``, the result is stored as ``model.evol``.
Compute the chi-squared against all initialized observables with:

.. code-block:: python

   chi2_array = model.compute_chi_squared_fit()
   print("Chi-squared per observable:", chi2_array)
   print("Total chi-squared:", np.sum(chi2_array))

``compute_chi_squared_fit`` returns :math:`\chi^2_i = (m_i - d_i)^2 / \sigma_i^2` for
each observable that was passed to ``__init__``.

.. _sweeps:

4. Parameter Sweeps
--------------------

Use ``run_parameter_sweep`` to explore uncertainty in the initial rotation period or to
visualize the range of evolutionary tracks. It distributes ``LXUV_model`` calls across
all available CPU cores via ``multiprocessing.Pool``.

.. code-block:: python

   nsamp = 20

   mass_samp  = np.random.normal(0.089, 0.007, nsamp)
   prot_samp  = np.random.uniform(0.01, 5.0, nsamp)   # broad initial Prot prior
   age_samp   = np.ones(nsamp) * 7.6
   beta1_samp = np.random.normal(j21["beta1"][0], j21["beta1"][1], nsamp)
   beta2_samp = np.random.normal(j21["beta2"][0], j21["beta2"][1], nsamp)
   Rosat_samp = np.random.normal(j21["Rosat"][0], j21["Rosat"][1], nsamp)
   RXsat_samp = np.random.normal(j21["RXsat"][0], j21["RXsat"][1], nsamp)

   thetas = np.array([
       mass_samp, prot_samp, age_samp,
       beta1_samp, beta2_samp, Rosat_samp, RXsat_samp,
   ]).T   # shape (nsamp, 7)

   evols = model.run_parameter_sweep(thetas)

**Plotting methods:**

``plot_evolution(evols)`` — single panel showing X-ray luminosity vs. time for all tracks,
with the observed value as a red dashed line and shaded 1-sigma band.

.. code-block:: python

   fig = model.plot_evolution(evols, show=True)
   fig.savefig("xray_evolution.png")

``plot_evolution_multi(evols)`` — three-panel figure showing bolometric luminosity, X-ray
luminosity, and rotation period vs. time, with all applicable observational constraints
overplotted. A vertical band marks the age constraint.

.. code-block:: python

   fig = model.plot_evolution_multi(evols, show=True)
   fig.savefig("multi_evolution.png")

.. _new-star:

5. Configuring ``run_alabi.py`` for a New Star
------------------------------------------------

The section at the top of ``run_alabi.py`` marked ``CHANGE TO THE DATA FOR YOUR STAR``
is the only part you need to edit. Three example stars are already provided:

.. code-block:: python

   # TRAPPIST-1 — XUV and rotation measured
   # star_name = "Trappist-1"
   # mass_data = np.array([0.08,    0.007])  * u.Msun
   # Lbol_data = np.array([5.22e-4, 0.19e-4]) * u.Lsun
   # Lxuv_data = np.array([1.0e-4,  0.1e-4]) * Lbol_data
   # Prot_data = np.array([3.295,   0.003])  * u.day
   # age_data  = np.array([7.6,     2.2])    * u.Gyr

   # GJ 3470 — X-ray measured, no XUV
   # star_name  = "GJ_3470"
   # mass_data  = np.array([0.51,    0.06])    * u.Msun
   # Lbol_data  = np.array([0.029,   0.002])   * u.Lsun
   # Lxray_data = np.array([4.43e27, 7.88e26]) * u.erg/u.s
   # Prot_data  = np.array([21.54,   0.49])    * u.day

   # 55 Cnc — current default
   star_name  = "55_Cnc"
   mass_data  = np.array([0.905,    0.015])   * u.Msun
   Lbol_data  = np.array([0.582,    0.014])   * u.Lsun
   Lxray_data = np.array([6.05e26,  5.23e25]) * u.erg / u.s
   Prot_data  = np.array([38.8,     0.05])    * u.day
   age_data   = np.array([10.2,     2.5])     * u.Gyr

The rest of the script works unchanged because ``StellarEvolutionModel`` automatically
adapts its chi-squared to whichever observables were provided.

.. tip::

   If you only have X-ray data (not XUV), pass ``Lxray_data`` instead of ``Lxuv_data``.
   The chi-squared will use ``final.star.LXRAY`` rather than ``final.star.LXUV``.

.. _constraints:

6. Testing Different Constraint Combinations
---------------------------------------------

A key scientific question is: *which observables most strongly constrain the XUV history?*
``run_alabi.py`` initializes four model variants:

.. code-block:: python

   model1 = StellarEvolutionModel(star_name=star_name,
                                  Lbol_data=Lbol_data,
                                  Lxray_data=Lxray_data)       # Lbol + Lxray

   model2 = StellarEvolutionModel(star_name=star_name,
                                  Lbol_data=Lbol_data,
                                  Prot_data=Prot_data)          # Lbol + Prot

   model3 = StellarEvolutionModel(star_name=star_name,
                                  Lbol_data=Lbol_data,
                                  Lxray_data=Lxray_data,
                                  Prot_data=Prot_data)          # Lbol + Lxray + Prot

   model4 = StellarEvolutionModel(star_name=star_name,
                                  Lxray_data=Lxray_data)        # Lxray only

To run inference for a specific model, set the ``test`` variable near the bottom of the script:

.. code-block:: python

   test  = "model1"   # or "model2", "model3", "model4"
   model = eval(test)
   save_dir = f"results/{star_name}/{test}/"

Results for each configuration are saved to separate subdirectories, so you can run all
four and compare posteriors.

.. _prior-likelihood:

7. Prior and Likelihood Setup
------------------------------

**Prior**

The prior is specified as a list of ``(mean, std)`` tuples. ``None`` means a flat (uniform)
prior within the corresponding ``bounds`` entry.

.. code-block:: python

   prior_data = [
       (mass_data[0].value, mass_data[1].value),  # mass: Gaussian
       (None, None),                               # Prot_initial: uniform
       (None, None),                               # age: uniform
       (-0.135, 0.030),                            # beta1: Gaussian (Johnstone+2021)
       (-1.889, 0.079),                            # beta2: Gaussian
       (0.0605, 0.00331),                          # Rsat:  Gaussian
       (5.135e-4, 3.320e-5),                       # RXsat: Gaussian
   ]

Mass bounds are set to ±5σ from the observed mass, clipped to the valid stellar grid
range [0.07, 1.4] M☉. Activity law parameters are bounded at ±5σ from Johnstone+2021.

.. code-block:: python

   sigma_factor = 5
   min_mass = max(mass_data[0].value - sigma_factor * mass_data[1].value, 0.07)
   max_mass = min(mass_data[0].value + sigma_factor * mass_data[1].value, 1.4)

   bounds = [
       (min_mass, max_mass),   # mass
       (0.1, 12.0),            # Prot_initial [days]
       (0.5, 5.0),             # age [Gyr]
       (min_beta1, max_beta1), # beta1
       (min_beta2, max_beta2), # beta2
       (min_Rsat,  max_Rsat),  # Rsat
       (min_RXsat, max_RXsat), # RXsat
   ]

Three prior function signatures are prepared — one per sampler format:

.. code-block:: python

   from functools import partial
   from alabi import utility as ut

   # alabi: draws samples from the prior for GP training
   ps = partial(ut.prior_sampler_normal, prior_data=prior_data, bounds=bounds)

   # emcee: evaluates log-prior at a point
   lnprior = partial(ut.lnprior_normal, bounds=bounds, data=prior_data)

   # dynesty: maps unit-hypercube to parameter space
   prior_transform = partial(ut.prior_transform_normal, bounds=bounds, data=prior_data)

**Likelihood**

The log-likelihood is a Gaussian chi-squared:

.. math::

   \ln \mathcal{L}(\theta) = -\frac{1}{2} \sum_i \chi^2_i
   = -\frac{1}{2} \sum_i \left(\frac{m_i(\theta) - d_i}{\sigma_i}\right)^2

.. code-block:: python

   def lnlike(theta):
       _ = model.LXUV_model(theta)
       chi2_array = model.compute_chi_squared_fit()
       lnl = -0.5 * np.sum(chi2_array)
       return lnl

``LXUV_model`` stores the result in ``model.evol``, then ``compute_chi_squared_fit``
reads from it to compute :math:`\chi^2_i` for each observable initialized at construction.

.. _alabi-run:

8. Running ``alabi`` Surrogate MCMC
-------------------------------------

The full pipeline in ``run_alabi.py``:

.. code-block:: python

   from alabi import SurrogateModel

   sm = SurrogateModel(
       fn=lnlike,
       bounds=bounds,
       prior_sampler=ps,
       savedir=save_dir,
       cache=True,      # save GP and training set to disk after each iteration
       labels=labels,
       scale=None,
       ncore=22,        # parallel VPLanet processes
   )

   # 1. Draw initial training set and fit the first GP
   sm.init_samples(ntrain=200, ntest=100, reload=False)
   sm.init_gp(kernel="ExpSquaredKernel", fit_amp=False, fit_mean=True, white_noise=-15)

   # 2. Active learning: add 500 points using BAPE
   sm.active_train(niter=500, algorithm="bape", gp_opt_freq=10)
   sm.plot(plots=["gp_all"])

   # 3. MCMC on the trained surrogate
   sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=int(1e5), opt_init=False)
   sm.plot(plots=["emcee_corner"])

   sm.run_dynesty(ptform=prior_transform, mode='dynamic')
   sm.plot(plots=["dynesty_all"])

**Key** ``SurrogateModel`` **arguments:**

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Argument
     - Description
   * - ``fn``
     - Function to emulate — here ``lnlike``, returning a scalar
   * - ``bounds``
     - Hard parameter bounds (7-element list of tuples)
   * - ``prior_sampler``
     - Callable that draws prior samples to seed the GP training set
   * - ``savedir``
     - Directory for caching the training set and GP model
   * - ``cache``
     - Save progress after each active-learning iteration
   * - ``ncore``
     - Parallel workers for forward model evaluation
   * - ``scale``
     - GP output scaling; use ``"nlog"`` for a log-posterior, ``None`` for log-likelihood

**Active learning arguments:**

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Argument
     - Description
   * - ``niter``
     - Number of new training points to add (500 is a good starting point)
   * - ``algorithm``
     - ``"bape"`` — Bayesian Active Posterior Estimation (recommended)
   * - ``gp_opt_freq``
     - Re-optimize GP hyperparameters every N iterations

.. _reload:

9. Reloading and Extending a Run
----------------------------------

Because ``cache=True``, all training evaluations are saved to ``save_dir`` after each
iteration. Reload a completed run and continue without repeating any forward model calls:

.. code-block:: python

   import alabi

   sm = alabi.cache_utils.load_model_cache(save_dir)
   sm.savedir = save_dir   # re-set if save_dir path has changed

   # Continue active learning from where it left off
   sm.active_train(niter=500, algorithm="bape", gp_opt_freq=10)
   sm.plot(plots=["gp_all"])

   # Run MCMC without any additional training
   sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=int(1e5), opt_init=False)
   sm.plot(plots=["emcee_corner"])

This is especially useful on HPC clusters where you want to checkpoint progress or add
more training iterations after reviewing the GP diagnostics.

----

Summary
-------

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Step
     - Code location
     - Key action
   * - Configure star data
     - Top of ``run_alabi.py``
     - Set ``star_name``, ``mass_data``, luminosity / rotation / age data
   * - Choose constraints
     - ``model1``–``model4`` block
     - Pass different combinations of data to ``StellarEvolutionModel``
   * - Set active model
     - ``test = "model1"`` line
     - Select which constraint set to run
   * - Run single model
     - ``model.LXUV_model(theta)``
     - Inspect output; call ``compute_chi_squared_fit()``
   * - Run parameter sweep
     - ``model.run_parameter_sweep(thetas)``
     - Parallel forward models; visualize with ``plot_evolution_multi()``
   * - Run surrogate MCMC
     - ``run_alabi.py`` main block
     - ``init_samples`` → ``init_gp`` → ``active_train`` → MCMC
   * - Reload saved run
     - ``alabi.cache_utils.load_model_cache(save_dir)``
     - Continue or run MCMC on a completed GP
