Quickstart
==========

This page walks through the core workflows of ``vplanet_inference``, from running a single
forward model to setting up full Bayesian inference.

.. code-block:: python

   import vplanet_inference as vpi
   import numpy as np
   import astropy.units as u

Running a forward model
-----------------------

:class:`~vplanet_inference.VplanetModel` wraps a VPLanet simulation. You specify which
input parameters to vary, their units, which outputs to collect, and where to read/write
the infiles.

**1. Point to a template infile directory**

``vplanet_inference`` ships with several template infile sets under ``vpi.INFILE_DIR``.
You can also supply your own:

.. code-block:: python

   import os
   inpath = os.path.join(vpi.INFILE_DIR, "stellar/")

**2. Declare input and output parameters with units**

Input parameters follow the VPLanet ``<body>.<parameter>`` naming convention.
``"vpl.*"`` parameters belong to ``vpl.in``; all other prefixes correspond to body infiles:

.. code-block:: python

   inparams = {
       "star.dMass":     u.Msun,   # stellar mass [solar masses]
       "vpl.dStopTime":  u.Gyr,    # simulation duration [Gyr]
   }

Output parameters are addressed as ``"final.<body>.<quantity>"`` or
``"initial.<body>.<quantity>"``:

.. code-block:: python

   outparams = {
       "final.star.Radius":     u.Rsun,   # final stellar radius [solar radii]
       "final.star.Luminosity": u.Lsun,   # final stellar luminosity [solar luminosities]
   }

**3. Initialize the model and run it**

.. code-block:: python

   vpm = vpi.VplanetModel(
       inparams=inparams,
       outparams=outparams,
       inpath=inpath,
       outpath="output/",
   )

   theta = np.array([1.0, 2.5])   # [M_sun, Gyr]
   output = vpm.run_model(theta)
   print(output)   # array of final values in the units declared above

Tracking time evolution
-----------------------

Pass ``timesteps`` to record output at regular intervals throughout the simulation:

.. code-block:: python

   vpm = vpi.VplanetModel(
       inparams=inparams,
       outparams=outparams,
       inpath=inpath,
       outpath="output/",
       timesteps=1e7 * u.yr,   # record every 10 Myr
   )

   output = vpm.run_model(theta)
   # output is now a dict: {"Time": ..., "final.star.Radius": ..., ...}

   time = output["Time"]
   radius_evol = output["final.star.Radius"]

Two-body tidal evolution example
---------------------------------

The snippet below models tidal spin-down and circularisation for a stellar binary using
the CTL tidal model:

.. code-block:: python

   inpath = os.path.join(vpi.INFILE_DIR, "stellar_eqtide/ctl")

   inparams = {
       "primary.dMass":       u.Msun,
       "secondary.dMass":     u.Msun,
       "primary.dRotPeriod":  u.day,
       "secondary.dRotPeriod": u.day,
       "primary.dTidalTau":   u.dex(u.s),
       "secondary.dTidalTau": u.dex(u.s),
       "secondary.dEcc":      u.dimensionless_unscaled,
       "secondary.dOrbPeriod": u.day,
       "vpl.dStopTime":       u.Gyr,
   }

   outparams = {
       "final.primary.RotPer":        u.day,
       "final.secondary.RotPer":      u.day,
       "final.secondary.OrbPeriod":   u.day,
       "final.secondary.Eccentricity": u.dimensionless_unscaled,
   }

   vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams,
                          timesteps=1e7 * u.yr, time_init=1e6 * u.yr)

   theta = np.array([1.08, 1.07, 5.0, 5.0, -1.0, -1.0, 0.5, 6.0, 5.0])
   output = vpm.run_model(theta)

