vplanet_inference
=================

Python tools for statistical inference with `VPLanet <https://github.com/VirtualPlanetaryLaboratory/vplanet>`_ — a planetary system evolution simulator.

``vplanet_inference`` provides a high-level Python interface for:

- Running VPLanet forward models with unit-aware parameter substitution
- Performing parameter sweeps and sensitivity analysis
- Running Bayesian inference (MCMC / nested sampling) constrained by observational data

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   tutorials/index
   tutorials/01_vplanet_infiles
   tutorials/02_sensitivity_analysis
   tutorials/03_mcmc_custom_likelihood

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
