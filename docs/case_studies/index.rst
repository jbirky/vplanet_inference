Case Studies Overview
=====================

End-to-end science applications of ``vplanet_inference`` on real research problems.
Each case study is available as a downloadable Jupyter notebook.

----

.. list-table::
   :widths: 5 40 55
   :header-rows: 1

   * - #
     - Notebook
     - Description
   * - 4
     - :doc:`04_earth_mle` :download:`↓ <04_earth_mle.ipynb>`
     - Recover Earth's interior initial conditions from present-day geophysical observations
       using maximum likelihood estimation (MLE). Covers VPLanet's ``thermint`` and ``radheat``
       modules, Gaussian log-likelihood construction, single and multi-start Nelder-Mead
       optimization, local sensitivity analysis, and Fisher information matrix computation.
       Based on Gilbert-Janizek et al. (2026).
   * - 5
     - :doc:`05_custom_rotation_model` 
     - Fit a custom rotation-activity-XUV model (Johnstone et al. 2021) to stellar
       observational constraints using the ``alabi`` surrogate MCMC framework. Covers
       implementing a custom likelihood with ``StellarEvolutionModel``, testing different
       combinations of luminosity, X-ray, rotation, and age constraints, and running
       active-learning Bayesian inference with ``emcee`` and ``dynesty``.
