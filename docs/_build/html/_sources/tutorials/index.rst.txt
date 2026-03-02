Tutorial Overview
=========

Step-by-step guides to common ``vplanet_inference`` workflows.
Each tutorial is available as a downloadable Jupyter notebook.

----

.. list-table::
   :widths: 5 40 55
   :header-rows: 1

   * - #
     - Notebook
     - Description
   * - 1
     - :doc:`01_vplanet_infiles` :download:`↓ <01_vplanet_infiles.ipynb>`
     - Set up VPLanet input files, run a stellar evolution model from the command line,
       understand the output files, and run the same model through ``vplanet_inference``
       with unit-aware input/output parameters.
   * - 2
     - :doc:`02_sensitivity_analysis` :download:`↓ <02_sensitivity_analysis.ipynb>`
     - Run a Sobol variance-based global sensitivity analysis on a binary star tidal evolution
       model to identify which input parameters drive the most variance in the outputs.
       Covers the SALib workflow, parallel model evaluation, and heatmap visualization,
       as well as the YAML-driven ``AnalyzeVplanetModel`` convenience class.
   * - 3
     - :doc:`03_mcmc_custom_likelihood` :download:`↓ <03_mcmc_custom_likelihood.ipynb>`
     - Infer TRAPPIST-1 stellar parameters from observed luminosities using a custom
       likelihood function. Covers direct MCMC with ``dynesty`` and ``emcee``,
       the ``alabi`` surrogate model approach for expensive forward models,
       and advanced likelihood patterns (Student-t, multi-observable, failed-run handling).
