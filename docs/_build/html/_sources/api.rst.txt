API Reference
=============

.. currentmodule:: vplanet_inference

vplanet_inference.model
-----------------------

.. autoclass:: vplanet_inference.VplanetModel
   :members:
   :undoc-members:
   :show-inheritance:

----

vplanet_inference.analysis
--------------------------

.. autofunction:: vplanet_inference.sort_yaml_key

.. autoclass:: vplanet_inference.AnalyzeVplanetModel
   :members:
   :undoc-members:
   :show-inheritance:

----

vplanet_inference.parameters
-----------------------------

.. autoclass:: vplanet_inference.VplanetParameters
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: vplanet_inference.get_vplanet_units

.. autofunction:: vplanet_inference.get_vplanet_units_source

.. autofunction:: vplanet_inference.check_units

----

vplanet_inference.fischer
--------------------------

.. autofunction:: vplanet_inference.compute_fisher_information

.. autofunction:: vplanet_inference.analyze_fisher_information

----

vplanet_inference.info_vis
---------------------------

Local Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vplanet_inference.local_sensitivity_analysis

.. autofunction:: vplanet_inference.plot_sensitivity_heatmap

Fisher Information Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vplanet_inference.plot_fisher_heatmap

.. autofunction:: vplanet_inference.plot_correlation_matrix

.. autofunction:: vplanet_inference.plot_uncertainty_ellipse

.. autofunction:: vplanet_inference.plot_eigenvalue_spectrum

.. autofunction:: vplanet_inference.plot_standard_errors

.. autofunction:: vplanet_inference.plot_condition_number_breakdown

.. autofunction:: vplanet_inference.create_fisher_dashboard

Posterior Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vplanet_inference.plot_corner_probability
