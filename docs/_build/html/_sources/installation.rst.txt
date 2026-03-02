Installation
============

Requirements
------------

``vplanet_inference`` requires Python 3.7+ and the following dependencies (installed automatically):

- `numpy <https://numpy.org>`_
- `matplotlib <https://matplotlib.org>`_ >= 2.0
- `scipy <https://scipy.org>`_
- `astropy <https://www.astropy.org>`_
- `emcee <https://emcee.readthedocs.io>`_ >= 3.0
- `dynesty <https://dynesty.readthedocs.io>`_
- `corner <https://corner.readthedocs.io>`_
- `seaborn <https://seaborn.pydata.org>`_
- `SALib <https://salib.readthedocs.io>`_
- `george <https://george.readthedocs.io>`_
- `scikit-learn <https://scikit-learn.org>`_
- `h5py <https://www.h5py.org>`_
- `tqdm <https://tqdm.github.io>`_
- `vplanet <https://github.com/VirtualPlanetaryLaboratory/vplanet>`_ >= 2.0.6

Setting up a conda environment
-------------------------------

It is recommended to install ``vplanet_inference`` in a dedicated conda environment
to avoid dependency conflicts. If you don't have conda, install it via
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
`Anaconda <https://www.anaconda.com>`_.

Create and activate a new environment with Python 3.7+:

.. code-block:: bash

   conda create -n vpi python=3.10
   conda activate vpi

Installing vplanet_inference
----------------------------

Clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/jbirky/vplanet_inference
   cd vplanet_inference
   pip install -e .

Or install using ``setup.py``:

.. code-block:: bash

   git clone https://github.com/jbirky/vplanet_inference
   cd vplanet_inference
   python setup.py install

Optional dependency: alabi
--------------------------

The ``AnalyzeVplanetModel`` class supports surrogate-model-accelerated MCMC via the
`alabi <https://github.com/jbirky/alabi>`_ package. To enable this functionality:

.. code-block:: bash

   git clone https://github.com/jbirky/alabi
   cd alabi
   python setup.py install

Verifying the installation
--------------------------

.. code-block:: python

   import vplanet_inference as vpi
   print(vpi.INFILE_DIR)   # path to bundled template infiles
