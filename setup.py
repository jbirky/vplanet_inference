from setuptools import setup

setup(name="vplanet_inference",
      version="0.0.1",
      description="Python tools for statistical inference with VPLanet",
      author="Jessica Birky",
      author_email="jbirky@uw.edu",
      license = "MIT",
      url="https://github.com/jbirky/vplanet_inference",
      packages=["vplanet_inference"],
      install_requires = ["numpy",
                          "matplotlib >= 2.0.0",
                          "seaborn",
                          "SALib",
                          "scipy",
                          "george",
                          "emcee >= 3.0",
                          "dynesty",
                          "corner",
                          "scikit-learn",
                          "pybind11",
                          "pytest",
                          "h5py",
                          "tqdm",
                          "vplanet >= 2.0.6"]
     )
