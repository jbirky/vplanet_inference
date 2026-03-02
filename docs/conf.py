import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "vplanet_inference"
author = "Jessica Birky"
copyright = "2024, Jessica Birky"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

autodoc_member_order = "bysource"
# Only mock packages not installable in a standard docs environment.
# When building locally, install the package first: pip install -e ..
autodoc_mock_imports = ["alabi", "vplanet"]

# Use pre-executed notebook outputs; do not re-run notebooks during the build
nbsphinx_execute = "never"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "renku"
html_static_path = ["_static"]

html_theme_options = {}
