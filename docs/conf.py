"""Sphinx configuration for the PairMatch documentation."""

from importlib.metadata import version as _get_version

# -- Project information -----------------------------------------------------

project = "PairMatch"
copyright = "2026, Bob Wilson"
author = "Bob Wilson"
release = _get_version("PairMatch")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # `make doctest` executes every `>>>` example in the guides, so a number
    # quoted from the paper cannot silently drift from what the library computes.
    "sphinx.ext.doctest",
    # Figures are regenerated from code on every build rather than committed.
    "matplotlib.sphinxext.plot_directive",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# The guides' doctests share one running example.
doctest_global_setup = """
import math

from scipy.stats import binom

from pair_match import (
    PairedOutcomeTable,
    attributable_effect_interval,
    design_sensitivity_binary,
    mcnemar_ate_interval,
    sensitivity_value,
    worst_case_pvalue,
)

table = PairedOutcomeTable(s00=800, s01=30, s10=70, s11=100)
"""

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 150)]
plot_rcparams = {"figure.figsize": (6.4, 3.6)}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"PairMatch {release}"
html_theme_options = {
    "github_url": "https://github.com/rwilson4/PairMatch",
    "show_prev_next": False,
}
