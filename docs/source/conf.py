import sys
import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Soot Particle'
copyright = '2024, Computational Engineering and Design, Fraunhofer-Chalmers Centre for Industrial Mathematics, Chalmers Science Park, Gothenburg, SE-412 88, Sweden'
author = 'Benjamin Elm Jonsson, Niclas Persson'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Specify project path
sys.path.insert(0, os.path.abspath('..'))
sys.path.append("C:/Users/NiclasPersson/OneDrive - Fraunhofer-Chalmers Centre/soot-particle")
