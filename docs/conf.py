import os
import sys


# Add the parent directory to the Python path
# This allows Sphinx to find and import the package for documentation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configuration file for the Sphinx documentation builder.
project = 'CausalExplain'
copyright = '2024, J. Renero'
author = 'J. Renero'
release = '0.6.3'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.autosummary',
    'myst_parser',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'private-members': False,
    'exclude-members': (
        '__weakref__,get_metadata_routing,set_output,set_fit_request,'
        'set_predict_request,set_transform_request,set_score_request,'
        'set_metadata_routing'
    ),
}

# Autodoc type hints settings
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_type_aliases = {}

# Autosummary settings
autosummary_generate = True
add_module_names = False

# Numpydoc settings
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Add any paths that contain templates here
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'pydata_sphinx_theme'

# PyData theme options
html_theme_options = {
    "github_url": "https://github.com/renero/causalexplain",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_nav_level": 1,
    "collapse_navigation": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/renero/causalexplain",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "sidebar-ethical-ads"]
}

html_context = {
    "default_mode": "light",
    "doc_path": "docs",
    "github_user": "renero",
    "github_repo": "causalexplain",
    "github_version": "main",
}

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'js/custom.js',
]

# GitHub Pages specific settings
html_baseurl = 'https://renero.github.io/causalexplain/'
