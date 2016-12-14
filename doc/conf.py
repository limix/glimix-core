import sphinx_rtd_theme

try:
    import limix_inference
    version = limix_inference.__version__
except ImportError:
    version = 'unknown'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'limix-inference'
copyright = '2016, Danilo Horta'
author = 'Danilo Horta'
release = version
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes", ]
htmlhelp_basename = 'limix-inferencedoc'
latex_elements = {}
latex_documents = [
    (master_doc, 'limix-inference.tex', 'limix-inference Documentation',
     'Danilo Horta', 'manual'),
]
man_pages = [
    (master_doc, 'limix-inference', 'limix-inference Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'limix-inference', 'limix-inference Documentation',
     author, 'limix-inference', 'One line description of project.',
     'Miscellaneous'),
]
intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}
