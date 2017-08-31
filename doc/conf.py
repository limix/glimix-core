from __future__ import unicode_literals

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'glimix-core'
copyright = '2016, Danilo Horta'
author = 'Danilo Horta'
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'default'
htmlhelp_basename = 'glimix-coredoc'
latex_elements = {}
latex_documents = [
    (master_doc, 'glimix-core.tex', 'glimix-core Documentation',
     'Danilo Horta', 'manual'),
]
man_pages = [(master_doc, 'glimix-core', 'glimix-core Documentation', [author],
              1)]
texinfo_documents = [
    (master_doc, 'glimix-core', 'glimix-core Documentation', author,
     'glimix-core', 'One line description of project.', 'Miscellaneous'),
]
intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'optimix': ('http://optimix.readthedocs.io/en/latest/', None)
}
