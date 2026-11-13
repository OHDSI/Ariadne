# Code for setting up and deploying MkDocs documentation

pip install mkdocs mkdocs-material
pip install mkdocstrings mkdocstrings-python
pip install pymdown-extensions
pip install "mkdocstrings[python]"
pip install mkdocs-jupyter
pip install mkdocs-autoapi
pip install mkdocs-autoapi

# To serve the documentation locally
mkdocs serve

# To build and deploy the documentation to GitHub Pages
mkdocs gh-deploy