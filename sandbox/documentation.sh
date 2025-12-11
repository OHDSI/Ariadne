# Code for setting up and deploying MkDocs documentation

pip install mkdocs mkdocs-material
pip install pymdown-extensions
pip install "mkdocstrings[python]"
pip install mkdocs-jupyter

# To serve the documentation locally
mkdocs serve

# To build and deploy the documentation to GitHub Pages
mkdocs gh-deploy