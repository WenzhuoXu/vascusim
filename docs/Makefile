# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*

# Create directory structure for documentation
setup:
	mkdir -p $(SOURCEDIR)/api
	mkdir -p $(SOURCEDIR)/user_guide
	mkdir -p $(SOURCEDIR)/examples
	mkdir -p $(SOURCEDIR)/development
	mkdir -p $(SOURCEDIR)/_static/css
	touch $(SOURCEDIR)/_static/css/custom.css

# Copy examples from code to documentation
copy-examples:
	mkdir -p $(SOURCEDIR)/examples/code
	cp -r ../examples/* $(SOURCEDIR)/examples/code/

# API documentation auto-generation
apidoc:
	sphinx-apidoc -o $(SOURCEDIR)/api_auto ../vascusim/ ../vascusim/_version.py

# Shortcut for generating HTML
html-quick: clean setup apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)