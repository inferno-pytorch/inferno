#!/bin/bash
cd docs
rm -r -f inferno-apidoc
sphinx-apidoc -o inferno-apidoc ../inferno
make html
cd ..