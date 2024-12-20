#!/bin/sh
# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

# check for external conda being available
if ! which conda; then
    echo "No conda found. Please install miniconda3 and add it to your PATH."
fi

# install required conda packages into conda environment ics
conda env create -f environment.yml
# you can verify with conda env list

# install pip packages
conda run -n ics ./01-pip-setup.sh
