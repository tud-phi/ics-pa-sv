#!/usr/bin/env bash
# deactivate jupyter notebook extensions
# jupyter nbextension disable --sys-prefix --py nbgrader
# jupyter serverextension disable --sys-prefix --py nbgrader

# run jupyter notebook and allow to access the notebook from a remote machine
jupyter notebook
