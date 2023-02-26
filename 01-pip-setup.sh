#!/bin/sh

# install pip packages
pip install . --upgrade

# install jupyter notebook extensions for nbgrader
jupyter nbextension install --sys-prefix --py nbgrader --overwrite

# trust all jupyter notebooks
# for example source/problem_2/task_2a-1_pd_control.ipynb
jupyter trust ./*/*/*.ipynb
