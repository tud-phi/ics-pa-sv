#!/bin/sh

# install pip packages
pip install . --upgrade

# trust all jupyter notebooks
# for example source/problem_2/task_2a-1_pd_control.ipynb
jupyter trust ./*/*/*.ipynb
