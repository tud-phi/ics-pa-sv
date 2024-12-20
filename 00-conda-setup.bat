@echo off

REM install required conda packages into conda environment ics
call conda env create -f environment.yml

REM install pip packages
call conda run -n ics .\01-pip-setup.bat
