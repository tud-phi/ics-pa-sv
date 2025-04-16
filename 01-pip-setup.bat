@echo off

REM install pip packages
call pip install . --upgrade

REM trust all jupyter notebooks
for /r %%i in (*.ipynb) do call jupyter trust "%%i"
