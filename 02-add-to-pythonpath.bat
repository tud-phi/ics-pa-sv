@echo off
REM absolute path of this script file
set "THIS_DIR=%~dp0"

REM use 'source' for TA's (they have the 'source' dir), 'release' or `assignment` (for students)
if exist "%THIS_DIR%source" (
    set "CODE_DIR=%THIS_DIR%source"
) else if exist "%THIS_DIR%release" (
    set "CODE_DIR=%THIS_DIR%release"
) else if exist "%THIS_DIR%assignment" (
    set "CODE_DIR=%THIS_DIR%assignment"
)

echo Adding %CODE_DIR% to PYTHONPATH
set "PYTHONPATH=%CODE_DIR%;%PYTHONPATH%"
