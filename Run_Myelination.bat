@echo off
rem ==========================================================================
rem  Double-click launcher for Myelination.py (no .exe needed).
rem  Keep this file in the same folder as Myelination.py. For a desktop icon:
rem  right-click this file > Send to > Desktop (create shortcut).
rem
rem  It uses python_env if build_app.bat has made one ("build_app.bat setup"
rem  installs the libraries without building the .exe), otherwise the Python
rem  on PATH. With Anaconda instead: delete "rem " from the start of the CALL
rem  line below and put your environment's name in place of myenv.
rem ==========================================================================

cd /d "%~dp0"
set "PY=python"
if exist "%~dp0python_env\Scripts\python.exe" set "PY=%~dp0python_env\Scripts\python.exe"

rem call "%USERPROFILE%\anaconda3\Scripts\activate.bat" myenv

"%PY%" Myelination.py %*
if errorlevel 1 pause
