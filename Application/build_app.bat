@echo off
setlocal
rem ==========================================================================
rem  build_app.bat - installs everything and builds Myelination.exe
rem
rem  Put this file next to Myelination.py and double-click it. It will:
rem    1. find Python 3.10-3.14 (and offer to install Python 3.12 if none)
rem    2. create a private Python environment, python_env, next to this file
rem    3. install PyTorch (GPU or CPU build)
rem    4. install every other library the pipeline uses, plus PyInstaller
rem    5. build dist\Myelination\Myelination.exe
rem
rem  Needs an internet connection and about 10 GB of free disk space.
rem  The first run takes 15-40 minutes; later runs reuse python_env and only
rem  rebuild the app.
rem
rem    build_app.bat          install what is missing, then build
rem    build_app.bat setup    install only - no .exe (for Run_Myelination.bat)
rem    build_app.bat clean    delete python_env, reinstall everything, build
rem ==========================================================================

rem ---- settings you may change ---------------------------------------------
rem  PyTorch build for NVIDIA GPUs: cu126 (older drivers), cu128, cu130 (newest)
set "TORCH_CUDA=cu128"
rem  Libraries besides PyTorch. To pin one that a new release breaks, write it
rem  with a version, e.g. transformers==X.Y.Z (the version that last worked).
set "PACKAGES=numpy pillow scipy scikit-image readlif transformers accelerate pyinstaller pyinstaller-hooks-contrib"
rem --------------------------------------------------------------------------

cd /d "%~dp0"
set "ENV_DIR=%~dp0python_env"
set "PY=%ENV_DIR%\Scripts\python.exe"
set "MODE=%~1"

if not exist "Myelination.py" goto :no_script

if /i "%MODE%"=="clean" if exist "%ENV_DIR%" (
    echo Deleting the old python_env...
    rmdir /s /q "%ENV_DIR%"
)

rem ---- 1-2. Python and the private environment -------------------------------
echo.
echo [1/5] Python
if exist "%PY%" goto :have_env

set "BASEPY="
rem  Prefer a Python version PyTorch is known to support, via the py launcher
for %%V in (3.12 3.13 3.11 3.10 3.14) do (
    py -%%V -c "import tkinter" >nul 2>&1 && (set "BASEPY=py -%%V" & goto :have_python)
)
rem  Otherwise the python on PATH (e.g. Anaconda)
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1
if errorlevel 1 goto :no_python
python -c "import tkinter" >nul 2>&1
if errorlevel 1 goto :no_tkinter
set "BASEPY=python"

:have_python
echo Using %BASEPY%
echo.
echo [2/5] Creating python_env (private to this app; your other Python setups are not touched)...
%BASEPY% -m venv "%ENV_DIR%"
if errorlevel 1 goto :fail
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
goto :env_ready

:have_env
echo.
echo [2/5] Reusing python_env  (run "build_app.bat clean" to start it afresh)
:env_ready
"%PY%" -c "import sys; print('Python', sys.version.split()[0], 'at', sys.prefix)"

rem ---- 3. PyTorch -------------------------------------------------------------
echo.
echo [3/5] PyTorch
nvidia-smi -L >nul 2>&1
if errorlevel 1 goto :no_gpu
echo An NVIDIA GPU was found on this PC.
choice /c GC /t 30 /d G /m "GPU build [G] - runs on any PC, uses the GPU where there is one - or CPU-only build [C]? G in 30 s"
goto :got_choice
:no_gpu
echo No NVIDIA GPU was found on this PC.
choice /c GC /t 30 /d C /m "GPU build [G] - for using the app on GPU PCs - or CPU-only build [C]? C in 30 s"
:got_choice
if errorlevel 2 (set "TORCH_VARIANT=cpu") else (set "TORCH_VARIANT=%TORCH_CUDA%")

set "OLD_VARIANT="
if exist "%ENV_DIR%\torch_variant.txt" set /p OLD_VARIANT=<"%ENV_DIR%\torch_variant.txt"
if not defined OLD_VARIANT goto :install_torch
if /i "%OLD_VARIANT%"=="%TORCH_VARIANT%" goto :torch_done
echo Switching PyTorch from %OLD_VARIANT% to %TORCH_VARIANT%...
"%PY%" -m pip uninstall -y torch
:install_torch
echo Installing PyTorch (%TORCH_VARIANT%) - the GPU build is a 2-3 GB download...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/%TORCH_VARIANT%
if errorlevel 1 goto :torch_fail
> "%ENV_DIR%\torch_variant.txt" echo %TORCH_VARIANT%
:torch_done
echo PyTorch %TORCH_VARIANT% is installed.

rem ---- 4. Other libraries ------------------------------------------------------
echo.
echo [4/5] Installing the other libraries...
"%PY%" -m pip install %PACKAGES%
if errorlevel 1 goto :fail
echo.
echo Checking everything imports...
"%PY%" -c "import numpy, PIL, scipy, skimage, tkinter, readlif, transformers, accelerate, PyInstaller, torch; print('  PyTorch', torch.__version__, '- CUDA', torch.version.cuda or 'none (CPU build)', '- GPU usable on this PC:', torch.cuda.is_available()); print('  transformers', transformers.__version__, '- numpy', numpy.__version__, '- scikit-image', skimage.__version__)"
if errorlevel 1 goto :fail

if /i "%MODE%"=="setup" goto :setup_done

rem ---- 5. Build -----------------------------------------------------------------
echo.
echo [5/5] Building the app (5-20 minutes)...
rem  --onedir  : a folder with the .exe inside; starts fast (a one-file .exe
rem              would unpack several GB of PyTorch on every launch)
rem  --console : keep the console window - progress, the run summary and any
rem              error messages print there
rem  transformers loads model code on demand, so the backbone families offered
rem  on the Training tab are listed explicitly. To train with a different
rem  backbone, add  --collect-submodules transformers.models.<family>  below.
rem  --recursive-copy-metadata : transformers checks its dependencies' versions
rem              when it starts, so their details go in too - whichever ones
rem              the installed version actually uses.
"%PY%" -m PyInstaller --noconfirm --clean --onedir --console --name Myelination ^
  --collect-all skimage ^
  --collect-all readlif ^
  --collect-data transformers ^
  --collect-submodules transformers.models.vit ^
  --collect-submodules transformers.models.convnext ^
  --collect-submodules transformers.models.resnet ^
  --collect-submodules transformers.models.dinov2 ^
  --recursive-copy-metadata transformers ^
  --recursive-copy-metadata accelerate ^
  --exclude-module tensorflow --exclude-module keras ^
  --exclude-module jax --exclude-module flax ^
  Myelination.py
if errorlevel 1 goto :fail

rem  Carry existing settings over to the app, if there are any
if exist wrap_pipeline_settings.json copy /y wrap_pipeline_settings.json dist\Myelination\ >nul

echo.
echo ==========================================================================
echo  Done:  %~dp0dist\Myelination\Myelination.exe   (PyTorch: %TORCH_VARIANT%)
echo.
echo  Keep the whole dist\Myelination folder together - the .exe needs the
echo  _internal folder next to it. Copy the folder anywhere, then right-click
echo  Myelination.exe ^> Send to ^> Desktop (create shortcut).
echo.
echo  python_env is only needed for building. Keep it to make rebuilds fast,
echo  or delete it to free several GB.
echo ==========================================================================
pause
exit /b 0

:setup_done
echo.
echo ==========================================================================
echo  Libraries installed in python_env. Run_Myelination.bat will use them.
echo  Run build_app.bat without "setup" to build the .exe.
echo ==========================================================================
pause
exit /b 0

rem ---- problems ------------------------------------------------------------------
:no_script
echo Myelination.py is not in this folder - put build_app.bat next to it.
goto :fail

:no_python
echo.
echo No suitable Python (3.10 to 3.14, with tkinter) was found on this PC.
where winget >nul 2>&1
if errorlevel 1 goto :python_manual
choice /c YN /m "Install Python 3.12 now (with winget, for your user only)"
if errorlevel 2 goto :python_manual
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set BASEPY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :have_python
)
echo.
echo Python was installed. Close this window and run build_app.bat again.
goto :fail

:python_manual
echo Install Python 3.12 from https://www.python.org/downloads/ - on the first
echo installer screen tick "Add python.exe to PATH" - then run build_app.bat again.
goto :fail

:no_tkinter
echo The Python on PATH has no tkinter, which the settings window needs.
echo Install Python 3.12 from https://www.python.org/downloads/ (tkinter is
echo included by default), then run build_app.bat again.
goto :fail

:torch_fail
echo.
echo PyTorch (%TORCH_VARIANT%) could not be installed.
echo  - Check the internet connection.
echo  - For a GPU build, try another CUDA version: change TORCH_CUDA at the top
echo    of this file to cu126 or cu130, then run build_app.bat again.
echo  - A very new Python may not have PyTorch builds yet: install Python 3.12,
echo    then run "build_app.bat clean".
goto :fail

:fail
echo.
echo Stopped - see the messages above.
pause
exit /b 1
