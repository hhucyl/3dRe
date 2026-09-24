@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "CONDA_PYTHON=D:\software\anaconda\install\python.exe"

if exist "%CONDA_PYTHON%" (
    "%CONDA_PYTHON%" "%PROJECT_DIR%main.py" %*
) else (
    python "%PROJECT_DIR%main.py" %*
)

if errorlevel 1 pause
endlocal
