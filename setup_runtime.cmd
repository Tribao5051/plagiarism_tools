@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  py -3.12 -m venv venv
)

call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements_runtime.txt

echo.
echo Runtime environment is ready.
echo Configure [dspace] in plagiarism_config.toml, then run run_dspace_once.cmd.
