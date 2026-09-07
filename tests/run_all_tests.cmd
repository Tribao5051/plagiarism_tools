@echo off
setlocal
cd /d "%~dp0.."

if not exist "venv\Scripts\python.exe" (
  echo Missing venv. Run runtime\setup_runtime.cmd first.
  exit /b 1
)

call "venv\Scripts\activate.bat"
python -m unittest discover -s tests -t . -v
