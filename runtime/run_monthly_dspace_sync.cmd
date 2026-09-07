@echo off
setlocal
cd /d "%~dp0.."

if not exist "venv\Scripts\python.exe" (
  echo Missing venv\Scripts\python.exe
  exit /b 1
)

if not exist "runtime\plagiarism_config.toml" (
  echo Missing runtime\plagiarism_config.toml
  exit /b 1
)

call "venv\Scripts\activate.bat"
python runtime\corpus_scheduler.py --dspace-config runtime\plagiarism_config.toml --interval-days 30
