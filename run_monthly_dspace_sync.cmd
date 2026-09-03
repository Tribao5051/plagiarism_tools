@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Missing venv\Scripts\python.exe
  exit /b 1
)

if not exist "plagiarism_config.toml" (
  echo Missing plagiarism_config.toml
  exit /b 1
)

call "venv\Scripts\activate.bat"
python corpus_scheduler.py --dspace-config plagiarism_config.toml --interval-days 30
