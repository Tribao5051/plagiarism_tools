@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Missing venv. Run setup_runtime.cmd first.
  exit /b 1
)

if not exist "plagiarism_config.toml" (
  echo Missing plagiarism_config.toml.
  exit /b 1
)

call "venv\Scripts\activate.bat"
python dspace_vector_pipeline.py --config plagiarism_config.toml
