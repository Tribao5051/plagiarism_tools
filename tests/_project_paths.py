import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for folder_name in ("runtime", "performance"):
    folder_path = str(PROJECT_ROOT / folder_name)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)
