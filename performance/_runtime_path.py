import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"

runtime_path = str(RUNTIME_ROOT)
if runtime_path not in sys.path:
    sys.path.insert(0, runtime_path)
