"""Compatibility launcher for the accidentally nested project copy."""

from pathlib import Path
import os
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)
runpy.run_path(str(ROOT / "Main.py"), run_name="__main__")
