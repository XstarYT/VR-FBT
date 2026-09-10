"""Compatibility launcher for the accidentally nested project copy."""

from pathlib import Path
import os
import runpy


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
runpy.run_path(str(ROOT / "Main.py"), run_name="__main__")
