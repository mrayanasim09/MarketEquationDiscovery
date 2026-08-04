"""pytest configuration file for the v2.1 benchmark test suite."""
import sys
from pathlib import Path

# Ensure the project root is on the path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
