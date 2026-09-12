import sys
import os

# Ensure the project root is on sys.path so pytest can
# resolve `tritonforge.*` imports without a pip install step.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
