import os
import sys


# Allow tests to import sibling modules (config.py, events.py, etc.)
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

