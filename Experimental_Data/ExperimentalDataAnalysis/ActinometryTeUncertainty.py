# -*- coding: utf-8 -*-
"""
ActinometryTeUncertainty.py  -  moved to pipeline/ActinometryUncertainty.py

Kept under its old name so older scripts keep working:
  import ActinometryTeUncertainty     gives the pipeline module ActinometryUncertainty itself (the same module object, so a setting changed
                       through either name is seen by both);
  running this file    runs pipeline/ActinometryUncertainty.py.
New code should import ActinometryUncertainty from pipeline/ directly.
"""
import os
import sys

_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(_PIPELINE, "ActinometryUncertainty.py"), run_name="__main__")
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("ActinometryUncertainty")
