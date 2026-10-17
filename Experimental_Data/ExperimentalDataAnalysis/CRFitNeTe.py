# -*- coding: utf-8 -*-
"""
CRFitNeTe.py  -  moved to pipeline/CRTableFit.py

Kept under its old name so older scripts keep working:
  import CRFitNeTe     gives the pipeline module CRTableFit itself (the same module object, so a setting changed
                       through either name is seen by both);
  running this file    runs pipeline/CRTableFit.py.
New code should import CRTableFit from pipeline/ directly.
"""
import os
import sys

_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(_PIPELINE, "CRTableFit.py"), run_name="__main__")
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("CRTableFit")
