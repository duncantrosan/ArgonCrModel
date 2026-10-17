# -*- coding: utf-8 -*-
"""
Argon_Nitrogen_Mix_Analysis_V2.py  -  moved to pipeline/SpectrumLineFitter.py

Kept under its old name so older scripts keep working:
  import Argon_Nitrogen_Mix_Analysis_V2     gives the pipeline module SpectrumLineFitter itself (the same module object, so a setting changed
                       through either name is seen by both);
  running this file    runs pipeline/SpectrumLineFitter.py.
New code should import SpectrumLineFitter from pipeline/ directly.
"""
import os
import sys

_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(_PIPELINE, "SpectrumLineFitter.py"), run_name="__main__")
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("SpectrumLineFitter")
