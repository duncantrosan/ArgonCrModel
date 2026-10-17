# -*- coding: utf-8 -*-
"""
N2SweepPureArEEDF.py  -  moved to pipeline/RunDatasets.py

Kept under its old name so older scripts keep working:
  import N2SweepPureArEEDF     gives the pipeline module RunDatasets itself (the same module object, so a setting changed
                       through either name is seen by both);
  running this file    runs pipeline/RunDatasets.py.
New code should import RunDatasets from pipeline/ directly.
"""
import os
import sys

_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(_PIPELINE, "RunDatasets.py"), run_name="__main__")
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("RunDatasets")
