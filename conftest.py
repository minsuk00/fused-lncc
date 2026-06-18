import os, sys
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "baselines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
