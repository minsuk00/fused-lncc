import os, sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# baselines/ has no installed distribution, so it must be importable from the tree.
_bl = os.path.join(_ROOT, "baselines")
if _bl not in sys.path:
    sys.path.insert(0, _bl)

# Append (do NOT insert at the front) the repo root, so an INSTALLED `fused_lncc` /
# `fused_lncc_cuda` takes precedence over the in-tree source and any stale in-tree `.so`
# (e.g. an editable build linked to a different torch). The source tree is only a fallback
# when the package is not installed.
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
