import os, re, subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME


def _nvcc_major():
    """Major version of the nvcc that will compile the extension (None if undetectable)."""
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc") if CUDA_HOME else "nvcc"
    try:
        m = re.search(r"release (\d+)\.", subprocess.check_output([nvcc, "--version"], text=True))
        return int(m.group(1)) if m else None
    except Exception:
        return None


# Multi-arch + PTX so the wheel runs on Volta..Blackwell and JIT-forward-compiles on newer GPUs.
# CUDA 13 dropped sm_70 (Volta), so only include 7.0 when building with CUDA < 13 (otherwise nvcc
# fails with "Unsupported gpu architecture 'compute_70'"). Volta GPUs require a CUDA <= 12.x toolkit.
# Override TORCH_CUDA_ARCH_LIST to build for just your card (smaller/faster build).
_archs = "7.5;8.0;8.6;8.9;9.0;12.0+PTX"
if (_nvcc_major() or 12) < 13:
    _archs = "7.0;" + _archs
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _archs)

setup(
    name="fused_lncc",
    version="0.1.0",
    description="Fully-fused 3D Local Normalized Cross-Correlation loss (CUDA)",
    author="MRI2CT",
    packages=["fused_lncc"],
    package_data={"fused_lncc": ["csrc/*.cu"]},   # ship the source for the JIT fallback + sdist
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="fused_lncc_cuda",
            sources=["fused_lncc/csrc/lncc3d.cu"],
            # NOTE: deliberately NO --use_fast_math: the variance (sum-of-squares minus
            # square-of-sums) is cancellation-sensitive and fast-math hurts accuracy here.
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.3.0"],
)
