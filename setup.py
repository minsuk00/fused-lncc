import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Multi-arch + PTX so the wheel runs on Turing..Hopper and JIT-forward-compiles on newer GPUs.
# Override with TORCH_CUDA_ARCH_LIST to build for just your card (smaller/faster build).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX")

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
