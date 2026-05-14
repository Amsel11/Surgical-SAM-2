# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Build hook for the SAM2 CUDA connected-components extension (sam2._C).

All project metadata + dependencies live in pyproject.toml. This file exists
only to compile sam2/csrc/connected_components.cu.

Disable the build with `SAM2_BUILD_CUDA=0` (default on bp inference nodes —
the Python fallback is sufficient for our pipeline). Errors during the build
are non-fatal unless `SAM2_BUILD_ALLOW_ERRORS=0`.
"""
import os

from setuptools import setup

BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension. SAM 2 still runs without it, but some "
    "post-processing falls back to the Python implementation. See INSTALL.md.\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []
    try:
        from torch.utils.cpp_extension import CUDAExtension

        return [CUDAExtension(
            "sam2._C",
            ["sam2/csrc/connected_components.cu"],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            return []
        raise


try:
    from torch.utils.cpp_extension import BuildExtension

    class BuildExtensionIgnoreErrors(BuildExtension):
        def finalize_options(self):
            try:
                super().finalize_options()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

    cmdclass = {
        "build_ext": (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        ),
    }
except Exception:
    cmdclass = {}


setup(ext_modules=get_extensions(), cmdclass=cmdclass)
