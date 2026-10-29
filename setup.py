# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Build hook for pip: compiles the engine into the package while it is built.

    pip install .                        CPU engine for this machine (-march=native) + the GPU kernels' CPU emulation;
                                         the CUDA executor too if nvcc is found (TORBLES_CUDA=0 skips it)
    TORBLES_PORTABLE=1 python -m build   a wheel for other machines (x86-64-v2; the popcount kernels still pick
                                         AVX2 / AVX-512 at run time)

Everything else is in pyproject.toml. The engine is loaded with ctypes, so it is built as plain shared libraries
(libalg.so, libgpu_emu.so, libgpu.so), not as Python extension modules."""
import os, platform, shutil, subprocess, sys
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


def _run(cmd, cwd):
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode == 0


class BuildEngine(build_py):
    def run(self):
        super().run()
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torbles")
        out = os.path.abspath(os.path.join(self.build_lib, "torbles"))
        os.makedirs(out, exist_ok=True)
        if platform.machine().lower() not in ("x86_64", "amd64"):
            sys.exit(f"torbles builds on x86-64 (this machine is {platform.machine()}): the engine uses x86 popcount kernels")
        cc = os.environ.get("CC", "gcc"); cxx = os.environ.get("CXX", "g++")
        arch = ["-march=x86-64-v2", "-mtune=generic"] if os.environ.get("TORBLES_PORTABLE") else ["-march=native"]
        if not _run([cc, "-O2", *arch, "-shared", "-fPIC", os.path.join(src, "alg.c"), "-o", os.path.join(out, "libalg.so"), "-lpthread"], src):
            sys.exit("could not compile torbles/alg.c: a C compiler (gcc or clang) is needed")
        if not _run([cxx, "-O2", "-fwrapv", "-DEMULATE", "-x", "c++", "-shared", "-fPIC", os.path.join(src, "gpu.cu"), "-o", os.path.join(out, "libgpu_emu.so")], src):
            sys.exit("could not compile the GPU emulation (torbles/gpu.cu as C++): a C++ compiler is needed")
        nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
        if nvcc and os.environ.get("TORBLES_CUDA", "1") != "0" and not os.environ.get("TORBLES_PORTABLE"):
            gpu = os.path.join(out, "libgpu.so")
            for flags in (["-arch=native"], ["-gencode", "arch=compute_90,code=compute_90"],
                          ["-gencode", "arch=compute_80,code=compute_80"], ["-DNO_B1MMA"]):
                if _run([nvcc, "-O3", *flags, "-shared", "-Xcompiler", "-fPIC", os.path.join(src, "gpu.cu"), "-o", gpu], src):
                    print(f"torbles: CUDA executor built ({' '.join(flags)})"); break
            else:
                print("torbles: nvcc found but the CUDA executor did not build; the CPU engine is installed")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):          # the wheel carries compiled libraries: platform-specific
        return True


cmds = {"build_py": BuildEngine}
try:                                    # the libraries are loaded with ctypes: any Python 3, one platform
    try: from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError: from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class BdistWheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options(); self.root_is_pure = False
        def get_tag(self):
            _, _, plat = super().get_tag(); return "py3", "none", plat
    cmds["bdist_wheel"] = BdistWheel
except ImportError:
    pass

setup(cmdclass=cmds, distclass=BinaryDistribution)
