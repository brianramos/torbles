"""The mathematical modules' CUDA library: libmathgpu.so (nvcc, an actual device) or libmathgpu_emu.so (the same
kernels compiled as C++ and run on the CPU, for tests). Which one ran is always reported: an emulated run is never
presented as a CUDA run.

    TORBLES_MATH_GPU_LIB=<path>   use this library
    TORBLES_MATH_GPU=emulate      use the emulation even when libmathgpu.so exists
"""
import ctypes as C
import os
from pathlib import Path

_lib = None
HERE = Path(__file__).resolve().parent


def lib():
    """the loaded library (real if built and a device answers, else the emulation)"""
    global _lib
    if _lib is None:
        path = os.environ.get("TORBLES_MATH_GPU_LIB")
        if not path:
            real, emu = HERE / "libmathgpu.so", HERE / "libmathgpu_emu.so"
            path = str(real if real.exists() and os.environ.get("TORBLES_MATH_GPU") != "emulate" else emu)
        if not Path(path).exists():
            raise RuntimeError(f"{path} is not built: run `make` (emulation) or `make gpu` (CUDA)")
        handle = C.CDLL(path)
        handle.tr_mgpu_device_name.restype = C.c_char_p
        handle.tr_mgpu_emulated.restype = C.c_int
        handle.tr_mgpu_core_selftest.restype = C.c_int
        handle.tr_mgpu_core_selftest.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.POINTER(C.c_int64)]
        _lib = handle
    return _lib


def device():
    """{"name": the device or "CPU emulation of the CUDA kernels", "emulated": bool, "library": path}"""
    handle = lib()
    return dict(name=handle.tr_mgpu_device_name().decode(), emulated=bool(handle.tr_mgpu_emulated()),
                library=handle._name)


def check(code, refusal=None, what=""):
    """map a native error code of a GPU entry point to the exception the CPU modules raise"""
    if code == 0:
        return
    from . import math_errors
    if code == -11:
        raise math_errors.DeviceError(f"CUDA device error on {device()['name']}")
    math_errors.check(code, refusal, what)


def require_device():
    """refuse the emulation where an actual device is required (TORBLES_MATH_REQUIRE_DEVICE=1, benchmarks --device)"""
    info = device()
    if info["emulated"]:
        from .math_errors import DeviceError
        raise DeviceError("an actual CUDA device is required, but only the emulation is loaded (run `make gpu`)")
    return info


def bind(name, restype, argtypes):
    """a GPU entry point with its ctypes signature"""
    fn = getattr(lib(), name)
    fn.restype, fn.argtypes = restype, argtypes
    return fn


def run(name, *args, refusal=None, what=""):
    """call a bound GPU entry point and map its code; with TORBLES_MATH_REQUIRE_DEVICE=1 the emulation is refused"""
    if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
        require_device()
    check(getattr(lib(), name)(*args), refusal, what)


def i64_array(values):
    import numpy as np
    a = np.ascontiguousarray(values, dtype=np.int64)
    return a, a.ctypes.data_as(C.POINTER(C.c_int64))
