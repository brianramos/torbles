# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Exceptions of the native mathematical modules, by native error code.

The codes are those of torbles/mathematics.h (-1 .. -5), math_domain.h (-6 .. -8) and math_num.h (-9 ..). Every
mathematical module maps a native code through check(); a refusal never turns into a Python computation.
"""
from .mathematics import AlgebraBudgetError
from .math_domains import DomainError, CarrierError, OperatorError

RESOURCES = ("variables", "degree", "support", "tensor", "circuit", "iterations", "output", "states", "bytes")
UNKNOWN = (1 << 64) - 1   # a size past 64 bits


class SingularError(ArithmeticError):
    """A system has no unique solution (singular or rank deficient)."""


class NoSolutionError(ValueError):
    """An inverse or preimage query has no answer in its declared domain."""


class DeviceError(RuntimeError):
    """The CUDA device failed or is absent; nothing falls back to the CPU silently."""


class UnsupportedError(ValueError):
    """The request is outside the capability's declared scope."""


class StaleStateError(RuntimeError):
    """A state's source changed (revision, rows or scope) since the state was built."""


class NotConvergedError(ArithmeticError):
    """An iterative method did not certify its result within its iteration budget."""


class InfeasibleError(ValueError):
    """The constraints of an optimization or search admit no point."""


class UnboundedError(ValueError):
    """An objective is unbounded below on its declared domain."""


class ClosedHandleError(RuntimeError):
    """The handle (or its table) was closed; raised the same way whether or not the payload was freed yet."""


_BY_CODE = {
    -1: (ValueError, "invalid native mathematical input"),
    -2: (MemoryError, "native mathematical allocation failed"),
    -3: (OverflowError, "an exact native value exceeds its fixed-width carrier"),
    -5: (AlgebraBudgetError, "native mathematical budget exceeded"),
    -6: (DomainError, "incompatible native mathematical domains"),
    -7: (CarrierError, "native operator does not support this carrier"),
    -8: (OperatorError, "native module does not implement this operator"),
    -9: (SingularError, "the system has no unique solution"),
    -10: (NoSolutionError, "no answer exists in the declared domain"),
    -11: (DeviceError, "CUDA device error"),
    -12: (UnsupportedError, "outside the capability's declared scope"),
    -13: (StaleStateError, "the state's source changed since it was built"),
    -14: (NotConvergedError, "not certified within the iteration budget"),
    -15: (InfeasibleError, "the constraints admit no point"),
    -16: (UnboundedError, "the objective is unbounded below"),
    -17: (ClosedHandleError, "the handle is closed"),
}


def budget_error(resource, need, limit, what=""):
    """AlgebraBudgetError naming the refused resource, with .resource, .need and .limit"""
    name = RESOURCES[resource] if isinstance(resource, int) and 0 <= resource < len(RESOURCES) else str(resource)
    shown = "more than 2^64-1" if need == UNKNOWN else str(need)
    err = AlgebraBudgetError(f"{what + ': ' if what else ''}needs {shown} {name}, budget {limit}")
    err.resource, err.need, err.limit = name, int(need), int(limit)
    return err


def check(code, refusal=None, what=""):
    """raise the exception of a negative native code (with the refusal's resource for a budget refusal); else return code"""
    if code >= 0:
        return code
    if code == -5 and refusal is not None and refusal.resource >= 0:
        raise budget_error(refusal.resource, refusal.need, refusal.limit, what)
    cls, message = _BY_CODE.get(code, (ValueError, f"native mathematical error {code}"))
    raise cls(f"{what}: {message}" if what else message)
