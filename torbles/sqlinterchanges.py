# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Certified switch-conditioned projections of encoded foreign keys."""

from dataclasses import dataclass

import numpy as np

from .sqlsnapshot import _revision


@dataclass(frozen=True)
class Interchange:
    sectors: np.ndarray
    shift: int
    bits: int
    covered: int
    total: int


@dataclass(frozen=True)
class Coordinate:
    shift: int
    width: int
    labels: np.ndarray


def certify_coordinate(raw, values, bits):
    """Certify equality in both directions before grouping by a surrogate field."""
    if not bits or len(bits) > 16 or not len(raw) or len(raw) != len(values):
        return None
    shift, width = min(bits), len(bits)
    if not 0 <= shift <= 32 - width or sorted(bits) != list(range(shift, shift + width)):
        return None
    codes = (np.asarray(raw, np.uint32) >> shift) & ((1 << width) - 1)
    order = np.argsort(codes, kind="stable")
    keys, targets = codes[order], np.asarray(values, np.uint32)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    low, high = np.minimum.reduceat(targets, starts), np.maximum.reduceat(targets, starts)
    if not np.array_equal(low, high) or len(np.unique(low)) != len(low):
        return None
    labels = np.zeros(1 << width, dtype=np.uint32)
    labels[keys[starts]] = low
    labels.flags.writeable = False
    return Coordinate(shift, width, labels)


def _fit(raw, values, width):
    """Fit affine GF(2) coefficients, then certify every row in the sector."""
    bias = int(values[0])
    if np.all(values == bias):
        return [1, bias, *([0] * width)]
    equations = {}
    for i in np.linspace(0, len(raw) - 1, min(96, len(raw)), dtype=np.intp):
        key, target = int(raw[i]) | (1 << 32), int(values[i])
        while key:
            pivot = key.bit_length() - 1
            prior = equations.get(pivot)
            if prior is None:
                equations[pivot] = key, target
                break
            key ^= prior[0]
            target ^= prior[1]
        if not key and target:
            return None
    coefficients = {}
    for pivot, (key, target) in sorted(equations.items()):
        for bit, value in coefficients.items():
            if key & (1 << bit):
                target ^= value
        if target:
            coefficients[pivot] = target
    bias = coefficients.pop(32, 0)
    predicted = np.full(len(raw), bias, dtype=np.uint32)
    for bit, value in coefficients.items():
        predicted ^= ((raw >> bit) & 1) * np.uint32(value)
    if not np.array_equal(predicted, values):
        return None
    masks = [sum(1 << bit for bit, value in coefficients.items() if value >> j & 1) for j in range(width)]
    return [1, bias, *masks]


def compile_interchange(raw, values, code_width, width, *, min_coverage=0.25):
    """Choose a bounded switch table over the high bits of the physical FK code."""
    if not 1 <= code_width <= 32 or not 1 <= width <= 32 or not len(raw) or len(raw) != len(values):
        return None
    raw, values = np.asarray(raw, np.uint32), np.asarray(values, np.uint32)
    if int(raw.max()) >= 1 << code_width or int(values.max()) >= 1 << width:
        return None
    best = None
    # Avoid learning tiny sectors which merely memorize a sparse dimension.
    sizes = sorted({min(code_width, bits) for bits in (2, 4, 6, 8, 10)})
    for bits in sizes:
        shift = code_width - bits
        sector = raw >> shift
        order = np.argsort(sector, kind="stable")
        boundaries = np.flatnonzero(np.diff(sector[order])) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(raw)]
        table = np.zeros((1 << bits, width + 2), dtype=np.uint32)
        covered = 0
        for start, end in zip(starts, ends):
            if end - start < 64:
                continue
            selected = order[start:end]
            row = _fit(raw[selected], values[selected], width)
            if row is not None:
                table[sector[selected[0]]] = row
                covered += end - start
        if covered and covered >= len(raw) * min_coverage and (best is None or covered > best.covered):
            table.flags.writeable = False
            best = Interchange(table, shift, bits, covered, len(raw))
        if covered == len(raw):
            break
    return best


def prepare(fact, determinant, name, projection, *, affine=False):
    """Cache a proof against the exact dimension and physical FK revisions."""
    lookup, _, width = projection
    dim = fact._refdim[determinant]
    surrogate = fact.surr.get(determinant)
    if surrogate is not None and surrogate.version:
        return None
    hint = getattr(surrogate, "coords", {}).get(name)
    if not affine and (not hint or len(hint) >= fact.width(determinant)):
        return None
    part, dim_part = fact._part_of(determinant), dim._part_of(name)
    signature = (_revision(part.at), _revision(dim_part.at), id(lookup), id(surrogate), width, affine)
    cache = fact.__dict__.setdefault("_sector_interchanges", {})
    key = determinant, name
    if key not in cache or cache[key][0] != signature:
        rows = dim.used_ids
        raw = rows if surrogate is None else surrogate.s_of_row[rows]
        values = lookup[rows]
        plan = certify_coordinate(raw, values, hint)
        if plan is None and affine:
            plan = compile_interchange(raw, values, fact.width(determinant), width)
        cache[key] = signature, plan
    return cache[key][1]
