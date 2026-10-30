"""Bounded exact native mathematical query objects.

``Context`` builds formal polynomials and parameter functions. ``MomentBasis``
materializes one immutable scalar, degree-two moment snapshot through ordinary
native Torbles aggregate queries. Later reduction and parameter calculus run in
C and never rescan the source. Construction currently takes one native aggregate
query per basis term; it is not a fused or single-pass executor.

Native limits: eight variables, degree eight, 128 terms, and canonical rationals
whose numerator and positive denominator fit int64. Overflow and missing basis
states are explicit errors, with no wrapping or approximation. Moment inputs are
one to four non-null, integer fact columns. Empty selected input has the zero
linear functional, rather than SQL SUM's NULL result.
"""
import copy
import ctypes as C
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from numbers import Integral
import time
from types import MappingProxyType
import uuid

from . import algq
from .math_domains import (Domain, MomentCache, DomainError, _domain_api,
                           _ModuleTrace, _trace_dict, OPERATIONS, _check as _domain_check)


MAX_VARIABLES, MAX_DEGREE, MAX_TERMS, MAX_DATA_VARIABLES = 8, 8, 128, 4
__all__ = ["Context", "Expression", "Function", "MomentBasis", "MomentCache", "Domain", "AlgebraBudgetError"]


class AlgebraBudgetError(ValueError):
    """An exact result needs a representation beyond the configured native budget."""


class _Rational(C.Structure):
    _fields_ = [("numerator", C.c_int64), ("denominator", C.c_int64)]


class _Term(C.Structure):
    _fields_ = [("exponents", C.c_uint8 * MAX_VARIABLES), ("coefficient", _Rational)]


_native = None


def _api():
    global _native
    if _native is None:
        ptr, err = C.c_void_p, C.POINTER(C.c_int32)
        bindings = {
            "poly_constant": (ptr, [C.c_int64, C.c_int64, err]),
            "poly_variable": (ptr, [C.c_int32, err]),
            **{name: (ptr, [ptr, ptr, err]) for name in ("poly_add", "poly_sub", "poly_mul")},
            "poly_pow": (ptr, [ptr, C.c_uint32, err]),
            "poly_derivative": (ptr, [ptr, C.c_int32, err]),
            "poly_integrate": (ptr, [ptr, C.c_int32, err]),
            "poly_substitute": (ptr, [ptr, C.c_int32, C.c_int64, C.c_int64, err]),
            "poly_evaluate": (C.c_int32, [ptr, C.POINTER(_Rational), C.POINTER(_Rational)]),
            "poly_equal": (C.c_int32, [ptr, ptr]),
            "poly_term_count": (C.c_int32, [ptr]),
            "poly_get_term": (C.c_int32, [ptr, C.c_int32, C.POINTER(_Term)]),
            "poly_degree": (C.c_int32, [ptr]),
            "poly_support": (C.c_int32, [ptr]),
            "poly_free": (None, [ptr]),
            "bank_create": (ptr, [C.c_int32, C.c_int32, C.POINTER(C.c_uint8), C.POINTER(C.c_int64), err]),
            "bank_variables": (C.c_int32, [ptr]),
            "bank_term_count": (C.c_int32, [ptr]),
            "bank_count": (C.c_int32, [ptr, C.POINTER(C.c_int64)]),
            "bank_moment": (C.c_int32, [ptr, C.POINTER(C.c_uint8), C.POINTER(C.c_int64)]),
            "bank_reduce": (ptr, [ptr, ptr, err]),
            "bank_free": (None, [ptr]),
            "table_revision": (C.c_uint64, [ptr]),
        }
        native = {}
        try:
            for name, (restype, args) in bindings.items():
                fn = getattr(algq.lib, "tr_math_" + name)
                fn.restype, fn.argtypes = restype, args
                native[name] = fn
        except AttributeError as exc:
            raise RuntimeError("native mathematics module is unavailable; rebuild the prototype CPU engine") from exc
        _native = native
    return _native


def _check(code):
    if code == -2:
        raise MemoryError("native mathematical allocation failed")
    if code == -3:
        raise OverflowError("exact native mathematical value exceeds int64 rational limits")
    if code == -5:
        raise AlgebraBudgetError("native mathematical degree, term, or moment-basis budget exceeded")
    if code < 0:
        raise ValueError("invalid native mathematical input")
    return code


def _i64(value):
    if not isinstance(value, Integral):
        raise TypeError("an exact integer is required")
    value = int(value)
    if not -(1 << 63) <= value < (1 << 63):
        raise OverflowError("exact integer does not fit int64")
    return value


def _fraction(value):
    if isinstance(value, Integral):
        value = Fraction(int(value))
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("a finite exact constant is required")
        value = Fraction(value)
    elif not isinstance(value, Fraction):
        raise TypeError("use an integer, Fraction, or finite Decimal; floats are not exact input")
    numerator, denominator = _i64(value.numerator), _i64(value.denominator)
    return numerator, denominator


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_thaw(child) for child in value)
    return value


class Context:
    """One named formal variable domain, shared by compatible expressions."""
    def __init__(self):
        self._names = []
        self._ndata = 0
        self._domain_token = uuid.uuid4().bytes
        self._formal = None

    def _formal_domain(self):
        if self._formal is None:
            self._formal = Domain("formal", self._domain_token, carrier="rational",
                                  nvars=MAX_VARIABLES, ndata=self._ndata,
                                  origin=getattr(self, "_row_domain", None))
        return self._formal

    @property
    def domain(self):
        """Return an owned clone of this context's stable formal domain."""
        return self._formal_domain().clone()

    @property
    def names(self):
        return tuple(self._names)

    def _index(self, variable):
        if isinstance(variable, str):
            try:
                return self._names.index(variable)
            except ValueError as exc:
                raise ValueError(f"unknown mathematical variable {variable!r}") from exc
        if isinstance(variable, Expression) and variable.context is self and variable._symbol is not None:
            variable._open()
            return variable._symbol
        raise TypeError("choose a variable name or a symbol from this context")

    def symbol(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("a nonempty variable name is required")
        if name not in self._names:
            if len(self._names) == MAX_VARIABLES:
                raise AlgebraBudgetError("native mathematical contexts have at most eight variables")
            self._names.append(name)
        index = self._names.index(name)
        return self._create("poly_variable", index, symbol=index)

    def symbols(self, *names):
        return tuple(self.symbol(name) for name in names)

    def parameter(self, name):
        if name in self._names[:self._ndata]:
            raise ValueError(f"{name!r} is a fixed data variable, not a free parameter")
        return self.symbol(name)

    def constant(self, value):
        numerator, denominator = _fraction(value)
        return self._create("poly_constant", numerator, denominator)

    def _coerce(self, value):
        if isinstance(value, Expression):
            if value.context is not self:
                raise DomainError("mathematical expressions belong to different variable contexts")
            value._open()
            return value
        return self.constant(value)

    def _create(self, operation, *args, symbol=None, snapshot=None):
        error = C.c_int32()
        native = _api()
        domain = self._formal_domain().clone()
        try:
            handle = native[operation](*args, C.byref(error))
            _check(error.value)
            if not handle:
                raise MemoryError("native mathematical object unavailable")
        except BaseException:
            domain.close()
            raise
        cls = Function if snapshot is not None else Expression
        return cls(self, handle, native, symbol=symbol, snapshot=snapshot, domain=domain)

    def _operate(self, operation, left, right=None, *, variable=0, exponent=0,
                 numerator=0, denominator=1, snapshot=None):
        native = _domain_api()
        error, output, trace = C.c_int32(), C.c_void_p(), _ModuleTrace()
        handle = native["module_apply"](
            1, OPERATIONS[operation], left._domain._open(), left._open(),
            None if right is None else right._domain._open(),
            None if right is None else right._open(), variable, exponent, numerator, denominator,
            C.byref(output), C.byref(trace), C.byref(error))
        _domain_check(error.value)
        if not handle:
            raise MemoryError("native mathematical operator result unavailable")
        domain = Domain._from_handle(output.value)
        cls = Function if snapshot is not None else Expression
        return cls(self, handle, _api(), snapshot=snapshot, domain=domain,
                   module_plan=_trace_dict(trace))


class Expression:
    """An immutable, individually owned native polynomial."""
    __hash__ = None

    def __init__(self, context, handle, native, *, symbol=None, snapshot=None,
                 domain=None, module_plan=None):
        self.context, self._handle, self._native = context, handle, native
        self._symbol, self._snapshot = symbol, snapshot
        self._domain = context._formal_domain().clone() if domain is None else domain
        self._module_plan = module_plan

    def _open(self):
        if not self._handle:
            raise RuntimeError("mathematical expression is closed")
        return self._handle

    def close(self):
        if self._handle:
            self._native["poly_free"](self._handle)
            self._handle = None
        if self._domain is not None:
            self._domain.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self._native["poly_free"](self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def term_count(self):
        return _check(self._native["poly_term_count"](self._open()))

    @property
    def degree(self):
        return _check(self._native["poly_degree"](self._open()))

    @property
    def support(self):
        mask = _check(self._native["poly_support"](self._open()))
        return tuple(name for index, name in enumerate(self.context.names) if mask >> index & 1)

    @property
    def terms(self):
        out = []
        for index in range(self.term_count):
            term = _Term()
            _check(self._native["poly_get_term"](self._open(), index, C.byref(term)))
            out.append((tuple(term.exponents), Fraction(term.coefficient.numerator, term.coefficient.denominator)))
        return tuple(out)

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def metadata(self):
        return None if self._snapshot is None else _thaw(self._snapshot)

    @property
    def domain(self):
        self._open()
        return self._domain.clone()

    @property
    def module_plan(self):
        return None if self._module_plan is None else dict(self._module_plan)

    def _binary(self, other, operation):
        other = self.context._coerce(other)
        snapshot = self._snapshot if self._snapshot is not None else other._snapshot
        if snapshot is not None:
            data_mask = (1 << self.context._ndata) - 1
            for value in (self, other):
                if value._snapshot is None and self._native["poly_support"](value._open()) & data_mask:
                    raise ValueError("cannot mix unaggregated data variables with an aggregate function")
        return self.context._operate(operation, self, other, snapshot=snapshot)

    def __add__(self, other): return self._binary(other, "poly_add")
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self._binary(other, "poly_sub")
    def __rsub__(self, other): return self.context._coerce(other)._binary(self, "poly_sub")
    def __mul__(self, other): return self._binary(other, "poly_mul")
    def __rmul__(self, other): return self.__mul__(other)
    def __neg__(self): return self.__mul__(-1)

    def __pow__(self, exponent):
        if not isinstance(exponent, Integral) or not 0 <= int(exponent) <= 0xFFFFFFFF:
            raise ValueError("a nonnegative uint32 integer exponent is required")
        return self.context._operate("poly_pow", self, exponent=int(exponent), snapshot=self._snapshot)

    def __eq__(self, other):
        try:
            other = self.context._coerce(other)
        except TypeError:
            return NotImplemented
        if not self._domain.context_equal(other._domain):
            raise DomainError("mathematical expressions belong to different typed domains")
        return bool(_check(self._native["poly_equal"](self._open(), other._open())))

    def _calculus_index(self, variable):
        index = self.context._index(variable)
        if self._snapshot is not None and index < self.context._ndata:
            raise ValueError("aggregate functions differentiate and integrate free parameters, not fixed data variables")
        return index

    def diff(self, variable):
        return self.context._operate("poly_derivative", self, variable=self._calculus_index(variable), snapshot=self._snapshot)

    def substitute(self, variable, value):
        numerator, denominator = _fraction(value)
        return self.context._operate("poly_substitute", self, variable=self._calculus_index(variable),
                                     numerator=numerator, denominator=denominator, snapshot=self._snapshot)

    def integral(self, variable, lower=None, upper=None):
        """Antiderivative with zero constant, or a definite integral preserving other parameters."""
        index = self._calculus_index(variable)
        if (lower is None) != (upper is None):
            raise ValueError("provide both integration bounds or neither")
        result = self.context._operate("poly_integrate", self, variable=index, snapshot=self._snapshot)
        if lower is None:
            return result
        try:
            high, low = result.substitute(variable, upper), result.substitute(variable, lower)
            try:
                return high - low
            finally:
                high.close()
                low.close()
        finally:
            result.close()

    def evaluate(self, bindings=None, **values):
        """Evaluate exactly; every used free variable must have an explicit binding."""
        if bindings is not None:
            bindings = dict(bindings)
            if bindings.keys() & values.keys():
                raise ValueError("duplicate mathematical variable bindings")
            values = {**bindings, **values}
        unknown = values.keys() - set(self.context.names)
        if unknown:
            raise ValueError(f"unknown mathematical variable bindings: {sorted(unknown)}")
        missing = set(self.support) - values.keys()
        if missing:
            raise ValueError(f"missing mathematical variable bindings: {sorted(missing)}")
        if self._snapshot is not None and values.keys() & set(self.context.names[:self.context._ndata]):
            raise ValueError("aggregate functions bind free parameters, not fixed data variables")
        inputs = (_Rational * MAX_VARIABLES)(*(_Rational(0, 1) for _ in range(MAX_VARIABLES)))
        for name, value in values.items():
            inputs[self.context._index(name)] = _Rational(*_fraction(value))
        result = _Rational()
        trace = _ModuleTrace()
        _domain_check(_domain_api()["module_evaluate"](
            1, self._domain._open(), self._open(), inputs, C.byref(result), C.byref(trace)))
        self._module_plan = _trace_dict(trace)
        value = Fraction(result.numerator, result.denominator)
        return value.numerator if value.denominator == 1 else value

    __call__ = evaluate

    def __repr__(self):
        if not self._handle:
            return f"{type(self).__name__}(closed)"
        terms = []
        for exponents, coefficient in self.terms:
            powers = [name + (f"**{exponent}" if exponent != 1 else "")
                      for name, exponent in zip(self.context.names, exponents) if exponent]
            terms.append(str(coefficient) + ("*" + "*".join(powers) if powers else ""))
        return f"{type(self).__name__}(" + (" + ".join(terms) or "0") + ")"


class Function(Expression):
    """A parameter polynomial reduced against an immutable moment snapshot."""


def _source_signature(tables):
    """Detect native revisions and supported scope changes during materialization.

    This is not MVCC: source mutation must be serialized with materialization.
    Native revisions catch even a flip subsequently repaired between passes.
    """
    native = _api()
    signatures = []
    for table in tables:
        identity = table.__dict__.get("_math_domain_identity")
        if identity is None:
            # setdefault keeps first-seen assignment stable even when separate
            # readers race. A new Table wrapper receives a fresh lifetime token,
            # so allocator address reuse cannot impersonate an earlier source.
            identity = table.__dict__.setdefault("_math_domain_identity", uuid.uuid4().hex)
        codecs = {name: codec.to_json() for name, codec in table.codecs.items()}
        signatures.append((table.name, C.cast(table.at, C.c_void_p).value, table.n,
                           int(native["table_revision"](table.at)), tuple(table.cols),
                           tuple(table.width(name) for name in table.cols),
                           hashlib.sha256(table.used_bm.tobytes()).hexdigest(),
                           hashlib.sha256(json.dumps(codecs, sort_keys=True, default=str).encode()).hexdigest(),
                           tuple(sorted((name, surrogate.version) for name, surrogate in getattr(table, "surr", {}).items())),
                           tuple(sorted((name, C.cast(sidecar["side"], C.c_void_p).value,
                                         int(native["table_revision"](sidecar["side"])),
                                         hashlib.sha256(sidecar["rank"].tobytes()).hexdigest())
                                        for name, sidecar in getattr(table, "sidecars", {}).items())), identity))
    return tuple(signatures)


class MomentBasis(Context):
    """Immutable complete degree-two scalar moment snapshot of a fixed query scope.

    ``variables`` are formal data coordinates. ``parameter(name)`` creates free
    coordinates, and ``reduce(expression)`` returns their native function.
    Materialization is serialized with source mutation. Later source mutations
    do not alter already materialized snapshots or functions.
    """
    def __init__(self, query, columns, *, engine="alg", cache=None):
        super().__init__()
        self._handle = None
        self._native = _api()
        self._row_domain = None
        if cache is not None and not isinstance(cache, MomentCache):
            raise TypeError("cache must be a MomentCache")
        if not isinstance(query, algq.Query):
            raise TypeError("MomentBasis requires an engine-level Torbles Query")
        if engine not in ("alg", "gpu", "plain"):
            raise ValueError("engine must be alg, gpu, or plain")
        if query.keys or getattr(query, "rjoin", None) is not None:
            raise ValueError("initial moment snapshots require scalar scope without range joins")
        if any(getattr(query, name, None) is not None for name in ("_xrows", "_word_ranges", "_dom", "_ext")) \
                or getattr(query, "_rows", False) or getattr(query, "_codes", False):
            raise ValueError("external row scopes and internal lowered queries are not supported by moment snapshots")
        fact = query.fact
        names = []
        for column in columns:
            if isinstance(column, algq.Col):
                if column.table is not fact:
                    raise ValueError("moment coordinates must be columns of the fact table")
                name = column.name
            elif isinstance(column, str):
                name = column
            else:
                raise TypeError("moment columns must be names or fact columns")
            if name not in fact.cols:
                raise ValueError(f"unknown moment column {name!r}")
            codec = fact.codecs.get(name)
            if codec is not None and (codec.kind != "int" or codec.affine is None or codec.affine[1]):
                raise ValueError("initial moment snapshots support exact integer columns only")
            if name in getattr(fact, "nullable", {}) or name in getattr(fact, "wide", {}) or fact.width(name) > 32 or "#" in name:
                raise ValueError("moment coordinates must be non-null integer columns with at most 32-bit stored codes")
            if name in names:
                raise ValueError("moment coordinates must be distinct")
            names.append(name)
        if not 1 <= len(names) <= MAX_DATA_VARIABLES:
            raise ValueError("choose one to four integer moment coordinates")
        for name, surrogate in getattr(fact, "surr", {}).items():
            if name in query.joins.values() and getattr(surrogate, "stale", None):
                raise ValueError("moment snapshots of moved-key delta scopes are not supported")
        self._names, self._ndata = names, len(names)
        tables = tuple(sorted({fact, *query.joins}, key=lambda table: (table.name, id(table))))
        if query.pred is not None and algq.tables_of(query.pred) - set(tables):
            raise ValueError("moment predicates may reference only the fact and explicitly joined dimensions")
        source_before = _source_signature(tables)
        scoped = algq.copy_query(query)
        scoped.joins, scoped.keys = dict(query.joins), []
        memo = {id(table): table for table in tables}
        scoped.pred = copy.deepcopy(query.pred, memo)
        scoped._both = False
        scope_sql = scoped.count().sql()
        powers = [tuple(0 for _ in names)]
        powers.extend(tuple(int(index == variable) for index in range(self._ndata)) for variable in range(self._ndata))
        powers.extend(tuple(int(index == left) + int(index == right) for index in range(self._ndata))
                      for left in range(self._ndata) for right in range(left, self._ndata))
        scope_key = json.dumps(dict(version=1, scope_sql=scope_sql, columns=names,
                                    sources=source_before, numeric="logical integer-affine",
                                    validity="nonnull", grouping="scalar", empty="zero linear functional"),
                               sort_keys=True, separators=(",", ":")).encode()
        basis_key = json.dumps(dict(version=1, degree=2, columns=names, powers=powers,
                                    codecs={name: fact.codecs[name].to_json() if name in fact.codecs else None
                                            for name in names}), sort_keys=True, separators=(",", ":")).encode()
        self._row_domain = Domain("snapshot", scope_key, carrier="integer", nvars=self._ndata,
                                  ndata=self._ndata, revision=int(self._native["table_revision"](fact.at)),
                                  universe_rows=int(fact.n))
        values, devices = [], []
        started = time.perf_counter()
        if cache is not None:
            self._handle = cache._get(self._row_domain, basis_key, self._ndata)
        cache_hit = bool(self._handle)
        if not cache_hit:
            for exponents in powers:
                current = algq.copy_query(scoped)
                if not any(exponents):
                    value = current.count().run(engine)
                else:
                    measure = None
                    for name, exponent in zip(names, exponents):
                        for _ in range(exponent):
                            coordinate = algq.Col(fact, name)
                            measure = coordinate if measure is None else measure * coordinate
                    value = current.sum(measure).run(engine)
                    if value is None:
                        value = 0
                values.append(_i64(value))
                devices.append(getattr(current, "device", "unknown"))
        if source_before != _source_signature(tables):
            raise RuntimeError("source snapshot changed during moment materialization; retry after mutation completes")
        cache_store_status = "hit" if cache_hit else "disabled"
        if not cache_hit:
            exponents = (C.c_uint8 * (len(powers) * self._ndata))(*(power for term in powers for power in term))
            moments = (C.c_int64 * len(values))(*values)
            error = C.c_int32()
            self._handle = self._native["bank_create"](self._ndata, len(powers), exponents, moments, C.byref(error))
            _check(error.value)
            if not self._handle:
                raise MemoryError("native moment bank unavailable")
            if cache is not None:
                cache_store_status = cache._put(self._row_domain, basis_key, self._handle)
        self._powers = tuple(powers)
        gpu_executor = algq.gpu_name() if engine == "gpu" and not cache_hit else None
        self._snapshot = _freeze(dict(scope_sql=scope_sql, columns=tuple(names), row_count=self.row_count,
                                     source_signatures=source_before, engine_requested=engine,
                                     native_query_calls=0 if cache_hit else len(powers), actual_devices=tuple(devices),
                                     cache_hit=cache_hit, cache_enabled=cache is not None,
                                     cache_store_status=cache_store_status, basis_terms=len(powers),
                                     row_domain=self._row_domain.spec,
                                     domain_key_sha256=hashlib.sha256(scope_key).hexdigest(),
                                     gpu_executor=gpu_executor,
                                     gpu_emulated=("emulat" in gpu_executor.lower()) if gpu_executor is not None else None,
                                     coefficient_executor="native CPU",
                                     materialization_ms=(time.perf_counter()-started)*1000,
                                     empty_semantics="zero linear functional, not SQL SUM NULL",
                                     construction="owned native cache clone; no aggregate queries" if cache_hit else
                                                  "one existing native aggregate query per basis term; not fused",
                                     lifetime="immutable snapshot; later source mutations do not refresh it"))

    def _open(self):
        if not self._handle:
            raise RuntimeError("moment snapshot is closed")
        return self._handle

    @property
    def variables(self):
        return {name: self.symbol(name) for name in self._names[:self._ndata]}

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def metadata(self):
        return _thaw(self._snapshot)

    @property
    def row_domain(self):
        self._open()
        return self._row_domain.clone()

    @property
    def row_count(self):
        result = C.c_int64()
        _check(self._native["bank_count"](self._open(), C.byref(result)))
        return result.value

    def moment(self, powers):
        powers = tuple(powers)
        if len(powers) != self._ndata or any(not isinstance(value, Integral) or not 0 <= value <= 2 for value in powers):
            raise ValueError("provide one degree-two nonnegative exponent per data coordinate")
        exponents = (C.c_uint8 * self._ndata)(*powers)
        result = C.c_int64()
        _check(self._native["bank_moment"](self._open(), exponents, C.byref(result)))
        return result.value

    def reduce(self, expression):
        expression = self._coerce(expression)
        if expression.snapshot is not None:
            raise ValueError("an aggregate function is already reduced")
        error, output, trace = C.c_int32(), C.c_void_p(), _ModuleTrace()
        handle = _domain_api()["module_reduce"](
            2, self._row_domain._open(), self._open(), expression._domain._open(), expression._open(),
            C.byref(output), C.byref(trace), C.byref(error))
        _domain_check(error.value)
        if not handle:
            raise MemoryError("native aggregate function unavailable")
        return Function(self, handle, self._native, snapshot=self._snapshot,
                        domain=Domain._from_handle(output.value), module_plan=_trace_dict(trace))

    def close(self):
        if self._handle:
            self._native["bank_free"](self._handle)
            self._handle = None
        if self._row_domain is not None:
            self._row_domain.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self._native["bank_free"](self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()
