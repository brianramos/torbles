"""Check actual native dispatch, including checks bypassing Python operators."""
import ctypes as C
from fractions import Fraction
import unittest
from unittest.mock import patch

import pandas as pd
import torbles
from torbles.mathematics import Context, MomentBasis, _api, _Rational
from torbles.math_domains import Domain, modules, _domain_api, _ModuleInfo, _ModuleTrace


class NativeMathematicalModules(unittest.TestCase):
    def domain(self,*args,**kwargs):
        value=Domain(*args,**kwargs)
        self.addCleanup(value.close)
        return value

    def raw_apply(self,left,left_domain,right=None,right_domain=None,operation=3,module=1,variable=0):
        out=C.c_void_p(123)
        trace=_ModuleTrace(); error=C.c_int32()
        result=_domain_api()["module_apply"](module,operation,left_domain._open(),left._open(),
                None if right_domain is None else right_domain._open(),None if right is None else right._open(),
                variable,2,0,1,C.byref(out),C.byref(trace),C.byref(error))
        if result:
            _api()["poly_free"](result)
        if out.value:
            _domain_api()["domain_free"](out)
        return result,out.value,error.value,trace

    def test_registry_describes_only_two_executable_modules(self):
        records=modules()
        self.assertEqual({record["id"] for record in records},{1,2})
        self.assertTrue(all(record["backend"]=="native CPU" for record in records))
        self.assertTrue(all(record["operator_ids"] for record in records))
        self.assertEqual({record["id"]:record["operator_ids"] for record in records}[2],[9])
        self.assertEqual(C.sizeof(_ModuleInfo),72)
        self.assertEqual(C.sizeof(_ModuleTrace),40)

    def test_expression_arithmetic_calculus_and_evaluation_use_registry(self):
        context=Context(); x=context.symbol("x")
        expression=Fraction(1,3)*x**3 + x
        self.assertTrue(expression.module_plan["executed_via_registry"])
        self.assertEqual(expression.module_plan["operation"],1)
        derivative=expression.diff(x)
        self.assertEqual(derivative.module_plan["operation"],5)
        self.assertEqual(derivative.module_plan["native_calls"],1)
        self.assertEqual(derivative(x=2),5)
        self.assertEqual(derivative.module_plan["operation"],8)
        self.assertEqual(derivative.module_plan["backend"],"native CPU")
        self.assertIn("not elapsed",derivative.module_plan["cost_unit"])

    def test_native_checks_reject_wrong_context_carrier_and_domain_kind(self):
        x=Context().symbol("x"); y=Context().symbol("y")
        xd=x.domain; yd=y.domain
        self.addCleanup(xd.close); self.addCleanup(yd.close)
        for other,expected in ((yd,-6),
                (self.domain("formal",b"gf2",carrier="gf2"),-7),
                (self.domain("assignments",b"constraint"),-6)):
            result,out,code,trace=self.raw_apply(x,xd,y,other)
            self.assertFalse(result); self.assertFalse(out)
            self.assertEqual(code,expected)
            self.assertEqual(trace.native_calls,0)
        result,out,code,_=self.raw_apply(x,xd,operation=1,module=99)
        self.assertFalse(result); self.assertFalse(out); self.assertEqual(code,-8)

    def test_native_aggregate_data_role_and_source_origin_checks(self):
        db=torbles.from_frames({"t":pd.DataFrame(dict(x=[-2,1,3]))})
        self.addCleanup(db.close)
        bank=MomentBasis(db.query(db["t"]),["x"])
        self.addCleanup(bank.close)
        x=bank.variables["x"]; a=bank.parameter("a")
        function=bank.reduce((x-a)**2)
        self.assertEqual(function.module_plan["module_id"],2)
        fd=function.domain; xd=x.domain
        self.addCleanup(fd.close); self.addCleanup(xd.close)
        result,out,code,_=self.raw_apply(function,fd,x,xd)
        self.assertFalse(result); self.assertFalse(out); self.assertEqual(code,-6)
        result,out,code,_=self.raw_apply(function,fd,operation=5,variable=0)
        self.assertFalse(result); self.assertFalse(out); self.assertEqual(code,-6)
        source=bank.row_domain
        self.addCleanup(source.close)
        changed=self.domain("snapshot",source.key+b"changed",carrier="integer",nvars=1,ndata=1,
                            universe_rows=3)
        target=C.c_void_p(123); error=C.c_int32(); trace=_ModuleTrace()
        value=_domain_api()["module_reduce"](2,changed._open(),bank._open(),xd._open(),x._open(),
                    C.byref(target),C.byref(trace),C.byref(error))
        self.assertFalse(value); self.assertFalse(target.value)
        self.assertEqual(error.value,-6)

    def test_native_evaluation_failure_does_not_publish_partial_result(self):
        context=Context(); x=context.symbol("x")
        expression=((1<<63)-1)*x
        domain=expression.domain
        self.addCleanup(domain.close)
        values=(_Rational*8)(*(_Rational(0,1) for _ in range(8)))
        values[0]=_Rational(2,1)
        out=_Rational(17,19); trace=_ModuleTrace()
        rc=_domain_api()["module_evaluate"](1,domain._open(),expression._open(),values,C.byref(out),C.byref(trace))
        self.assertEqual(rc,-3)
        self.assertEqual((out.numerator,out.denominator),(17,19))
        self.assertEqual(trace.reason,-3)

    def test_registered_operations_execute_without_python_direct_operation_routes(self):
        db=torbles.from_frames({"t":pd.DataFrame(dict(x=[-2,1,3]))})
        self.addCleanup(db.close)
        bank=MomentBasis(db.query(db["t"]),["x"])
        self.addCleanup(bank.close)
        x=bank.variables["x"]; a=bank.parameter("a")
        forbidden=lambda *args: self.fail("Python bypassed executable module registry")
        routes={name:forbidden for name in ("poly_add","poly_sub","poly_mul","poly_pow",
                                            "poly_derivative","poly_integrate","poly_substitute",
                                            "poly_evaluate","bank_reduce")}
        with patch.dict(_api(),routes):
            function=bank.reduce((x-a)**2)
            self.assertEqual(function(a=2),18)
            self.assertEqual(function.diff(a)(a=2),8)
            self.assertEqual(function.integral(a,0,1)(),13)


if __name__=="__main__": unittest.main()
