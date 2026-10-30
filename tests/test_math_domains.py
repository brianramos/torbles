"""Native domain identity must preserve semantic boundaries and ownership."""
import ctypes as C
import unittest

from torbles.mathematics import Context
from torbles.math_domains import Domain, DomainError, _DomainSpec, _domain_api


class TypedMathematicalDomains(unittest.TestCase):
    def domain(self,*args,**kwargs):
        value=Domain(*args,**kwargs)
        self.addCleanup(value.close)
        return value

    def test_exact_binary_identity_and_owned_key(self):
        key=bytearray(b"scope\0A")
        left=self.domain("snapshot",key,carrier="integer",nvars=1,ndata=1,universe_rows=100)
        equal=self.domain("snapshot",bytes(key),carrier="integer",nvars=1,ndata=1,universe_rows=100)
        key[-1]=ord("B")
        changed=self.domain("snapshot",key,carrier="integer",nvars=1,ndata=1,universe_rows=100)
        self.assertEqual(left.key,b"scope\0A")
        self.assertEqual(left,equal)
        self.assertNotEqual(left,changed)
        self.assertGreater(left.bytes,len(left.key))
        clone=left.clone()
        self.addCleanup(clone.close)
        left.close()
        self.assertEqual(clone,equal)

    def test_revision_universe_validity_grouping_and_layout_are_identity(self):
        base=dict(carrier="integer",nvars=2,ndata=2,revision=7,universe_rows=100)
        original=self.domain("snapshot",b"same-sql",**base)
        for change in (dict(revision=8),dict(universe_rows=101),dict(validity="sql"),
                       dict(grouping="dense"),dict(ndata=1),dict(carrier="rational")):
            with self.subTest(change=change):
                other=self.domain("snapshot",b"same-sql",**{**base,**change})
                self.assertNotEqual(original,other)

    def test_domain_kinds_remain_distinct(self):
        domains=[self.domain(kind,b"same-key") for kind in
                 ("physical","snapshot","assignments","formal","geometry")]
        self.assertEqual(len({value.spec["kind"] for value in domains}),5)
        for left in range(5):
            for right in range(left+1,5):
                self.assertNotEqual(domains[left],domains[right])

    def test_origin_is_owned_and_aggregate_flag_is_not_context_identity(self):
        source=self.domain("snapshot",b"rows",carrier="integer",nvars=1,ndata=1,universe_rows=10)
        formal=self.domain("formal",b"context",ndata=1,origin=source)
        aggregate=formal.with_aggregate()
        self.addCleanup(aggregate.close)
        self.assertNotEqual(formal,aggregate)
        self.assertTrue(formal.context_equal(aggregate))
        source.close()
        origin=aggregate.origin
        self.addCleanup(origin.close)
        self.assertEqual(origin.key,b"rows")
        self.assertEqual(origin.spec["universe_rows"],10)
        returned=aggregate.spec
        returned["revision"]=999
        self.assertEqual(aggregate.spec["revision"],0)

    def test_invalid_origin_and_aggregate_requests_are_rejected(self):
        with self.assertRaises(DomainError):
            self.domain("formal",b"missing-origin",ndata=1)
        with self.assertRaises(DomainError):
            self.domain("formal",b"missing-origin",aggregate=True)
        source=self.domain("snapshot",b"source",carrier="integer",nvars=1,ndata=1)
        with self.assertRaises(DomainError):
            self.domain("formal",b"wrong-layout",ndata=2,origin=source)
        with self.assertRaises(DomainError):
            self.domain("geometry",b"wrong-origin",origin=source)

    def test_formal_context_identity_survives_new_symbol_names(self):
        context=Context()
        x=context.symbol("x")
        before=x.domain
        self.addCleanup(before.close)
        y=context.symbol("y")
        after=y.domain
        self.addCleanup(after.close)
        self.assertEqual(before,after)
        self.assertEqual((x+y).evaluate(x=2,y=3),5)
        foreign=Context().symbol("x").domain
        self.addCleanup(foreign.close)
        self.assertFalse(before.context_equal(foreign))

    def test_closed_handles_and_resource_limits(self):
        value=self.domain("formal",b"closed")
        value.close(); value.close()
        with self.assertRaises(RuntimeError):
            value.clone()
        for key in (b"",b"x"*65537):
            with self.assertRaises(ValueError):
                self.domain("formal",key)
        for kwargs in (dict(nvars=9),dict(ndata=5),dict(revision=-1),dict(universe_rows=1<<64)):
            with self.assertRaises(ValueError):
                self.domain("formal",b"bad",**kwargs)

    def test_native_abi_validation_and_atomic_key_copy(self):
        native=_domain_api()
        spec=_DomainSpec(999,4,2,8,0,0,0,0,0,0)
        error=C.c_int32()
        self.assertFalse(native["domain_create"](C.byref(spec),b"a",1,None,C.byref(error)))
        self.assertEqual(error.value,-1)
        value=self.domain("formal",b"binary\0key")
        output=C.create_string_buffer(b"sentinel",9)
        before=output.raw
        self.assertEqual(native["domain_key_copy"](value._open(),output,1),-5)
        self.assertEqual(output.raw,before)
        self.assertEqual(C.sizeof(_DomainSpec),48)


if __name__=="__main__": unittest.main()
