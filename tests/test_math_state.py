"""Compatible state reuse must never change selected rows or bank ownership."""
import ctypes as C
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torbles
from torbles import algq
from torbles.mathematics import MomentBasis, AlgebraBudgetError
from torbles.math_domains import MomentCache, _CacheStats


class CompatibleMomentStates(unittest.TestCase):
    def make(self):
        values=dict(x=[-3,-3,1,4,7,1],y=[5,5,-2,8,0,-2],flag=[1,1,1,0,1,2])
        db=torbles.from_frames({"t":pd.DataFrame(values)},keep_plain=True)
        self.addCleanup(db.close)
        return db,db["t"],values

    def cache(self,**kwargs):
        cache=MomentCache(**kwargs)
        self.addCleanup(cache.close)
        return cache

    def basis(self,query,columns,cache,**kwargs):
        bank=MomentBasis(query,columns,cache=cache,**kwargs)
        self.addCleanup(bank.close)
        return bank

    def test_separate_equivalent_queries_reuse_with_zero_fact_queries(self):
        db,table,values=self.make()
        cache=self.cache()
        first=self.basis(db.query(table).where(table.flag==1),["x","y"],cache)
        self.assertFalse(first.metadata["cache_hit"])
        self.assertEqual(first.metadata["native_query_calls"],6)
        with patch.object(algq.Query,"run",side_effect=AssertionError("cache hit scanned data")):
            second=self.basis(db.query(table).where(table.flag==1),[table.x,table.y],cache)
        self.assertTrue(second.metadata["cache_hit"])
        self.assertEqual(second.metadata["native_query_calls"],0)
        self.assertEqual(second.metadata["actual_devices"],())
        self.assertEqual(first.moment((1,1)),second.moment((1,1)))
        self.assertEqual(cache.stats["hits"],1)
        self.assertEqual(cache.stats["misses"],1)
        self.assertEqual(cache.stats["puts"],1)

    def test_predicate_and_ordered_basis_are_part_of_identity(self):
        db,table,values=self.make()
        cache=self.cache()
        left=self.basis(db.query(table).where(table.flag==1),["x","y"],cache)
        other=self.basis(db.query(table).where(table.flag==2),["x","y"],cache)
        reverse=self.basis(db.query(table).where(table.flag==1),["y","x"],cache)
        self.assertFalse(other.metadata["cache_hit"])
        self.assertFalse(reverse.metadata["cache_hit"])
        selected=[i for i,flag in enumerate(values["flag"]) if flag==1]
        self.assertEqual(left.moment((1,0)),sum(values["x"][i] for i in selected))
        self.assertEqual(reverse.moment((1,0)),sum(values["y"][i] for i in selected))
        self.assertEqual(other.row_count,1)
        self.assertEqual(cache.stats["entries"],3)

    def test_revision_and_codec_changes_prevent_reuse(self):
        db,table,values=self.make()
        cache=self.cache()
        original=self.basis(db.query(table),["x"],cache)
        table.flip(0,0); table.repair()
        revised=self.basis(db.query(table),["x"],cache)
        self.assertFalse(revised.metadata["cache_hit"])
        self.assertEqual(original.moment((1,)),revised.moment((1,)))
        codec=table.codecs["x"]
        try:
            table.codecs["x"]=type(codec)(codec.offset+1,codec.sql_type)
            changed=self.basis(db.query(table),["x"],cache)
            self.assertFalse(changed.metadata["cache_hit"])
            self.assertEqual(changed.moment((1,)),original.moment((1,))+original.row_count)
        finally:
            table.codecs["x"]=codec

    def test_mutable_membership_and_different_tables_do_not_alias(self):
        db,table,_=self.make()
        cache=self.cache()
        self.basis(db.query(table),["x"],cache)
        old=table.used_bm.copy()
        try:
            table.used_bm[0]^=np.uint64(1)
            changed=self.basis(db.query(table),["x"],cache)
            self.assertFalse(changed.metadata["cache_hit"])
        finally:
            table.used_bm[:]=old
        other,second,_=self.make()
        separate=self.basis(other.query(second),["x"],cache)
        self.assertFalse(separate.metadata["cache_hit"])

    def test_cached_bank_and_functions_survive_input_and_cache_release(self):
        db,table,values=self.make()
        cache=self.cache()
        initial=self.basis(db.query(table),["x"],cache)
        initial.close()
        restored=self.basis(db.query(table),["x"],cache)
        x=restored.variables["x"]; a=restored.parameter("a")
        function=restored.reduce((x-a)**2)
        cache.close(); cache.close(); restored.close()
        self.assertEqual(function(a=2),sum((x-2)**2 for x in values["x"]))
        self.assertTrue(function.domain.spec["aggregate"])

    def test_capacity_lru_eviction_and_clear(self):
        db,table,_=self.make()
        cache=self.cache(capacity=2)
        for flag in (0,1):
            self.basis(db.query(table).where(table.flag==flag),["x"],cache)
        touched=self.basis(db.query(table).where(table.flag==0),["x"],cache)
        self.assertTrue(touched.metadata["cache_hit"])
        self.basis(db.query(table).where(table.flag==2),["x"],cache)
        with patch.object(algq.Query,"run",side_effect=AssertionError("most recent state was evicted")):
            self.basis(db.query(table).where(table.flag==0),["x"],cache)
        evicted=self.basis(db.query(table).where(table.flag==1),["x"],cache)
        self.assertFalse(evicted.metadata["cache_hit"])
        self.assertEqual(cache.stats["entries"],2)
        self.assertEqual(cache.stats["evictions"],2)
        cache.clear()
        self.assertEqual(cache.stats["entries"],0)
        self.assertEqual(cache.stats["hits"],0)
        self.assertEqual(cache.stats["puts"],0)

    def test_tiny_payload_budget_declines_caching_without_losing_result(self):
        db,table,values=self.make()
        probe=self.cache(capacity=1)
        fixed_bytes=probe.stats["bytes"]
        bounded=self.cache(capacity=1,max_bytes=fixed_bytes)
        bank=self.basis(db.query(table),["x"],bounded)
        self.assertEqual(bank.moment((2,)),sum(x*x for x in values["x"]))
        self.assertFalse(bank.metadata["cache_hit"])
        self.assertEqual(bounded.stats["entries"],0)
        self.assertLessEqual(bounded.stats["bytes"],bounded.stats["max_bytes"])
        again=self.basis(db.query(table),["x"],bounded)
        self.assertFalse(again.metadata["cache_hit"])

    def test_cached_values_can_cross_requested_backend_without_false_gpu_work(self):
        db,table,_=self.make()
        cache=self.cache()
        first=self.basis(db.query(table),["x"],cache,engine="alg")
        with patch.object(algq.Query,"run",side_effect=AssertionError("compatible cached values rescanned")):
            hit=self.basis(db.query(table),["x"],cache,engine="gpu")
        self.assertTrue(hit.metadata["cache_hit"])
        self.assertEqual(hit.metadata["actual_devices"],())
        self.assertIsNone(hit.metadata["gpu_executor"])
        self.assertEqual(hit.metadata["engine_requested"],"gpu")
        self.assertEqual(first.moment((2,)),hit.moment((2,)))

    def test_closed_invalid_and_empty_states(self):
        for capacity in (0,129):
            with self.assertRaises(ValueError): MomentCache(capacity)
        with self.assertRaises(AlgebraBudgetError): MomentCache(1,max_bytes=0)
        db,table,_=self.make()
        cache=self.cache()
        first=self.basis(db.query(table).where(table.flag<0),["x"],cache)
        with patch.object(algq.Query,"run",side_effect=AssertionError("empty state cache miss")):
            second=self.basis(db.query(table).where(table.flag<0),["x"],cache)
        self.assertEqual(first.row_count,0)
        self.assertTrue(second.metadata["cache_hit"])
        self.assertEqual(second.row_count,0)
        cache.close()
        with self.assertRaises(RuntimeError):
            self.basis(db.query(table),["x"],cache)
        self.assertEqual(C.sizeof(_CacheStats),56)

    def test_metadata_is_serializable_and_returned_containers_are_independent(self):
        db,table,_=self.make()
        cache=self.cache()
        for bank in (self.basis(db.query(table),["x"],cache),
                     self.basis(db.query(table),["x"],cache)):
            metadata=bank.metadata
            encoded=json.dumps(metadata)
            self.assertIn("row_domain",encoded)
            original=bank.metadata["row_domain"].copy()
            metadata["row_domain"]["kind"]="invented"
            self.assertEqual(bank.metadata["row_domain"],original)
            x=bank.variables["x"]
            function=bank.reduce(x*x)
            json.dumps(function.metadata)
            changed=function.metadata
            changed["row_domain"]["kind"]="invented"
            self.assertEqual(function.metadata["row_domain"],original)

    def test_table_lifetime_identity_does_not_depend_on_allocator_nonreuse(self):
        db,table,_=self.make()
        cache=self.cache()
        first=self.basis(db.query(table),["x"],cache)
        # Model a distinct wrapper/lifetime with identical pointer and metadata.
        # Do not free or alter native storage to manufacture allocator reuse.
        alias=copy.copy(table)
        alias.__dict__.pop("_math_domain_identity",None)
        second=self.basis(algq.Query(alias),["x"],cache)
        self.assertNotEqual(table._math_domain_identity,alias._math_domain_identity)
        self.assertFalse(second.metadata["cache_hit"])
        self.assertEqual(first.moment((1,)),second.moment((1,)))


if __name__=="__main__": unittest.main()
