-- Copyright 2026 Brian Richard RAMOS
-- SPDX-License-Identifier: Apache-2.0
-- The Star Schema Benchmark schema (O'Neil, O'Neil, Chen, Revilak; SSB spec rev. 3), as standard SQL DDL with its
-- keys. Column types are those of the DuckDB database bench/ssb/ssb_common.py builds from the dbgen files.
-- torbles reads the keys through INFORMATION_SCHEMA (python: torbles.from_duckdb(db, schema="schema.sql")).
CREATE TABLE customer (
  c_custkey INTEGER PRIMARY KEY, c_name VARCHAR, c_address VARCHAR, c_city VARCHAR, c_nation VARCHAR,
  c_region VARCHAR, c_phone VARCHAR, c_mktsegment VARCHAR);
CREATE TABLE supplier (
  s_suppkey INTEGER PRIMARY KEY, s_name VARCHAR, s_address VARCHAR, s_city VARCHAR, s_nation VARCHAR,
  s_region VARCHAR, s_phone VARCHAR);
CREATE TABLE part (
  p_partkey INTEGER PRIMARY KEY, p_name VARCHAR, p_mfgr VARCHAR, p_category VARCHAR, p_brand1 VARCHAR,
  p_color VARCHAR, p_type VARCHAR, p_size INTEGER, p_container VARCHAR);
CREATE TABLE date (
  d_datekey INTEGER PRIMARY KEY, d_date VARCHAR, d_dayofweek VARCHAR, d_month VARCHAR, d_year INTEGER,
  d_yearmonthnum INTEGER, d_yearmonth VARCHAR, d_daynuminweek INTEGER, d_daynuminmonth INTEGER, d_daynuminyear INTEGER,
  d_monthnuminyear INTEGER, d_weeknuminyear INTEGER, d_sellingseason VARCHAR, d_lastdayinweekfl INTEGER,
  d_lastdayinmonthfl INTEGER, d_holidayfl INTEGER, d_weekdayfl INTEGER);
CREATE TABLE lineorder (
  lo_orderkey INTEGER, lo_linenumber INTEGER,
  lo_custkey INTEGER REFERENCES customer (c_custkey),
  lo_partkey INTEGER REFERENCES part (p_partkey),
  lo_suppkey INTEGER REFERENCES supplier (s_suppkey),
  lo_orderdate INTEGER REFERENCES date (d_datekey),
  lo_orderpriority VARCHAR, lo_shippriority INTEGER, lo_quantity INTEGER, lo_extendedprice INTEGER,
  lo_ordtotalprice INTEGER, lo_discount INTEGER, lo_revenue INTEGER, lo_supplycost INTEGER, lo_tax INTEGER,
  lo_commitdate INTEGER REFERENCES date (d_datekey),
  lo_shipmode VARCHAR,
  PRIMARY KEY (lo_orderkey, lo_linenumber));
