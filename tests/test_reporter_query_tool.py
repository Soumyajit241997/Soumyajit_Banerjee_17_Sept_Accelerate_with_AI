"""Regression test for "AttributeError: 'NoneType' object has no attribute
'head'": run_sql_query called `.fetchdf()` and assumed it always returns a
DataFrame. Some statements can leave `.fetchdf()` returning None without
raising, which crashed the whole Reporter agent run on `df.head(200)`.
run_sql_query must now detect that and return a clear JSON error instead.
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd

from agents.reporter import run_sql_query


def test_run_sql_query_returns_rows_for_a_real_select():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES ('Apparel', 0.3), ('Books', 0.02)) AS v(category, return_rate)")
    result = json.loads(run_sql_query(con, "SELECT * FROM t ORDER BY return_rate DESC"))
    assert result[0]["category"] == "Apparel"


def test_run_sql_query_reports_sql_errors_without_crashing():
    con = duckdb.connect(":memory:")
    result = json.loads(run_sql_query(con, "SELECT * FROM does_not_exist"))
    assert "error" in result


class _NoneFetchCursor:
    def fetchdf(self):
        return None


class _FakeConnectionReturningNone:
    """Simulates a connection whose .execute(sql).fetchdf() returns None
    without raising -- the exact shape that crashed production."""

    def execute(self, sql, *args, **kwargs):
        return _NoneFetchCursor()


def test_run_sql_query_survives_none_fetchdf_instead_of_crashing():
    result = json.loads(run_sql_query(_FakeConnectionReturningNone(), "SELECT 1"))
    assert "error" in result
    assert "no result set" in result["error"]
