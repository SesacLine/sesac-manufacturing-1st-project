"""Layer 2 — validate_sql_query 결정론적 단위 테스트 + execute_readonly_sql 실행 테스트.

실행:
    uv run pytest tests/test_sql_quality.py -v

API 키: 불필요 (validate_sql_query는 regex 전용, execute_readonly_sql은 SQLite 전용)
대상:   manufacturing_agent/agents/sql_agent.py
        → validate_sql_query(), execute_readonly_sql(), DEFAULT_SQL_DEPS
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest


def _load():
    from manufacturing_agent.agents.sql_agent import (
        DEFAULT_SQL_DEPS,
        SQLAgentDeps,
        execute_readonly_sql,
        validate_sql_query,
    )
    return validate_sql_query, execute_readonly_sql, DEFAULT_SQL_DEPS, SQLAgentDeps


_validate, _exec_sql, _deps, _Deps = _load()

# 공통 허용 deps
_STANDARD = _Deps(
    db_uri=str(_deps.db_uri),
    allowed_tables=["failure_history"],
    max_rows=50,
)


# ── 통과해야 할 SQL ────────────────────────────────────────────────────────────

VALID_QUERIES = [
    (
        "SELECT * FROM failure_history LIMIT 10",
        "기본 SELECT",
    ),
    (
        "SELECT id, event_date FROM failure_history WHERE failure_type = 'TWF' LIMIT 50",
        "LIMIT 50 (경계)",
    ),
    (
        "SELECT failure_type, COUNT(*) AS case_count FROM failure_history GROUP BY failure_type LIMIT 20",
        "집계 쿼리",
    ),
    (
        "SELECT a.id FROM failure_history AS a JOIN failure_history AS b ON a.id = b.id LIMIT 5",
        "self-join (같은 허용 테이블)",
    ),
    (
        "SELECT * FROM failure_history LIMIT ?",
        "parameterized LIMIT",
    ),
]

@pytest.mark.parametrize("sql,desc", VALID_QUERIES)
def test_valid_sql_passes(sql, desc):
    try:
        _validate(sql, _STANDARD)
    except ValueError as e:
        pytest.fail(f"유효한 SQL이 거부됨 [{desc}]: {e}")


# ── 차단해야 할 SQL ────────────────────────────────────────────────────────────

INVALID_QUERIES = [
    # 빈 SQL
    ("", "빈 문자열"),
    ("   ", "공백만"),
    # SELECT가 아닌 DML
    ("INSERT INTO failure_history (id) VALUES (1) LIMIT 1", "INSERT"),
    ("UPDATE failure_history SET notes = 'x' LIMIT 1", "UPDATE"),
    ("DELETE FROM failure_history LIMIT 1", "DELETE"),
    # DDL
    ("DROP TABLE failure_history", "DROP TABLE"),
    ("ALTER TABLE failure_history ADD COLUMN x TEXT", "ALTER TABLE"),
    ("TRUNCATE TABLE failure_history", "TRUNCATE"),
    ("CREATE TABLE x (id INT)", "CREATE TABLE"),
    # 위험 pragma / attach
    ("PRAGMA integrity_check", "PRAGMA"),
    ("ATTACH DATABASE '/tmp/x.db' AS x", "ATTACH"),
    # 멀티 스테이트먼트 (세미콜론 포함)
    (
        "SELECT * FROM failure_history LIMIT 10; DROP TABLE failure_history",
        "멀티 스테이트먼트",
    ),
    # 허용되지 않은 테이블
    (
        "SELECT * FROM users LIMIT 10",
        "허용되지 않은 테이블",
    ),
    (
        "SELECT * FROM sqlite_master LIMIT 10",
        "시스템 테이블",
    ),
    # LIMIT 없음
    ("SELECT * FROM failure_history", "LIMIT 없음"),
    # LIMIT 초과
    (
        "SELECT * FROM failure_history LIMIT 51",
        "LIMIT 51 (max_rows=50 초과)",
    ),
    (
        "SELECT * FROM failure_history LIMIT 100",
        "LIMIT 100 초과",
    ),
    # FROM 없음 (테이블 확인 불가)
    ("SELECT 1", "테이블 없는 SELECT"),
    # SELECT로 시작하지 않음 (공백 있어도 이미 걸러짐)
    ("WITH cte AS (SELECT 1) SELECT * FROM cte LIMIT 1", "CTE — failure_history 아님"),
]

@pytest.mark.parametrize("sql,desc", INVALID_QUERIES)
def test_invalid_sql_blocked(sql, desc):
    with pytest.raises(ValueError, match=r".+"):
        _validate(sql, _STANDARD)


# ── LIMIT 경계값 ───────────────────────────────────────────────────────────────

class TestLimitBoundary:
    def test_limit_exactly_max_passes(self):
        sql = "SELECT * FROM failure_history LIMIT 50"
        _validate(sql, _STANDARD)

    def test_limit_one_over_max_raises(self):
        sql = "SELECT * FROM failure_history LIMIT 51"
        with pytest.raises(ValueError):
            _validate(sql, _STANDARD)

    def test_custom_max_rows(self):
        custom = _Deps(
            db_uri=str(_deps.db_uri),
            allowed_tables=["failure_history"],
            max_rows=10,
        )
        with pytest.raises(ValueError):
            _validate("SELECT * FROM failure_history LIMIT 11", custom)

    def test_parameterized_limit_ok(self):
        sql = "SELECT * FROM failure_history LIMIT ?"
        _validate(sql, _STANDARD)


# ── allowed_tables 격리 ────────────────────────────────────────────────────────

class TestAllowedTables:
    def test_disallowed_table_raises(self):
        with pytest.raises(ValueError, match="허용되지 않은"):
            _validate("SELECT * FROM users LIMIT 10", _STANDARD)

    def test_allowed_table_passes(self):
        _validate("SELECT * FROM failure_history LIMIT 10", _STANDARD)

    def test_empty_allowed_list_allows_any(self):
        open_deps = _Deps(
            db_uri=str(_deps.db_uri),
            allowed_tables=[],  # 제한 없음
            max_rows=50,
        )
        _validate("SELECT * FROM any_table LIMIT 10", open_deps)

    def test_join_foreign_table_blocked(self):
        with pytest.raises(ValueError):
            _validate(
                "SELECT a.id FROM failure_history a JOIN users u ON a.id = u.id LIMIT 10",
                _STANDARD,
            )


# ── execute_readonly_sql (DB 존재 시에만 실행) ──────────────────────────────────

@pytest.fixture(scope="module")
def db_available() -> bool:
    """bootstrapped SQLite DB 존재 여부."""
    db_path = (str(_deps.db_uri or "")).replace("sqlite:///", "", 1)
    return bool(db_path and os.path.exists(db_path))


class TestExecuteReadonlySQL:
    def test_valid_select_returns_list(self, db_available):
        if not db_available:
            pytest.skip("SQLite DB 없음 — bootstrap_failure_history_db() 실행 필요")
        rows = _exec_sql(
            "SELECT * FROM failure_history LIMIT 5",
            deps=_STANDARD,
        )
        assert isinstance(rows, list)
        assert len(rows) <= 5

    def test_empty_condition_returns_empty_list(self, db_available):
        if not db_available:
            pytest.skip("SQLite DB 없음")
        rows = _exec_sql(
            "SELECT * FROM failure_history WHERE failure_type = '__NONEXISTENT__' LIMIT 10",
            deps=_STANDARD,
        )
        assert rows == []

    def test_invalid_sql_raises_before_execution(self):
        with pytest.raises(ValueError):
            _exec_sql(
                "SELECT * FROM failure_history",  # LIMIT 없음
                deps=_STANDARD,
            )

    def test_forbidden_table_raises_before_execution(self):
        with pytest.raises(ValueError):
            _exec_sql(
                "SELECT * FROM sqlite_master LIMIT 10",
                deps=_STANDARD,
            )
