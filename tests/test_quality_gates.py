"""Layer 1 — quality_gates 상태 라우팅 단위 테스트.

실행:
    uv run pytest tests/test_quality_gates.py -v

API 키: 불필요 (artifact fixture 직접 주입, LLM 호출 없음)
대상:   manufacturing_agent/gates/quality_gates.py
        → prediction_gate, evidence_gate, sql_gate
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from manufacturing_agent.contracts.context import (
    EvidenceArtifact,
    ExecutionPlan,
    PredictionResult,
    SQLHistoryArtifact,
    TaskSpec,
)
from manufacturing_agent.gates.quality_gates import (
    evidence_gate,
    prediction_gate,
    sql_gate,
)


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def _state(**kwargs) -> dict:
    """ManufacturingState 최소 fixture."""
    base = {"active_task_id": None, "execution_plan": None, "gate_reports": []}
    base.update(kwargs)
    return base


def _pred(**kwargs) -> PredictionResult:
    defaults = dict(
        status="OK", confidence="high",
        risk_flags=[], missing_features=[], limitations=[], summary="",
        context_mode="NEW", changed_features=[], reused_features=[],
        safety_hints=[],
    )
    defaults.update(kwargs)
    return PredictionResult(**defaults)


def _ev(status: str, docs: int = 0, citations: int = 0, **kwargs) -> EvidenceArtifact:
    documents = [{"source_id": f"C{i}", "snippet": "text"} for i in range(docs)]
    cits = [{"citation_id": f"C{i}"} for i in range(citations)]
    defaults = dict(
        status=status, documents=documents, citations=cits,
        evidence_summary="", limitations=[],
    )
    defaults.update(kwargs)
    return EvidenceArtifact(**defaults)


def _sql(status: str, rows: int = 0, **kwargs) -> SQLHistoryArtifact:
    row_data = [{"id": i} for i in range(rows)]
    defaults = dict(
        status=status, query_type="detail",
        sql="SELECT * FROM failure_history LIMIT 10",
        rows=row_data, results=[], summary="", limitations=[],
    )
    defaults.update(kwargs)
    return SQLHistoryArtifact(**defaults)


def _gate_status(report_list: list) -> str:
    return report_list[-1]["status"]


# ── prediction_gate ────────────────────────────────────────────────────────────

class TestPredictionGate:
    def test_ok_passes(self):
        result = prediction_gate(_state(prediction_result=_pred(status="OK")))
        assert _gate_status(result["gate_reports"]) == "PASS"

    def test_partial_passes(self):
        result = prediction_gate(_state(prediction_result=_pred(status="PARTIAL")))
        assert _gate_status(result["gate_reports"]) == "PASS"

    def test_needs_input(self):
        result = prediction_gate(_state(prediction_result=_pred(status="NEEDS_INPUT")))
        assert _gate_status(result["gate_reports"]) == "NEEDS_USER_INPUT"

    def test_skipped_pass_with_warnings(self):
        result = prediction_gate(_state(prediction_result=_pred(status="SKIPPED")))
        assert _gate_status(result["gate_reports"]) == "PASS_WITH_WARNINGS"

    def test_fail_retryable(self):
        result = prediction_gate(_state(prediction_result=_pred(status="FAIL")))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"

    def test_none_retryable(self):
        result = prediction_gate(_state(prediction_result=None))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"


# ── sql_gate ───────────────────────────────────────────────────────────────────

class TestSQLGate:
    def test_ok_passes(self):
        result = sql_gate(_state(sql_result=_sql(status="OK", rows=3)))
        assert _gate_status(result["gate_reports"]) == "PASS"

    def test_empty_pass_with_warnings(self):
        result = sql_gate(_state(sql_result=_sql(status="EMPTY")))
        assert _gate_status(result["gate_reports"]) == "PASS_WITH_WARNINGS"

    def test_invalid_request_needs_user_input(self):
        result = sql_gate(_state(sql_result=_sql(status="INVALID_REQUEST", summary="조건 부족")))
        assert _gate_status(result["gate_reports"]) == "NEEDS_USER_INPUT"

    def test_blocked_no_rerun_goes_to_block(self):
        # rerun_count == max_reruns → 재시도 없음 → BLOCK
        task = TaskSpec(task_id="t1", task_type="sql", rerun_count=1, max_reruns=1)
        plan = ExecutionPlan(intent="history_lookup", tasks=[task])
        result = sql_gate(_state(
            sql_result=_sql(status="BLOCKED"),
            active_task_id="t1",
            execution_plan=plan,
        ))
        assert _gate_status(result["gate_reports"]) == "BLOCK"

    def test_fail_with_rerun_plan_repair(self):
        # rerun_count < max_reruns → 재시도 가능 → PLAN_REPAIR_REQUIRED
        task = TaskSpec(task_id="t1", task_type="sql", rerun_count=0, max_reruns=1)
        plan = ExecutionPlan(intent="history_lookup", tasks=[task])
        result = sql_gate(_state(
            sql_result=_sql(status="FAIL"),
            active_task_id="t1",
            execution_plan=plan,
        ))
        assert _gate_status(result["gate_reports"]) == "PLAN_REPAIR_REQUIRED"

    def test_none_retryable(self):
        result = sql_gate(_state(sql_result=None))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"


# ── evidence_gate ──────────────────────────────────────────────────────────────

class TestEvidenceGate:
    def test_ok_with_docs_passes(self):
        result = evidence_gate(_state(evidence_bundle=_ev("OK", docs=2)))
        assert _gate_status(result["gate_reports"]) == "PASS"

    def test_ok_no_docs_retryable(self):
        # "OK" + docs=0: gate는 ev.documents가 falsy면 OK 분기에 진입 못함 → else → RETRYABLE_FAIL
        result = evidence_gate(_state(evidence_bundle=_ev("OK", docs=0)))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"

    def test_empty_not_required_warns(self):
        result = evidence_gate(_state(evidence_bundle=_ev("EMPTY")))
        assert _gate_status(result["gate_reports"]) == "PASS_WITH_WARNINGS"

    def test_empty_required_retryable(self):
        # required=True인 task를 주입
        task = TaskSpec(
            task_id="e1", task_type="evidence",
            params={"evidence_required": True},
            rerun_count=0, max_reruns=1,
        )
        plan = ExecutionPlan(intent="document_qa", tasks=[task])
        result = evidence_gate(_state(
            evidence_bundle=_ev("EMPTY"),
            active_task_id="e1",
            execution_plan=plan,
        ))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"

    def test_none_retryable(self):
        result = evidence_gate(_state(evidence_bundle=None))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"

    def test_ok_with_min_docs_check(self):
        # min_docs=3인데 docs=2 → RETRYABLE_FAIL
        task = TaskSpec(
            task_id="e1", task_type="evidence",
            params={"min_docs": 3, "evidence_required": True},
            rerun_count=0, max_reruns=1,
        )
        plan = ExecutionPlan(intent="document_qa", tasks=[task])
        result = evidence_gate(_state(
            evidence_bundle=_ev("OK", docs=2),
            active_task_id="e1",
            execution_plan=plan,
        ))
        assert _gate_status(result["gate_reports"]) == "RETRYABLE_FAIL"
