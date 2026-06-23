"""ContextEngine 결정 로직 deterministic 테스트.

LLM 호출(call_llm)은 monkeypatch로 가짜 응답을 주입해 네트워크 없이 검증한다.
실행: 프로젝트 루트에서  python -m pytest tests/test_context_engine.py -q
"""
from __future__ import annotations
import json
import datetime as _dt

import pytest

from manufacturing_agent.context import engine as engine_mod
from manufacturing_agent.context.engine import decide_context
from manufacturing_agent.context.normalizer import normalize_context
from manufacturing_agent.contracts.context import ContextResolution, DiagnosisContext


def _diag(ctx_id: str, features: dict, *, created_at: str) -> DiagnosisContext:
    return DiagnosisContext(
        id=ctx_id, turn_id="t1", user_id="u1", thread_id="th1",
        features=features, failure_types=["TWF"], prediction_summary="prev",
        created_at=created_at, is_safe_to_reuse=True,
    )


def _boom(*_a, **_k):
    raise AssertionError("call_llm 이 호출되면 안 된다 (short-circuit 기대)")


def test_short_circuit_skips_llm(monkeypatch):
    monkeypatch.setattr(engine_mod, "call_llm", _boom)
    selected = {
        "current_values": {"torque": 60.0},
        "active_context": None, "recent_contexts": [], "recent_turns": [],
        "previous_prediction_summary": None, "previous_evidence_summary": None,
        "previous_sql_summary": None,
    }
    d = decide_context("토크 60으로 진단", selected)
    assert d.llm_skipped is True
    assert d.mode == "CURRENT_ONLY"
    assert d.resolved_features == {"torque": 60.0}
    assert d.changed_features == ["torque"]
    assert d.reused_features == []


def test_patch_active_merges_base_with_current(monkeypatch):
    base = _diag("diag-1", {"torque": 50.0, "tool_wear": 200.0},
                created_at=_dt.datetime.now().isoformat(timespec="seconds"))
    selected = {
        "current_values": {"torque": 70.0},
        "active_context": base, "recent_contexts": [base], "recent_turns": [{"role": "user", "content": "이전"}],
        "previous_prediction_summary": "prev",
    }

    def fake_llm(_sys, _user, **_k):
        return json.dumps({
            "is_followup": True, "uses_previous_prediction": True,
            "referenced_artifacts": ["prediction"],
            "mode": "PATCH_ACTIVE", "base_context_id": "diag-1",
            "patch_values": {"torque": 70.0}, "reason": "토크만 변경",
        })

    monkeypatch.setattr(engine_mod, "call_llm", fake_llm)
    d = decide_context("토크만 70으로 바꾸면?", selected)
    assert d.mode == "PATCH_ACTIVE"
    assert d.resolved_features == {"torque": 70.0, "tool_wear": 200.0}
    assert d.changed_features == ["torque"]
    assert d.reused_features == ["tool_wear"]
    assert d.is_followup is True


def test_use_active_without_base_downgrades(monkeypatch):
    selected = {
        "current_values": {},
        "active_context": None, "recent_contexts": [], "recent_turns": [{"role": "user", "content": "x"}],
        "previous_prediction_summary": "prev",
    }

    def fake_llm(_sys, _user, **_k):
        return json.dumps({"is_followup": True, "mode": "USE_ACTIVE", "reason": "같은 조건"})

    monkeypatch.setattr(engine_mod, "call_llm", fake_llm)
    d = decide_context("아까 그 조건 그대로", selected)
    assert d.mode == "CURRENT_ONLY"
    assert any("active 진단 context가 없" in w for w in d.warnings)


def test_patch_values_whitelisted_to_current(monkeypatch):
    base = _diag("diag-9", {"torque": 50.0, "tool_wear": 200.0},
                created_at=_dt.datetime.now().isoformat(timespec="seconds"))
    selected = {
        "current_values": {"torque": 70.0},
        "active_context": base, "recent_contexts": [base], "recent_turns": [{"role": "user", "content": "x"}],
    }

    def fake_llm(_sys, _user, **_k):
        # LLM이 현재 턴에 없는 tool_wear를 patch하려 해도 코드가 차단해야 한다.
        return json.dumps({"is_followup": True, "mode": "PATCH_ACTIVE", "base_context_id": "diag-9",
                           "patch_values": {"torque": 70.0, "tool_wear": 999.0}, "reason": "x"})

    monkeypatch.setattr(engine_mod, "call_llm", fake_llm)
    d = decide_context("토크만 70", selected)
    assert d.patch_values == {"torque": 70.0}
    assert d.resolved_features["tool_wear"] == 200.0  # 999로 오염되지 않음


def test_parse_failure_falls_back_current_only(monkeypatch):
    # 선행 맥락(이전 진단 요약)이 있어야 short-circuit을 건너뛰고 LLM 경로를 탄다.
    selected = {"current_values": {"torque": 60.0}, "recent_turns": [{"role": "user", "content": "x"}],
                "previous_prediction_summary": "이전 진단 요약"}
    monkeypatch.setattr(engine_mod, "call_llm", lambda *_a, **_k: "이건 JSON이 아님")
    d = decide_context("질문", selected)
    assert d.mode == "CURRENT_ONLY"
    assert d.llm_skipped is False
    assert any("context_decision_llm_fallback" in w for w in d.warnings)


def test_short_circuit_ignores_chat_only_turns(monkeypatch):
    # 저장된 context/이전 artifact 없이 채팅 턴만 있으면 LLM 없이 CURRENT_ONLY로 단락된다.
    monkeypatch.setattr(engine_mod, "call_llm", _boom)
    selected = {"current_values": {"torque": 60.0},
                "recent_turns": [{"role": "user", "content": "이전 발화"}, {"role": "assistant", "content": "이전 답변"}],
                "active_context": None, "recent_contexts": [],
                "previous_prediction_summary": None, "previous_evidence_summary": None, "previous_sql_summary": None}
    d = decide_context("토크 60", selected)
    assert d.llm_skipped is True
    assert d.mode == "CURRENT_ONLY"


def test_normalize_marks_stale_for_old_base():
    old = (_dt.datetime.now() - _dt.timedelta(hours=5)).isoformat(timespec="seconds")
    base = _diag("diag-old", {"torque": 50.0}, created_at=old)
    resolution = ContextResolution(
        mode="USE_ACTIVE", current_values={}, base_context_id="diag-old",
        resolved_features={"torque": 50.0}, changed_features=[], reused_features=["torque"],
    )
    selected = {"context_resolution": resolution, "active_context": base, "recent_contexts": [base]}
    merged, warnings = normalize_context(selected)
    assert merged["torque"].is_stale is True
    assert merged["torque"].source == "active_context"
    assert any("오래되어" in w for w in warnings)


def test_normalize_fresh_base_not_stale():
    fresh = _dt.datetime.now().isoformat(timespec="seconds")
    base = _diag("diag-new", {"torque": 50.0}, created_at=fresh)
    resolution = ContextResolution(
        mode="USE_ACTIVE", current_values={}, base_context_id="diag-new",
        resolved_features={"torque": 50.0}, changed_features=[], reused_features=["torque"],
    )
    selected = {"context_resolution": resolution, "active_context": base, "recent_contexts": [base]}
    merged, _ = normalize_context(selected)
    assert merged["torque"].is_stale is False
