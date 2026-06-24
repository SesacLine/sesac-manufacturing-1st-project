from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import SQL_QUERY_TYPES, ExecutionPlan, SupervisorPlannerDecision, TaskSpec
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.util import _json_object
from manufacturing_agent.prompts.supervisor_planner import SUPERVISOR_PLANNER_SYS

# ---------- graph/planner.py — SupervisorPlanner (사용자 의도 -> ExecutionPlan) ----------
# 답변을 만들지 않고, 어떤 worker task가 필요한지와 task params만 판단한다.
# SUPERVISOR_PLANNER_SYS 시스템 프롬프트는 manufacturing_agent/prompts/ 로 분리됨

_PLANNER_INTENTS = {
    "prediction_diagnosis", "document_qa", "history_lookup",
    "combined_analysis", "general_manufacturing",
}


def _parse_supervisor_planner_decision(raw: str) -> SupervisorPlannerDecision:
    """LLM 출력을 계약 수준에서만 정규화한다(사용자 문구를 정규식으로 재분류하지 않는다)."""
    data = _json_object(raw)
    if data.get("intent") not in _PLANNER_INTENTS:
        data["intent"] = "general_manufacturing"
    qtypes = data.get("sql_query_intents") or []
    if isinstance(qtypes, str):
        qtypes = [qtypes]
    data["sql_query_intents"] = [q for q in qtypes if q in SQL_QUERY_TYPES]
    focus = data.get("evidence_focus") or []
    if isinstance(focus, str):
        focus = [focus]
    data["evidence_focus"] = [str(x) for x in focus]
    if data.get("confidence") is None:
        data["confidence"] = 0.0
    return SupervisorPlannerDecision.model_validate(data)


def _normalize_planner_decision(decision: SupervisorPlannerDecision,
                                has_structured_input: bool) -> SupervisorPlannerDecision:
    """구조화 입력 보정 + worker 없음 fallback + 복합 intent 보정을 한 곳에서 적용한다."""
    updates: dict = {}
    if has_structured_input:
        updates["needs_prediction"] = True
    needs = [updates.get("needs_prediction", decision.needs_prediction),
             decision.needs_evidence, decision.needs_sql]
    if not any(needs):
        updates.update(needs_evidence=True, intent="general_manufacturing")
        needs = [needs[0], True, needs[2]]
    if sum(bool(x) for x in needs) > 1 and decision.intent not in {"combined_analysis"}:
        updates["intent"] = "combined_analysis"
    return decision.model_copy(update=updates) if updates else decision


def _supervisor_planner_payload(state: ManufacturingState) -> dict:
    packet = state.get("context_packet")
    carry = packet.context_carryover if packet else None
    structured = state.get("input_features")
    if hasattr(structured, "model_dump"):
        structured = structured.model_dump(exclude_none=True)
    return {
        "user_message": state.get("user_message", ""),
        "has_structured_input_features": bool(structured),
        "input_features": structured or None,
        "recent_turns_summary": packet.recent_turns_summary if packet else "",
        "available_previous_prediction_summary": packet.previous_prediction_summary if packet else None,
        "available_previous_evidence_summary": packet.previous_evidence_summary if packet else None,
        "available_previous_sql_summary": packet.previous_sql_summary if packet else None,
        "previous_prediction_summary": packet.previous_prediction_summary if (packet and carry and carry.uses_previous_prediction) else None,
        "previous_evidence_summary": packet.previous_evidence_summary if (packet and carry and carry.uses_previous_evidence) else None,
        "previous_sql_summary": packet.previous_sql_summary if (packet and carry and carry.uses_previous_sql) else None,
        "current_constraints": packet.user_constraints if packet else {},
        "context_carryover": carry.model_dump() if carry else None,
    }


def _llm_supervisor_planner_decision(state: ManufacturingState) -> SupervisorPlannerDecision:
    payload = _supervisor_planner_payload(state)
    raw = call_llm(SUPERVISOR_PLANNER_SYS, json.dumps(payload, ensure_ascii=False), tier="default")
    try:
        decision = _parse_supervisor_planner_decision(raw)
    except Exception as e:
        decision = SupervisorPlannerDecision(
            needs_evidence=True, evidence_required=False,
            reason_summary=f"supervisor_planner_parse_error: {type(e).__name__}; evidence fallback",
            confidence=0.0,
        )
    return _normalize_planner_decision(decision, payload["has_structured_input_features"])


def _planner_retrieval_profile(decision: SupervisorPlannerDecision) -> str:
    if decision.needs_prediction:
        return "prediction_plus_rag"
    return "troubleshooting_rag"


def _prediction_task(decision: SupervisorPlannerDecision) -> TaskSpec:
    return TaskSpec(
        task_id="prediction_1", task_type="prediction",
        reason=decision.reason_summary or "SupervisorPlanner가 위험 진단 task 필요로 판단",
        params={"diagnosis_mode": "current_or_partial", "allow_partial": True, "allow_stale_context": False},
        success_criteria={"allow_status": ["OK", "PARTIAL", "NEEDS_INPUT"]},
    )


def _sql_task(decision: SupervisorPlannerDecision) -> TaskSpec:
    return TaskSpec(
        task_id="sql_1", task_type="sql",
        reason=decision.reason_summary or "SupervisorPlanner가 이력 조회 task 필요로 판단",
        params={"query_types": list(decision.sql_query_intents), "failure_type": None,
                "default_time_window_days": 30},
        success_criteria={"require_executed_sql": True, "allow_empty": True},
    )


def _evidence_task(decision: SupervisorPlannerDecision) -> TaskSpec:
    evidence_required = bool(decision.evidence_required or decision.evidence_focus)
    return TaskSpec(
        task_id="evidence_1", task_type="evidence",
        reason=decision.reason_summary or "SupervisorPlanner가 문서 근거 task 필요로 판단",
        params={"retrieval_profile": _planner_retrieval_profile(decision),
                "evidence_required": evidence_required, "focus": list(decision.evidence_focus),
                "min_docs": 2 if evidence_required else 0, "require_citation": evidence_required},
        success_criteria={"allow_empty": not evidence_required, "require_citation": evidence_required},
    )


def _general_evidence_task() -> TaskSpec:
    return TaskSpec(
        task_id="evidence_1", task_type="evidence",
        reason="일반 제조 질문은 문서 근거 검색 우선",
        params={"retrieval_profile": "troubleshooting_rag", "evidence_required": False,
                "focus": [], "min_docs": 0, "require_citation": False},
        success_criteria={"allow_empty": True, "require_citation": False},
    )


def supervisor_planner_node(state: ManufacturingState) -> dict:
    decision = _llm_supervisor_planner_decision(state)
    builders = [
        (decision.needs_prediction, _prediction_task),
        (decision.needs_sql, _sql_task),
        (decision.needs_evidence, _evidence_task),
    ]
    tasks: list[TaskSpec] = [build(decision) for need, build in builders if need]
    if not tasks:
        tasks.append(_general_evidence_task())
    tasks.append(TaskSpec(task_id="final_1", task_type="final_answer",
                          depends_on=[t.task_id for t in tasks],
                          reason="선행 task artifact를 종합해 최종 답변 생성"))
    plan = ExecutionPlan(
        intent=decision.intent, tasks=tasks, created_by="llm",
        reason_summary=decision.reason_summary or "LLM typed planner가 사용자 요청을 task로 분해함",
        confidence=decision.confidence,
    )
    return {"execution_plan": plan, "supervisor_planner_decision": decision,
            "active_task_id": None, "intent": plan.intent}


