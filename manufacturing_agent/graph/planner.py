from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import SQL_QUERY_TYPES, ExecutionPlan, SupervisorPlannerDecision, TaskSpec
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.util import _json_object

# ---------- graph/planner.py — SupervisorPlanner (사용자 의도 -> ExecutionPlan) ----------
# 답변을 만들지 않고, 어떤 worker task가 필요한지와 task params만 판단한다.
SUPERVISOR_PLANNER_SYS = (
    "너는 제조업 LangGraph Agent의 SupervisorPlanner다. 답변을 만들지 말고 필요한 worker task와 task params만 판단한다. "
    "정규식 키워드가 아니라 사용자 의도와 멀티턴 context를 의미로 해석한다. "
    "prediction은 현재 설비 수치 기반 위험 진단/부분 위험 진단이 필요할 때만 true다. 안전 자문만 있고 수치 진단 요청이 없으면 prediction=false다. "
    "evidence는 문서 근거, 매뉴얼, 절차, 원인 설명, 안전 절차, 해결 방법, 문서 기반 재발 방지 '절차', 일반 제조 QA에 필요하다. "
    "단, 직전 SQL 고장이력 결과에 기록된 대응·예방·재발 방지 '조치'를 이어서 더 묻는 후속질문은 evidence가 아니라 sql이다. "
    "sql은 과거 고장 이력, 고장 유형별 대응 방식, 반복 패턴, 다운타임, 현재 prediction failure_type과 유사한 과거 사례가 필요할 때만 true다. "
    "복합 요청이면 사용자가 요구한 산출물을 분리해 여러 task를 true로 둔다. 현재 상태, 과거 이력, 문서 근거가 함께 있으면 prediction/sql/evidence를 모두 포함한다. "
    "각 산출물은 독립적으로 판단한다. 한 task가 다른 task를 대체하지 않는다. 예: 문서 근거가 있어도 과거 유사 사례 조회를 대체할 수 없다. "
    "아무 worker도 필요 없다고 판단되면 일반 제조 질문 처리를 위해 evidence=true로 둔다. "
    "SQL 조회가 필요하면 sql_query_intents에 필요한 query type을 모두 넣는다. 가능한 값은 detail, aggregate 둘뿐이다. "
    "detail은 개별 고장 사례를 행 단위로 보는 조회다(유사 사례, 대응 조치, 다운타임 등). aggregate는 고장 유형/부품별 집계·반복 패턴·통계 조회다. "
    "SQLAgent는 failure_history 단일 테이블만 조회한다. 설비/자산 식별자 기반 task를 만들지 않는다. "
    "recent_turns_summary와 available_previous_*_summary는 사용자 의도 이해를 위한 참고 맥락이다. 이전 대화의 식별자성 표현을 SQL 조건으로 쓰지 않는다. "
    "중요한 제한: 사용자가 요청하지 않은 보강 task를 선제적으로 추가하지 마라. "
    "위험 진단 요청이라고 해서 자동으로 SQL 이력 조회나 문서 검색을 붙이지 않는다. "
    "SQL은 과거/최근/지난/고장 이력/대응 방식/유사 사례/반복 패턴/다운타임처럼 failure history 조회 의도가 명확하거나 이전 SQL artifact 후속질문일 때만 true다. "
    "Evidence는 문서/근거/매뉴얼/절차/방법/원인/안전 설명처럼 근거 설명 의도가 명확하거나 이전 evidence artifact 후속질문일 때만 true다. "
    "'재발 방지'·'대응 조치'·'예방 조치'라는 표현 하나만으로 evidence로 보내지 마라. 직전 SQL 이력 결과에 나온 그 조치를 더 자세히/이어서 묻는 후속질문이면 sql=true(evidence=false)다. "
    "'요약해줘'는 SQL 조회 결과 요약일 수 있으며, 그 자체만으로 문서 근거 task를 추가하지 않는다. "
    "판단 예시:\n"
    "- '토크 60만 있는데 고장 위험 진단해줘'처럼 현재 수치 진단이지만 이력/문서/절차 요청이 없으면 prediction=true, evidence=false, sql=false.\n"
    "- '최근 30일 고장 이력과 대응 방식을 조회해서 요약해줘'처럼 DB 이력 조회만 요청하면 sql=true, evidence=false, prediction=false.\n"
    "- '최근 TWF 사례에서 어떤 조치를 했어?'는 sql=true이며 sql_query_intents에는 detail을 포함한다.\n"
    "- '고장 유형별 반복 패턴과 다운타임을 정리해줘'는 sql=true이며 sql_query_intents에는 aggregate를 포함한다.\n"
    "- '현재 위험 진단, 과거 유사 사례, 점검 문서 근거까지'는 prediction=true, sql=true, evidence=true. sql_query_intents에는 detail을 포함한다.\n"
    "- '현재 고장 여부, 과거 사례의 조치, 해결 방법, 근거 문서'처럼 현재/과거/근거 산출물이 섞이면 prediction=true, sql=true, evidence=true.\n"
    "- '방금 근거 기준으로 더 구체화'처럼 이전 evidence artifact를 참조하면 evidence=true. '방금 조회한 고장 유형 중 HIGH만'처럼 이전 SQL artifact를 참조하면 sql=true.\n"
    "- '방금 이력에서 진행됐던 재발 방지/대응 조치를 더 자세히 알려줘'처럼 직전 SQL 결과의 조치를 이어 물으면 sql=true, evidence=false. sql_query_intents에는 detail을 포함한다.\n"
    "반드시 JSON만 출력하라: "
    "{\"intent\": \"prediction_diagnosis|document_qa|history_lookup|combined_analysis|general_manufacturing\", "
    "\"needs_prediction\": true/false, \"needs_evidence\": true/false, \"needs_sql\": true/false, "
    "\"evidence_required\": true/false, \"sql_query_intents\": [\"detail|aggregate\"], "
    "\"evidence_focus\": [\"검색/근거 초점\"], \"reason_summary\": \"짧은 이유\", \"confidence\": 0.0}"
)

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


