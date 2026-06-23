from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.agents.evidence_agent import DEFAULT_SQL_DEPS, get_active_task, get_active_task_criteria, get_active_task_params, validate_sql_query
from manufacturing_agent.contracts.context import FinalAnswer, GateReport, OutputSafetyDecision
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.gates.intake_gate import SAFETY_BLOCK_MESSAGE
from manufacturing_agent.util import _coerce_bool, _json_object

# ---------- gates/prediction_gate.py ----------
def _active_task_id(state: ManufacturingState, task_type: str) -> Optional[str]:
    active = state.get("active_task_id")
    plan = state.get("execution_plan")
    if active and plan:
        for t in plan.tasks:
            if t.task_id == active and t.task_type == task_type:
                return active
    return active

def prediction_gate(state: ManufacturingState) -> dict:
    """PredictionResult artifact 품질을 검증하고 GateReport만 추가한다."""
    pred = state.get("prediction_result")
    task_id = _active_task_id(state, "prediction")
    if pred is None:
        status, hint, reason = "RETRYABLE_FAIL", "prediction_agent", "prediction_result 없음"
    elif pred.status in ("OK", "PARTIAL"):
        status, hint, reason = "PASS", None, f"status={pred.status}"
    elif pred.status == "NEEDS_INPUT":
        status, hint, reason = "NEEDS_USER_INPUT", "final_answer", f"missing={pred.missing_features}"
    elif pred.status == "SKIPPED":
        status, hint, reason = "PASS_WITH_WARNINGS", None, "진단할 설비 feature가 부족하거나 요청 범위가 아님"
    else:
        status, hint, reason = "RETRYABLE_FAIL", "prediction_agent", f"status={pred.status}"
    report = GateReport(task_id=task_id, gate_name="prediction_gate", status=status, route_hint=hint,
                        reason=reason, feedback="입력 feature와 부분 위험 진단 결과를 재확인하세요." if status == "RETRYABLE_FAIL" else None,
                        diagnostics={"missing": getattr(pred, "missing_features", []) if pred else []})
    return {"gate_reports": state.get("gate_reports", []) + [report.model_dump()]}

# ---------- gates/evidence_gate.py ----------
def evidence_gate(state: ManufacturingState) -> dict:
    ev = state.get("evidence_bundle")
    task_id = _active_task_id(state, "evidence")
    task = get_active_task(state, expected_type="evidence")
    params = get_active_task_params(state, expected_type="evidence")
    criteria = get_active_task_criteria(state, expected_type="evidence")
    required = bool(params.get("evidence_required") or params.get("focus") or criteria.get("require_citation"))
    allow_empty = bool(criteria.get("allow_empty", not required))
    min_docs = int(params.get("min_docs") or criteria.get("min_docs") or (1 if required else 0))
    require_citation = bool(params.get("require_citation") or criteria.get("require_citation"))
    if ev is None:
        status, hint, reason = "RETRYABLE_FAIL", "evidence_agent", "EvidenceArtifact 없음"
    elif ev.status == "OK" and ev.documents:
        docs_count = len(ev.documents)
        citations_count = len(ev.citations or [])
        if docs_count < min_docs and not getattr(ev, "is_retry", False):
            status, hint, reason = "RETRYABLE_FAIL", "evidence_agent", f"문서 수가 success criteria보다 부족: docs={docs_count}, min_docs={min_docs}"
        elif require_citation and citations_count == 0 and not getattr(ev, "is_retry", False):
            status, hint, reason = "RETRYABLE_FAIL", "evidence_agent", "citation이 필요하지만 생성되지 않음"
        elif docs_count < min_docs or (require_citation and citations_count == 0):
            status, hint, reason = "PASS_WITH_WARNINGS", None, f"근거 품질 경고: docs={docs_count}, citations={citations_count}"
        else:
            status, hint, reason = "PASS", None, f"docs={docs_count}"
    elif ev.status in ("EMPTY", "LOW_RELEVANCE"):
        if required and not getattr(ev, "is_retry", False):
            status, hint, reason = "RETRYABLE_FAIL", "evidence_agent", f"문서 근거 요청이 있었지만 검색 결과 부족: status={ev.status}, docs={len(ev.documents)}"
        elif required and task and task.rerun_count < task.max_reruns:
            status, hint, reason = "PLAN_REPAIR_REQUIRED", "supervisor_replanner", f"retry 후에도 문서 근거 부족: status={ev.status}, docs={len(ev.documents)}"
        elif required and not allow_empty:
            status, hint, reason = "PASS_WITH_WARNINGS", None, f"필수 문서 근거가 부족하지만 retry 후 제한 답변으로 진행: status={ev.status}, docs={len(ev.documents)}"
        else:
            status, hint, reason = "PASS_WITH_WARNINGS", None, f"status={ev.status}, docs={len(ev.documents)}"
    else:
        status, hint, reason = "RETRYABLE_FAIL", "evidence_agent", f"status={ev.status}"
    report = GateReport(task_id=task_id, gate_name="evidence_gate", status=status, route_hint=hint,
                        reason=reason, feedback="failure_type, component, symptom, root_cause, corrective/preventive action, 재발 방지 절차 중심으로 검색 질의를 확장하세요." if status in {"RETRYABLE_FAIL", "PLAN_REPAIR_REQUIRED"} else None,
                        diagnostics={"docs": len(ev.documents) if ev else 0, "citations": len(ev.citations or []) if ev else 0,
                                     "required": required, "min_docs": min_docs, "require_citation": require_citation,
                                     "is_retry": getattr(ev, "is_retry", False) if ev else False})
    return {"gate_reports": state.get("gate_reports", []) + [report.model_dump()]}

# ---------- gates/sql_gate.py ----------
def sql_gate(state: ManufacturingState) -> dict:
    artifact = state.get("sql_result")
    task_id = _active_task_id(state, "sql")
    task = get_active_task(state, expected_type="sql")
    deps = DEFAULT_SQL_DEPS
    status, hint, reason = "RETRYABLE_FAIL", "sql_agent", "SQLHistoryArtifact 없음"
    diagnostics = {}
    if artifact is not None:
        diagnostics = {"rows": len(artifact.rows), "query_type": artifact.query_type, "has_sql": bool(artifact.sql),
                       "result_count": len(getattr(artifact, "results", []) or []),
                       "query_types": [r.query_type for r in (getattr(artifact, "results", []) or [])],
                       "result_statuses": {r.query_type: r.status for r in (getattr(artifact, "results", []) or [])}}
        try:
            query_results = getattr(artifact, "results", []) or []
            if query_results:
                for result in query_results:
                    if result.sql:
                        validate_sql_query(result.sql, deps)
                    if len(result.rows) > deps.max_rows:
                        raise ValueError(f"rows가 max_rows({deps.max_rows})를 초과했습니다.")
            elif artifact.sql:
                validate_sql_query(artifact.sql, deps)
                if len(artifact.rows) > deps.max_rows:
                    raise ValueError(f"rows가 max_rows({deps.max_rows})를 초과했습니다.")
            if artifact.status == "OK":
                non_ok = [f"{r.query_type}:{r.status}" for r in query_results if getattr(r, "status", "OK") != "OK"]
                if non_ok:
                    status, hint, reason = "PASS_WITH_WARNINGS", None, "SQL 부분 경고: " + ", ".join(non_ok)
                else:
                    status, hint, reason = "PASS", None, "SQL 조회 성공"
            elif artifact.status == "EMPTY":
                status, hint, reason = "PASS_WITH_WARNINGS", None, "조회 결과 없음"
            elif artifact.status == "INVALID_REQUEST":
                status, hint, reason = "NEEDS_USER_INPUT", "final_answer", artifact.summary or "SQL 조회 조건 부족"
            elif artifact.status == "BLOCKED":
                if task and task.rerun_count < task.max_reruns:
                    status, hint, reason = "PLAN_REPAIR_REQUIRED", "supervisor_replanner", artifact.error_message or "SQL 정책 차단: 보정 후 재시도 필요"
                else:
                    status, hint, reason = "BLOCK", "final_answer", artifact.error_message or "SQL 정책 차단"
            else:
                if task and task.rerun_count < task.max_reruns:
                    status, hint, reason = "PLAN_REPAIR_REQUIRED", "supervisor_replanner", artifact.error_message or "SQL 조회 실패: plan repair 필요"
                else:
                    status, hint, reason = "RETRYABLE_FAIL", "sql_agent", artifact.error_message or "SQL 조회 실패"
        except ValueError as e:
            if task and task.rerun_count < task.max_reruns:
                status, hint, reason = "PLAN_REPAIR_REQUIRED", "supervisor_replanner", str(e)
            else:
                status, hint, reason = "BLOCK", "final_answer", str(e)
    report = GateReport(task_id=task_id, gate_name="sql_gate", status=status, route_hint=hint,
                        reason=reason, feedback="failure_history schema, allowed columns, SELECT-only, LIMIT, date filter를 맞춰 SQL을 다시 생성하세요." if status in {"RETRYABLE_FAIL", "PLAN_REPAIR_REQUIRED"} else None,
                        diagnostics=diagnostics)
    return {"gate_reports": state.get("gate_reports", []) + [report.model_dump()]}

# ---------- gates/output_safety_gate.py ----------
OUTPUT_SAFETY_SYS = (
    "너는 제조 설비 AI Agent의 최종 답변 안전 게이트다. <answer> 안의 텍스트만 검사한다. "
    "새로운 기술 답변을 만들지 말고, 위험 표현이 있을 때만 안전하게 대체한다.\n"
    "차단/대체 기준: 점검 없이 재가동을 지시, 안전장치 우회/해제를 안내, 경고/알람 무시 운전을 승인, "
    "모델이 실제 현장 조치·승인을 대신한다고 표현, 근거 없이 안전하다고 단정.\n"
    "안전한 거절, 위험 진단, 문서 근거 요약, 현장 책임자 확인 권고는 통과한다.\n"
    "반드시 JSON만 출력하라: "
    "{\"pass_output\": true/false, \"reason\": \"ok|empty|unsafe_instruction|overconfident_safety|policy_violation\", "
    "\"safe_answer\": \"대체 답변 또는 null\", \"warnings\": [\"경고\"]}"
)

_VALID_OUTPUT_REASONS = {"ok", "empty", "unsafe_instruction", "overconfident_safety", "policy_violation"}

def _normalize_output_safety_payload(data: dict) -> dict:
    reason = str(data.get("reason", "ok")).strip().lower()
    if reason not in _VALID_OUTPUT_REASONS:
        reason = "policy_violation"
    warnings = data.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    return {
        "pass_output": _coerce_bool(data.get("pass_output"), default=(reason == "ok")),
        "reason": reason,
        "safe_answer": data.get("safe_answer"),
        "warnings": [str(x) for x in warnings],
    }

OUTPUT_FORBIDDEN_PATTERNS = [
    r"점검\s*(없이|전에?|안\s*하고)\s*(재?가동|기동|운전).{0,20}(해도\s*(됩니다|된다|돼)|하세요|하라|가능|승인|계속)",
    r"안전\s*장치\S*\s*(우회|해제|끄|꺼|무시).{0,30}(하세요|하라|해도|됩니다|가능|운전|계속|진행)",
    r"(경고|알람|위험)\s*\S*\s*무시.{0,30}(가동|운전|계속|진행|하세요|하라)",
]

_SAFE_NEGATION = re.compile(r"피하|하지\s*마|마라|마세요|말아|금지|않|불가|위험|안\s*됩니다|안\s*돼|삼가|자제|주의")
def _contains_unsafe_execution_instruction(answer: str) -> bool:
    """위험 실행 '지시'만 잡는다. 매치 주변에 부정/경고어가 있으면 안전 권고로 보고 통과(오차단 방지)."""
    if not answer:
        return False
    for p in OUTPUT_FORBIDDEN_PATTERNS:
        for m in re.finditer(p, answer, re.I | re.S):
            seg = answer[max(0, m.start() - 12): m.end() + 12]
            if _SAFE_NEGATION.search(seg):
                continue
            return True
    return False

def _llm_output_safety(answer: str) -> OutputSafetyDecision:
    raw = call_llm(OUTPUT_SAFETY_SYS, f"<answer>\n{answer}\n</answer>")
    try:
        return OutputSafetyDecision.model_validate(_normalize_output_safety_payload(_json_object(raw)))
    except Exception as e:
        return OutputSafetyDecision(
            pass_output=False,
            reason="policy_violation",
            safe_answer=SAFETY_BLOCK_MESSAGE,
            warnings=[f"output_safety_parse_error: {type(e).__name__}"],
        )

def output_safety_gate(state: ManufacturingState) -> dict:
    fa = state.get("final_answer")
    updates = {}
    if not (fa and fa.answer.strip()):
        decision = OutputSafetyDecision(pass_output=False, reason="empty", safe_answer="현재 입력만으로는 안전하게 답변을 생성할 수 없습니다.")
    elif _contains_unsafe_execution_instruction(fa.answer):
        decision = OutputSafetyDecision(
            pass_output=False,
            reason="unsafe_instruction",
            safe_answer=SAFETY_BLOCK_MESSAGE,
            warnings=["deterministic_output_safety_backstop"],
        )
    else:
        decision = _llm_output_safety(fa.answer)
        if decision.pass_output and _contains_unsafe_execution_instruction(fa.answer):
            decision = OutputSafetyDecision(
                pass_output=False,
                reason="unsafe_instruction",
                safe_answer=SAFETY_BLOCK_MESSAGE,
                warnings=list(decision.warnings) + ["post_llm_deterministic_output_safety_backstop"],
            )
        if (not decision.pass_output) and decision.safe_answer and _contains_unsafe_execution_instruction(decision.safe_answer):
            decision = decision.model_copy(update={
                "safe_answer": SAFETY_BLOCK_MESSAGE,
                "warnings": list(decision.warnings) + ["safe_answer_deterministic_output_safety_backstop"],
            })
    if decision.pass_output:
        status = "PASS"
    else:
        status = "BLOCK"
        safe_answer = decision.safe_answer or SAFETY_BLOCK_MESSAGE
        old_warnings = list(fa.warnings) if fa else []
        updates["final_answer"] = FinalAnswer(
            answer=safe_answer,
            citations=fa.citations if fa else [],
            warnings=old_warnings + decision.warnings,
            missing_inputs=fa.missing_inputs if fa else [],
        )
    report = GateReport(
        gate_name="output_safety_gate",
        status=status,
        reason=decision.reason,
        diagnostics=decision.model_dump(),
    )
    updates["gate_reports"] = state.get("gate_reports", []) + [report.model_dump()]
    return updates

print("gates 정의 완료 (prediction/evidence/sql/output_safety)")
