from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.context.policy import STANDARD_FEATURES, detect_injection
from manufacturing_agent.contracts.context import AgentContextPacket, ContextCarryoverDecision, ContextMode, ContextPacket, ContextResolution, DiagnosisContext, MachineValue
from manufacturing_agent.util import _json_object

# ---------- context/context_packer.py ----------
def _messages_to_recent_turns(messages: list, limit: int = 6) -> list[dict]:
    turns = []
    for m in messages or []:
        role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
        content = getattr(m, "content", "")
        if not content or detect_injection(str(content)):
            continue
        if role in {"human", "HumanMessage"}:
            role = "user"
        elif role in {"ai", "AIMessage"}:
            role = "assistant"
        turns.append({"role": role, "content": str(content), "created_at": "checkpoint"})
    return turns[-limit:]

# 멀티턴 대화 윈도우 정책:
# - 사용자 질문은 의도 추적에 중요하므로 윈도우 안에서 전부 유지한다.
# - AI 답변은 원문이 길어 토큰을 많이 쓰므로 최근 ASSISTANT_TURN_LIMIT개만 유지한다.
# - RECENT_TURN_WINDOW는 상류에서 가져오는 최근 대화 턴 상한(폭주 방지용 안전 한도)이다.
RECENT_TURN_WINDOW = 50
ASSISTANT_TURN_LIMIT = 4

def _dedup_turns(turns: list[dict]) -> list[dict]:
    """(role, content) 기준 중복 제거, 첫 등장 순서 보존.
    store 턴과 checkpoint(messages) 턴이 같은 대화를 중복 제공하는 것을 막는다."""
    seen: set = set()
    out: list[dict] = []
    for t in turns or []:
        key = (t.get("role"), str(t.get("content")))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out

def _summarize_recent_turns(turns: list[dict], limit: int = 6, chars: Optional[int] = None,
                            *, user_all: bool = False, assistant_limit: Optional[int] = ASSISTANT_TURN_LIMIT) -> str:
    """최근 대화를 'role:content' 한 줄 형태로 이어붙인다.
    chars=None(기본)이면 원문 전체를 자르지 않고 그대로 넣는다(개행만 공백으로 평탄화).
    user_all=True이면 사용자(user) 턴은 모두 유지하고, AI(assistant) 답변만 최근 assistant_limit개로 제한한다(순서 보존)."""
    seq = list(turns or [])
    if user_all:
        assistant_idx = [i for i, t in enumerate(seq) if t.get("role") == "assistant"]
        keep = set(assistant_idx) if assistant_limit is None else set(assistant_idx[-assistant_limit:])
        selected = [t for i, t in enumerate(seq) if t.get("role") != "assistant" or i in keep]
    else:
        selected = seq[-limit:]
    def _body(content: Any) -> str:
        text = str(content).replace(chr(10), " ")
        return text if chars is None else text[:chars]
    return " | ".join(f"{t['role']}:{_body(t['content'])}" for t in selected)

CONTEXT_CARRYOVER_SYS = (
    "너는 제조업 멀티턴 Agent의 컨텍스트 해석기다. 현재 사용자 발화가 이전 대화나 이전 artifact를 참조하는지 판단한다. "
    "정규식 키워드가 아니라 의미로 판단하라. 예: '그 이력', '방금 근거', '관련 조치', '이어서', '비슷한 사례'는 이전 artifact 참조일 수 있다. "
    "너는 task planner가 아니다. SQL 조회 필요 여부, 문서 검색 필요 여부, worker task 분해는 SupervisorPlanner가 담당한다. "
    "현재 질문이 이전 prediction/sql/evidence artifact를 참조하는지만 referenced_artifacts와 uses_previous_* 필드로 표시한다. "
    "고장 유형, 부품, 증상, 원인, 기간처럼 이전 artifact에서 이어받을 수 있는 맥락은 reason_summary에 짧게 설명하되, 최종 task 판단은 하지 않는다. "
    "단, 실제 실행이나 안전 승인을 대신 판단하지 말고 컨텍스트 carryover만 판단한다. 반드시 JSON만 출력하라. "
    "{\"is_followup\": true/false, \"uses_previous_prediction\": true/false, \"uses_previous_evidence\": true/false, "
    "\"uses_previous_sql\": true/false, \"inferred_time_range\": null 또는 객체, "
    "\"referenced_artifacts\": [\"prediction|sql|evidence\"], \"reason_summary\": \"짧은 이유\"}"
)

def _llm_context_carryover(user_message: str, selected: dict) -> ContextCarryoverDecision:
    recent_summary = _summarize_recent_turns(selected.get("recent_turns") or [], user_all=True)
    payload = {
        "current_user_message": user_message,
        "recent_turns_summary": recent_summary,
        "previous_prediction_summary": selected.get("previous_prediction_summary"),
        "previous_evidence_summary": selected.get("previous_evidence_summary"),
        "previous_sql_summary": selected.get("previous_sql_summary"),
    }
    raw = call_llm(CONTEXT_CARRYOVER_SYS, json.dumps(payload, ensure_ascii=False), tier="default")
    try:
        data = _json_object(raw)
        allowed_refs = {"prediction", "sql", "evidence"}
        refs = data.get("referenced_artifacts") or []
        if isinstance(refs, str):
            refs = [refs]
        data["referenced_artifacts"] = [x for x in refs if x in allowed_refs]
        decision = ContextCarryoverDecision.model_validate(data)
        if decision.is_followup and not decision.referenced_artifacts:
            refs = []
            if decision.uses_previous_prediction:
                refs.append("prediction")
            if decision.uses_previous_sql:
                refs.append("sql")
            if decision.uses_previous_evidence:
                refs.append("evidence")
            decision = decision.model_copy(update={"referenced_artifacts": refs})
        if not decision.is_followup:
            decision = decision.model_copy(update={
                "uses_previous_prediction": False,
                "uses_previous_evidence": False,
                "uses_previous_sql": False,
                "referenced_artifacts": [],
            })
        return decision
    except Exception as e:
        return ContextCarryoverDecision(reason_summary=f"context_carryover_parse_error: {type(e).__name__}")


CONTEXT_RESOLUTION_SYS = (
    "너는 제조업 Agent의 ContextManager다. 너는 task planner가 아니다. "
    "현재 질문이 이전 진단 입력 snapshot을 어떻게 참조하는지만 판단한다. "
    "mode는 CURRENT_ONLY, USE_ACTIVE, PATCH_ACTIVE, SELECT_HISTORY, REFER_ACTIVE_RESULT 중 하나다. "
    "CURRENT_ONLY는 현재 사용자가 직접 말한 값만 쓴다. 이전 feature 자동 보완은 금지다. "
    "USE_ACTIVE는 사용자가 방금/아까/같은 조건/이전 입력값 기준이라고 명시한 경우 active context 전체를 쓴다. "
    "PATCH_ACTIVE는 사용자가 특정 값만 바꾸라고 명시한 경우 active context 하나에 현재 변경값만 덮어쓴다. "
    "SELECT_HISTORY는 recent_contexts 중 사용자가 특정 과거 조건 하나를 지칭한 경우만 쓴다. 여러 context를 섞지 않는다. "
    "REFER_ACTIVE_RESULT는 재진단이 아니라 방금 결과/고장 유형/근거/이력만 참조하는 경우다. "
    "반드시 JSON만 출력하라: "
    "{\"mode\": \"CURRENT_ONLY|USE_ACTIVE|PATCH_ACTIVE|SELECT_HISTORY|REFER_ACTIVE_RESULT\", "
    "\"base_context_id\": null 또는 문자열, \"patch_values\": 객체, \"reason\": \"짧은 이유\"}"
)

def _context_brief(ctx: Optional[DiagnosisContext]) -> Optional[dict]:
    if not ctx:
        return None
    return {
        "id": ctx.id,
        "turn_id": ctx.turn_id,
        "features": ctx.features,
        "failure_types": ctx.failure_types,
        "prediction_summary": ctx.prediction_summary[:500],
        "created_at": ctx.created_at,
    }

def _contexts_by_id(selected: dict) -> dict[str, DiagnosisContext]:
    out: dict[str, DiagnosisContext] = {}
    active = selected.get("active_context")
    if active:
        out[active.id] = active
    for ctx in selected.get("recent_contexts") or []:
        out[ctx.id] = ctx
    return out

def _filter_patch_values(values: dict, current_values: dict) -> dict[str, Any]:
    allowed = set(current_values)
    return {k: v for k, v in (values or {}).items() if k in allowed}

def _llm_context_resolution(user_message: str, selected: dict) -> ContextResolution:
    payload = {
        "current_user_message": user_message,
        "current_values_extracted_from_this_turn": selected.get("current_values") or {},
        "active_context": _context_brief(selected.get("active_context")),
        "recent_contexts": [_context_brief(c) for c in (selected.get("recent_contexts") or [])],
        "recent_turns_summary": _summarize_recent_turns(selected.get("recent_turns") or [], user_all=True),
        "previous_prediction_summary_available": bool(selected.get("previous_prediction_summary")),
        "previous_sql_summary_available": bool(selected.get("previous_sql_summary")),
        "previous_evidence_summary_available": bool(selected.get("previous_evidence_summary")),
    }
    raw = call_llm(CONTEXT_RESOLUTION_SYS, json.dumps(payload, ensure_ascii=False), tier="default")
    data = _json_object(raw)
    mode = data.get("mode") if data.get("mode") in ContextMode.__args__ else "CURRENT_ONLY"
    return ContextResolution(
        mode=mode,
        current_values=selected.get("current_values") or {},
        base_context_id=data.get("base_context_id"),
        patch_values=_filter_patch_values(data.get("patch_values") or {}, selected.get("current_values") or {}),
        reason=str(data.get("reason") or "LLM context resolution"),
    )

def resolve_context(user_message: str, selected: dict) -> ContextResolution:
    current_values = dict(selected.get("current_values") or {})
    contexts = _contexts_by_id(selected)
    active = selected.get("active_context")
    warnings: list[str] = []
    try:
        decision = _llm_context_resolution(user_message, selected)
    except Exception as e:
        decision = ContextResolution(
            mode="CURRENT_ONLY",
            current_values=current_values,
            resolved_features=current_values,
            changed_features=list(current_values.keys()),
            warnings=[f"context_resolution_llm_fallback: {type(e).__name__}"],
            reason="Context resolution failed; current values only",
        )
        return decision

    mode = decision.mode
    base_context_id = decision.base_context_id
    base: Optional[DiagnosisContext] = None

    if mode in {"USE_ACTIVE", "PATCH_ACTIVE"}:
        base = active
        if base and not base_context_id:
            base_context_id = base.id
    elif mode == "REFER_ACTIVE_RESULT":
        base = active
        if base and not base_context_id:
            base_context_id = base.id
    elif mode == "SELECT_HISTORY":
        if base_context_id and base_context_id in contexts:
            base = contexts[base_context_id]
        else:
            warnings.append("특정 과거 조건을 안정적으로 선택하지 못해 현재 입력만 사용합니다.")
            mode = "CURRENT_ONLY"

    if mode in {"USE_ACTIVE", "PATCH_ACTIVE"} and not base:
        warnings.append("재사용할 active 진단 context가 없어 현재 입력만 사용합니다.")
        mode = "CURRENT_ONLY"

    patch_values = dict(decision.patch_values or {})
    if mode in {"PATCH_ACTIVE", "SELECT_HISTORY"} and current_values and not patch_values:
        patch_values = current_values

    if mode == "CURRENT_ONLY":
        resolved = current_values
        changed = list(current_values.keys())
        reused: list[str] = []
        base_context_id = None
    elif mode == "REFER_ACTIVE_RESULT":
        resolved = {}
        changed = []
        reused = []
    elif mode == "USE_ACTIVE" and base:
        resolved = dict(base.features or {})
        changed = []
        reused = list(resolved.keys())
    elif mode in {"PATCH_ACTIVE", "SELECT_HISTORY"} and base:
        if not patch_values:
            warnings.append("변경할 현재 값이 없어 base context를 그대로 사용합니다.")
        resolved = dict(base.features or {})
        for key, value in patch_values.items():
            resolved[key] = value
        changed = list(patch_values.keys())
        reused = [k for k in resolved.keys() if k not in changed]
    else:
        resolved = current_values
        changed = list(current_values.keys())
        reused = []
        base_context_id = None
        mode = "CURRENT_ONLY"

    return ContextResolution(
        mode=mode,
        current_values=current_values,
        base_context_id=base_context_id,
        patch_values=patch_values,
        resolved_features=resolved,
        changed_features=changed,
        reused_features=reused,
        warnings=list(decision.warnings or []) + warnings,
        reason=decision.reason,
    )

def pack_contexts(user_message: str, merged: dict[str, MachineValue],
                  selected: dict, warnings: list[str]) -> tuple[ContextPacket, dict[str, AgentContextPacket]]:
    """ContextPacket + Agent별 AgentContextPacket 생성."""
    recent_summary = _summarize_recent_turns(selected.get("recent_turns") or [], user_all=True)
    carry = selected.get("context_carryover") or ContextCarryoverDecision()
    prior_results = {
        "prediction_summary": selected.get("previous_prediction_summary"),
        "evidence_summary": selected.get("previous_evidence_summary"),
        "sql_summary": selected.get("previous_sql_summary"),
        "is_followup": carry.is_followup,
        "uses_previous_prediction": carry.uses_previous_prediction,
        "uses_previous_evidence": carry.uses_previous_evidence,
        "uses_previous_sql": carry.uses_previous_sql,
        "referenced_artifacts": carry.referenced_artifacts,
        "reason_summary": carry.reason_summary,
    }
    user_constraints = {}
    if carry.inferred_time_range:
        user_constraints["time_range"] = carry.inferred_time_range

    resolution = selected.get("context_resolution")
    packet = ContextPacket(
        current_question=user_message,
        recent_turns_summary=recent_summary,
        current_values=selected.get("current_values") or {},
        context_resolution=resolution,
        selected_machine_values=merged,
        previous_prediction_result=selected.get("previous_prediction_result"),
        previous_prediction_summary=selected.get("previous_prediction_summary"),
        previous_evidence_summary=selected.get("previous_evidence_summary"),
        previous_sql_summary=selected.get("previous_sql_summary"),
        context_carryover=carry,
        user_constraints=user_constraints,
        context_warnings=warnings,
    )

    feats = {k: v.value for k, v in merged.items()}
    missing = [f for f in STANDARD_FEATURES if f not in merged]

    context_meta = {
        "context_mode": resolution.mode if resolution else "CURRENT_ONLY",
        "base_context_id": resolution.base_context_id if resolution else None,
        "changed_features": resolution.changed_features if resolution else [],
        "reused_features": resolution.reused_features if resolution else [],
    }
    agent_ctx = {
        "prediction_agent": AgentContextPacket(
            agent_name="prediction_agent", current_question=user_message,
            selected_context={"features": feats, "missing": missing,
                              "sources": {k: v.source for k, v in merged.items()},
                              "stale": [k for k, v in merged.items() if v.is_stale], **context_meta}),
        "evidence_agent": AgentContextPacket(
            agent_name="evidence_agent", current_question=user_message,
            selected_context={"warnings": warnings, "recent_summary": recent_summary, **context_meta},
            prior_results=prior_results),
        "sql_agent": AgentContextPacket(
            agent_name="sql_agent", current_question=user_message,
            selected_context={"recent_summary": recent_summary, "failure_history_only": True, **context_meta},
            prior_results=prior_results),
        "final_answer": AgentContextPacket(
            agent_name="final_answer", current_question=user_message,
            selected_context={"recent_summary": recent_summary, "warnings": warnings, **context_meta},
            prior_results=prior_results),
    }
    return packet, agent_ctx
print("context_packer 정의 완료")
