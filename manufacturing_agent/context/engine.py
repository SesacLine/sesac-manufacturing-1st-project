from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.context.packer import _summarize_recent_turns
from manufacturing_agent.contracts.context import ContextDecision, ContextMode, DiagnosisContext
from manufacturing_agent.util import _json_object

# ---------- context/engine.py — 단일 ContextDecision (carryover + resolution 통합) ----------
# 기존 _llm_context_carryover + resolve_context(2콜)을 단일 LLM 1콜 + short-circuit으로 대체한다.
# LLM 판단을 그대로 믿지 않고 mode 강등/patch 화이트리스트/feature 계산을 코드가 결정적으로 검증한다.

CONTEXT_DECISION_SYS = (
    "너는 제조업 멀티턴 Agent의 ContextManager다. 너는 task planner가 아니다. "
    "현재 사용자 발화가 (1) 이전 artifact(prediction/sql/evidence)를 참조하는지와 "
    "(2) 이전 진단 입력 snapshot을 어떻게 재사용하는지를 한 번에 판단한다. 정규식이 아니라 의미로 판단하라.\n"
    "참조 판단: '그 이력', '방금 근거', '관련 조치', '이어서', '비슷한 사례'는 이전 artifact 참조일 수 있다. "
    "SQL 조회/문서 검색 필요 여부, worker task 분해는 SupervisorPlanner가 담당하므로 너는 판단하지 않는다.\n"
    "mode는 CURRENT_ONLY, USE_ACTIVE, PATCH_ACTIVE, SELECT_HISTORY, REFER_ACTIVE_RESULT 중 하나다. "
    "CURRENT_ONLY는 현재 사용자가 직접 말한 값만 쓴다. 이전 feature 자동 보완은 금지다. "
    "USE_ACTIVE는 '방금/아까/같은 조건/이전 입력값 기준'이라고 명시한 경우 active context 전체를 쓴다. "
    "PATCH_ACTIVE는 특정 값만 바꾸라고 명시한 경우 active context 하나에 현재 변경값만 덮어쓴다. "
    "SELECT_HISTORY는 recent_contexts 중 특정 과거 조건 하나를 지칭한 경우만 쓴다. 여러 context를 섞지 않는다. "
    "REFER_ACTIVE_RESULT는 재진단이 아니라 방금 결과/고장 유형/근거/이력만 참조하는 경우다.\n"
    "반드시 JSON만 출력하라: "
    "{\"is_followup\": true/false, \"uses_previous_prediction\": true/false, "
    "\"uses_previous_evidence\": true/false, \"uses_previous_sql\": true/false, "
    "\"inferred_time_range\": null 또는 객체, \"referenced_artifacts\": [\"prediction|sql|evidence\"], "
    "\"mode\": \"CURRENT_ONLY|USE_ACTIVE|PATCH_ACTIVE|SELECT_HISTORY|REFER_ACTIVE_RESULT\", "
    "\"base_context_id\": null 또는 문자열, \"patch_values\": 객체, \"reason\": \"짧은 이유\"}"
)

_ALLOWED_REFS = {"prediction", "sql", "evidence"}


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


def _has_prior_context(selected: dict) -> bool:
    """이번 턴 결정이 CURRENT_ONLY 외의 값이 될 수 있는, '재사용 가능한' 선행 맥락이 있는지.
    recent_turns(채팅 원문)는 제외한다: 저장된 DiagnosisContext나 이전 artifact 요약이 없으면
    어떤 mode도 base가 없어 CURRENT_ONLY로 강등되고 carryover도 참조 대상이 없으므로,
    채팅 턴만 있는 경우는 LLM 없이 short-circuit해도 결과가 동일하다(설계상 feature는 chat에서 자동 병합 금지)."""
    return bool(
        selected.get("active_context")
        or selected.get("recent_contexts")
        or selected.get("previous_prediction_summary")
        or selected.get("previous_evidence_summary")
        or selected.get("previous_sql_summary")
    )


def _current_only(current_values: dict, *, llm_skipped: bool, reason: str,
                  warnings: Optional[list] = None) -> ContextDecision:
    return ContextDecision(
        mode="CURRENT_ONLY",
        current_values=dict(current_values),
        resolved_features=dict(current_values),
        changed_features=list(current_values.keys()),
        reused_features=[],
        warnings=list(warnings or []),
        reason=reason,
        llm_skipped=llm_skipped,
    )


def _llm_decision_payload(user_message: str, selected: dict) -> dict:
    recent_summary = _summarize_recent_turns(selected.get("recent_turns") or [], user_all=True)
    return {
        "current_user_message": user_message,
        "current_values_extracted_from_this_turn": selected.get("current_values") or {},
        "active_context": _context_brief(selected.get("active_context")),
        "recent_contexts": [_context_brief(c) for c in (selected.get("recent_contexts") or [])],
        "recent_turns_summary": recent_summary,
        "previous_prediction_summary_available": bool(selected.get("previous_prediction_summary")),
        "previous_sql_summary_available": bool(selected.get("previous_sql_summary")),
        "previous_evidence_summary_available": bool(selected.get("previous_evidence_summary")),
    }


def _finalize_decision(data: dict, selected: dict, user_message: str) -> ContextDecision:
    """LLM 출력(dict)을 코드가 검증해 최종 ContextDecision으로 만든다.
    mode 강등/patch 화이트리스트/feature 계산은 결정적으로 수행한다."""
    current_values = dict(selected.get("current_values") or {})
    contexts = _contexts_by_id(selected)
    active = selected.get("active_context")
    warnings: list[str] = []

    # --- carryover 정규화 ---
    is_followup = bool(data.get("is_followup"))
    refs = data.get("referenced_artifacts") or []
    if isinstance(refs, str):
        refs = [refs]
    refs = [x for x in refs if x in _ALLOWED_REFS]
    uses_pred = bool(data.get("uses_previous_prediction"))
    uses_sql = bool(data.get("uses_previous_sql"))
    uses_ev = bool(data.get("uses_previous_evidence"))
    if not is_followup:
        uses_pred = uses_sql = uses_ev = False
        refs = []
    elif not refs:
        if uses_pred:
            refs.append("prediction")
        if uses_sql:
            refs.append("sql")
        if uses_ev:
            refs.append("evidence")

    # --- resolution 검증 (기존 resolve_context 규칙 이식) ---
    mode = data.get("mode") if data.get("mode") in ContextMode.__args__ else "CURRENT_ONLY"
    base_context_id = data.get("base_context_id")
    base: Optional[DiagnosisContext] = None

    if mode in {"USE_ACTIVE", "PATCH_ACTIVE", "REFER_ACTIVE_RESULT"}:
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

    patch_values = _filter_patch_values(data.get("patch_values") or {}, current_values)
    if mode in {"PATCH_ACTIVE", "SELECT_HISTORY"} and current_values and not patch_values:
        patch_values = dict(current_values)

    if mode == "CURRENT_ONLY":
        resolved = dict(current_values)
        changed = list(current_values.keys())
        reused: list[str] = []
        base_context_id = None
    elif mode == "REFER_ACTIVE_RESULT":
        resolved, changed, reused = {}, [], []
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
        resolved = dict(current_values)
        changed = list(current_values.keys())
        reused = []
        base_context_id = None
        mode = "CURRENT_ONLY"

    return ContextDecision(
        is_followup=is_followup,
        referenced_artifacts=refs,
        uses_previous_prediction=uses_pred,
        uses_previous_evidence=uses_ev,
        uses_previous_sql=uses_sql,
        inferred_time_range=data.get("inferred_time_range") if isinstance(data.get("inferred_time_range"), dict) else None,
        mode=mode,
        base_context_id=base_context_id,
        patch_values=patch_values,
        reason=str(data.get("reason") or "LLM context decision"),
        current_values=current_values,
        resolved_features=resolved,
        changed_features=changed,
        reused_features=reused,
        warnings=warnings,
        llm_skipped=False,
    )


def decide_context(user_message: str, selected: dict) -> ContextDecision:
    """단일 ContextDecision 생성. 이전 context가 전혀 없으면 LLM을 건너뛰고(CURRENT_ONLY) 비용을 0으로 만든다."""
    current_values = dict(selected.get("current_values") or {})
    if not _has_prior_context(selected):
        return _current_only(current_values, llm_skipped=True,
                             reason="trivial turn; no prior context")
    try:
        raw = call_llm(CONTEXT_DECISION_SYS, json.dumps(_llm_decision_payload(user_message, selected), ensure_ascii=False), tier="default")
        data = _json_object(raw)
    except Exception as e:
        return _current_only(current_values, llm_skipped=False,
                             reason="context decision failed; current values only",
                             warnings=[f"context_decision_llm_fallback: {type(e).__name__}"])
    return _finalize_decision(data, selected, user_message)


print("context_engine(단일 ContextDecision 1콜 + short-circuit) 정의 완료")
