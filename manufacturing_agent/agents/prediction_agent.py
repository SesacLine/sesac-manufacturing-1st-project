from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import PredictionResult
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.services.prediction_service import run_prediction

# ---------- agents/prediction_agent/agent.py ----------
def build_prediction_summary(result: PredictionResult) -> str:
    if result.status == "NEEDS_INPUT":
        return f"진단에 필요한 입력이 부족합니다: {result.missing_features}"
    if not result.risk_flags:
        return "현재 입력 기준으로 뚜렷한 고위험 신호는 확인되지 않았습니다."
    top = result.risk_flags[:3]
    items = [f"{r.get('failure_type')}={r.get('level')} (측정값: {r.get('detail')})" for r in top]
    causes = ", ".join(result.cause_features) if result.cause_features else "확인된 주요 변수 없음"
    return (
        f"현재 입력 기준 주요 위험은 {', '.join(items)}입니다. "
        f"주요 영향 변수는 {causes}입니다. "
        "이 결과는 규칙 기반 부분 위험 진단이며 실제 정비 판단은 현장 점검과 함께 확인해야 합니다."
    )

def prediction_agent(state: ManufacturingState) -> dict:
    """Rule-based diagnostic / partial risk assessment. 이름은 기존 호환을 위해 prediction_agent로 유지한다."""
    ctx = state["agent_contexts"]["prediction_agent"]
    feedback = (state.get("agent_feedback") or {}).get("prediction_agent")
    packet = state.get("context_packet")
    resolution = packet.context_resolution if packet else None
    sel = ctx.selected_context
    feats = dict(sel.get("features", {}))
    # 이번 턴에 명시적으로 들어온 구조화 센서 입력(데이터 입력란)은 context 해석보다 우선해 항상 반영한다.
    _structured = state.get("input_features")
    if _structured is not None:
        _sd = _structured.to_features() if hasattr(_structured, "to_features") else (
            _structured.model_dump(exclude_none=True) if hasattr(_structured, "model_dump") else dict(_structured))
        feats.update({k: v for k, v in _sd.items() if v is not None})
    out = run_prediction(feats)
    # 컨텍스트 메타는 ContextResolution을 단일 출처로 사용한다(둘 다 같은 ContextDecision에서 파생되므로 동일).
    # packet이 없을 때(context_manager 미경유)에만 agent_context로 폴백한다.
    if resolution is not None:
        context_mode = resolution.mode
        base_context_id = resolution.base_context_id
        changed_features = list(resolution.changed_features)
        reused_features = list(resolution.reused_features)
    else:
        context_mode = sel.get("context_mode", "CURRENT_ONLY")
        base_context_id = sel.get("base_context_id")
        changed_features = list(sel.get("changed_features") or [])
        reused_features = list(sel.get("reused_features") or [])
    # stale은 normalizer가 계산해 agent_context["stale"]에 담아둔 결과를 그대로 보고한다(기존 하드코딩 [] 제거).
    used_stale = list(sel.get("stale") or [])

    if out["full"]:
        status = "OK"
    elif out["risks"]:
        status = "PARTIAL"
    elif out["missing"]:
        status = "NEEDS_INPUT"
    else:
        status = "SKIPPED"

    risk_flags = [
        {
            "failure_type": r.failure_type,
            "level": r.level,
            "score": r.score,
            "detail": r.detail,
            "rule": r.rule,
            "formula": r.formula,
            "contributing_features": r.contributing_features,
            "recommended_checks": r.recommended_checks,
        }
        for r in out["risks"]
    ]
    failure_types = [r.failure_type for r in out["risks"]]
    cause_features: list[str] = []
    for r in out["risks"]:
        for f in r.contributing_features:
            if f in feats and f not in cause_features:
                cause_features.append(f)

    limitations = list(out["limitations"])
    if context_mode == "CURRENT_ONLY":
        limitations.append("이번 질문에서 사용자가 직접 제공한 값만 기준으로 판단했습니다.")
    elif context_mode == "PATCH_ACTIVE":
        changed = ", ".join(changed_features) or "현재 변경값"
        limitations.append(f"이전 진단 context를 기준으로 {changed}만 변경해 판단했습니다.")
    elif context_mode == "USE_ACTIVE":
        limitations.append("사용자가 명시적으로 참조한 이전 진단 context를 기준으로 판단했습니다.")
    elif context_mode == "SELECT_HISTORY":
        limitations.append("사용자가 지칭한 과거 진단 context 하나를 선택해 판단했습니다.")
    if feedback:
        limitations.append(f"보완 실행 피드백: {feedback}")

    result = PredictionResult(
        status=status,
        missing_features=out["missing"],
        risk_flags=risk_flags,
        failure_types=failure_types,
        cause_features=cause_features,
        safety_hints=out["safety_hints"],
        used_stale_features=used_stale,
        confidence=out["confidence"],
        limitations=limitations,
        summary="",
        context_mode=context_mode,
        base_context_id=base_context_id,
        changed_features=changed_features,
        reused_features=reused_features,
    )
    result.summary = build_prediction_summary(result)
    return {"prediction_result": result}
print("prediction_agent(rule-based diagnostic) 정의 완료")
