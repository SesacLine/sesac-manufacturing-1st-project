from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.context.policy import STALE_THRESHOLD_SECONDS
from manufacturing_agent.contracts.context import ContextResolution, DiagnosisContext, MachineValue

# ---------- context/context_normalizer.py ----------
def _machine_value_from_context(name: str, val: Any, *, is_current: bool, source: str,
                                is_stale: bool = False) -> MachineValue:
    return MachineValue(name=name, value=val, source=source, is_current=is_current, is_stale=is_stale)


def _base_context_for(selected: dict, base_context_id: Optional[str]) -> Optional[DiagnosisContext]:
    if not base_context_id:
        return None
    active = selected.get("active_context")
    if active and getattr(active, "id", None) == base_context_id:
        return active
    for ctx in selected.get("recent_contexts") or []:
        if getattr(ctx, "id", None) == base_context_id:
            return ctx
    return None


def normalize_context(selected: dict) -> tuple[dict[str, MachineValue], list[str]]:
    """ContextResolution 결과를 PredictionAgent 입력용 MachineValue로 변환한다.

    이전 feature를 자동 보완하지 않는다. resolved_features는 CURRENT_ONLY, USE_ACTIVE,
    PATCH_ACTIVE, SELECT_HISTORY 중 하나의 mode에서 만들어진 단일 context 결과다.
    재사용한 base 진단 context가 STALE_THRESHOLD_SECONDS보다 오래됐으면 reused feature를 stale로 표시한다.
    """
    resolution = selected.get("context_resolution") or ContextResolution(
        mode="CURRENT_ONLY",
        current_values=selected.get("current_values") or {},
        resolved_features=selected.get("current_values") or {},
        changed_features=list((selected.get("current_values") or {}).keys()),
        reason="context_resolution missing; current values only",
    )
    warnings: list[str] = list(resolution.warnings or [])
    merged: dict[str, MachineValue] = {}
    current_keys = set((resolution.current_values or {}).keys())
    changed = set(resolution.changed_features or [])
    reused = set(resolution.reused_features or [])

    # 재사용 base context의 신선도 판정: 오래되면 reused feature를 stale로 표시한다.
    base = _base_context_for(selected, resolution.base_context_id)
    base_is_stale = False
    if base is not None:
        age = base.age_seconds(_dt.datetime.now().isoformat(timespec="seconds"))
        base_is_stale = age is not None and age > STALE_THRESHOLD_SECONDS
        if base_is_stale and reused:
            warnings.append(
                f"재사용한 진단 context가 오래되어(약 {int(age // 60)}분 경과) 일부 입력값이 현재 상태와 다를 수 있습니다.")

    for name, val in (resolution.resolved_features or {}).items():
        is_current = name in current_keys and (resolution.mode == "CURRENT_ONLY" or name in changed)
        if is_current:
            source = "current"
            is_stale = False
        elif name in reused:
            source = "active_context" if resolution.mode in {"USE_ACTIVE", "PATCH_ACTIVE"} else "history_context"
            is_stale = base_is_stale
        else:
            source = "context"
            is_stale = False
        merged[name] = _machine_value_from_context(name, val, is_current=is_current, source=source, is_stale=is_stale)

    if selected.get("injection_in_current"):
        warnings.append("현재 입력에서 prompt injection 의심 패턴 감지 → 무력화")
    return merged, warnings
print("context_normalizer(stale 구현) 정의 완료")
