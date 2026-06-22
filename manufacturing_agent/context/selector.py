from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.context.packer import RECENT_TURN_WINDOW
from manufacturing_agent.context.policy import detect_injection, extract_machine_values
from manufacturing_agent.memory.store import ConversationStore

# ---------- context/context_selector.py ----------
def select_context(user_message: str, user_id: str, store: ConversationStore,
                   structured: Optional[dict] = None, thread_id: Optional[str] = None) -> dict:
    """현재 입력값과 재사용 가능한 진단 context 후보만 선택한다.

    feature별 최신값을 자동으로 가져오지 않는다. 이전 feature 재사용 여부는
    ContextResolution 단계에서 하나의 base DiagnosisContext를 선택한 뒤 결정한다.
    """
    nl_vals = extract_machine_values(user_message)
    structured = structured or {}
    current_vals = {**nl_vals, **structured}
    # 사용자 질문을 윈도우 내 전부 보존하기 위해 넉넉히 가져온다(요약 단계에서 user_all 정책 적용).
    recent = store.recent_turns(user_id, limit=RECENT_TURN_WINDOW, thread_id=thread_id)
    clean_recent = [t for t in recent if not detect_injection(t["content"])]
    active_context = store.get_active_context(user_id, thread_id) if thread_id else None
    recent_contexts = store.get_recent_contexts(user_id, thread_id, limit=5) if thread_id else []
    return {
        "current_values": current_vals,
        "active_context": active_context,
        "recent_contexts": recent_contexts,
        "recent_turns": clean_recent,
        "previous_prediction_summary": store.latest_summary(user_id, "prediction", thread_id=thread_id),
        "previous_evidence_summary": store.latest_summary(user_id, "evidence", thread_id=thread_id),
        "previous_sql_summary": store.latest_summary(user_id, "sql", thread_id=thread_id),
        "injection_in_current": detect_injection(user_message),
    }
print("context_selector 정의 완료")
