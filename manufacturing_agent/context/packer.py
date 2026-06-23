from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.context.policy import PER_TURN_CHAR_CAP, RECENT_SUMMARY_CHAR_BUDGET, STANDARD_FEATURES, detect_injection
from manufacturing_agent.contracts.context import AgentContextPacket, ContextCarryoverDecision, ContextPacket, MachineValue

# ---------- context/context_packer.py ----------
# carryover/resolution 판단(LLM)은 context/engine.py로 이동했다. 이 파일은 대화 요약/패킹/공용 컨텍스트 요약만 담당한다.
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
# - 사용자 질문은 의도 추적에 중요하므로 윈도우 안에서 전부 유지한다(단, 토큰 버짓 캡 적용).
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

def _apply_char_budget(parts: list[str], budget: int) -> list[str]:
    """말미(최신) 우선으로 char 버짓 안에 들어오는 항목만 남긴다. 오래된 항목부터 버린다."""
    if budget <= 0:
        return parts
    kept: list[str] = []
    total = 0
    for piece in reversed(parts):
        add = len(piece) + 3  # separator 여유
        if total + add > budget and kept:
            break
        kept.append(piece)
        total += add
    kept.reverse()
    return kept

def _summarize_recent_turns(turns: list[dict], limit: int = 6, chars: Optional[int] = None,
                            *, user_all: bool = False, assistant_limit: Optional[int] = ASSISTANT_TURN_LIMIT) -> str:
    """최근 대화를 'role:content' 한 줄 형태로 이어붙인다(개행은 공백으로 평탄화).
    user_all=True이면 사용자(user) 턴은 모두 유지하고, AI(assistant) 답변만 최근 assistant_limit개로 제한한다(순서 보존).
    토큰 폭주 방지를 위해 각 턴 본문은 PER_TURN_CHAR_CAP로, 전체 길이는 RECENT_SUMMARY_CHAR_BUDGET로 캡한다(최신 우선)."""
    seq = list(turns or [])
    if user_all:
        assistant_idx = [i for i, t in enumerate(seq) if t.get("role") == "assistant"]
        keep = set(assistant_idx) if assistant_limit is None else set(assistant_idx[-assistant_limit:])
        selected = [t for i, t in enumerate(seq) if t.get("role") != "assistant" or i in keep]
    else:
        selected = seq[-limit:]
    per_turn_cap = chars if chars is not None else PER_TURN_CHAR_CAP
    def _body(content: Any) -> str:
        text = str(content).replace(chr(10), " ")
        return text if per_turn_cap is None else text[:per_turn_cap]
    parts = [f"{t['role']}:{_body(t['content'])}" for t in selected]
    parts = _apply_char_budget(parts, RECENT_SUMMARY_CHAR_BUDGET)
    return " | ".join(parts)


def build_context_summary(packet: Optional[ContextPacket], *, for_sql: bool = False,
                          prediction_result: Optional[Any] = None) -> str:
    """planner payload와 evidence_agent SQL context가 공유하는 단일 컨텍스트 요약 빌더.
    두 LLM 채널(LangGraph call_llm / PydanticAI)이 동일한 멀티턴 맥락을 보도록 일원화한다.
    for_sql=True면 failure_history 가드 문장과 현재 prediction failure_types를 덧붙인다.
    (reference_date·time_range sanitize·task params 같은 SQL 실행 전용 항목은 호출측이 추가한다.)"""
    if not packet:
        return ""
    carry = packet.context_carryover
    blocks: list[str] = []
    if for_sql:
        blocks.append("현재 SQL DB는 failure_history 단일 테이블이다. 설비/자산 식별자 조건은 사용하지 않는다.")
    if packet.recent_turns_summary:
        blocks.append(f"참고용 최근 대화(thread context): {packet.recent_turns_summary}")

    def _label(used: Optional[bool], name: str) -> str:
        return f"현재 질문이 참조한 이전 {name} artifact" if used else f"참고용 이전 {name} artifact"

    if packet.previous_sql_summary:
        blocks.append(f"{_label(carry and carry.uses_previous_sql, 'SQL 이력')}: {packet.previous_sql_summary}")
    if packet.previous_evidence_summary:
        blocks.append(f"{_label(carry and carry.uses_previous_evidence, '문서 근거')}: {packet.previous_evidence_summary}")
    if packet.previous_prediction_summary:
        blocks.append(f"{_label(carry and carry.uses_previous_prediction, '위험 진단')}: {packet.previous_prediction_summary}")
    if for_sql and prediction_result is not None:
        blocks.append(
            f"현재 prediction failure_types: {getattr(prediction_result, 'failure_types', [])}; "
            f"cause_features: {getattr(prediction_result, 'cause_features', [])}")
    if not for_sql and packet.user_constraints:
        blocks.append(f"현재 제약/범위: {json.dumps(packet.user_constraints, ensure_ascii=False)}")
    return "\n".join(blocks)


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
        context_resolution=resolution,
        selected_machine_values=merged,
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
print("context_packer(요약 캡 + build_context_summary 일원화) 정의 완료")
