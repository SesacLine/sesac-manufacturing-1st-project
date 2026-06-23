from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.context.engine import decide_context
from manufacturing_agent.context.normalizer import normalize_context
from manufacturing_agent.context.packer import RECENT_TURN_WINDOW, _dedup_turns, _messages_to_recent_turns, pack_contexts
from manufacturing_agent.context.selector import select_context
from manufacturing_agent.contracts.context import EvidenceArtifact, PredictionResult, SQLHistoryArtifact
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.memory.store import conversation_store
from manufacturing_agent.nodes.memory_writer_node import _compact_evidence_artifact_for_memory, _compact_sql_artifact_for_memory

def _summary_from_artifact(kind: str, artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    if kind == "prediction":
        return getattr(artifact, "summary", None)
    if kind == "evidence":
        return _compact_evidence_artifact_for_memory(artifact)
    if kind == "sql":
        return _compact_sql_artifact_for_memory(artifact)
    return None


def context_manager(state: ManufacturingState, config: RunnableConfig = None) -> dict:
    msg = state["user_message"]
    cfg = (config or {}).get("configurable", {})
    user_id = cfg.get("user_id") or state["user_id"]
    thread_id = cfg.get("thread_id") or state.get("thread_id")
    structured = state.get("input_features") or {}
    if hasattr(structured, "model_dump"):
        structured = structured.model_dump(exclude_none=True)

    prev_pred = state.get("prediction_result")
    prev_ev = state.get("evidence_bundle")
    prev_sql = state.get("sql_result")
    # 체크포인트 복원 시 dict로 올 수 있어 모델로 보정한다.
    if isinstance(prev_pred, dict): prev_pred = PredictionResult.model_validate(prev_pred)
    if isinstance(prev_ev, dict): prev_ev = EvidenceArtifact.model_validate(prev_ev)
    if isinstance(prev_sql, dict): prev_sql = SQLHistoryArtifact.model_validate(prev_sql)

    selected = select_context(msg, user_id, conversation_store, structured, thread_id=thread_id)
    selected["previous_prediction_summary"] = _summary_from_artifact("prediction", prev_pred) or selected.get("previous_prediction_summary")
    selected["previous_evidence_summary"] = _summary_from_artifact("evidence", prev_ev) or selected.get("previous_evidence_summary")
    selected["previous_sql_summary"] = _summary_from_artifact("sql", prev_sql) or selected.get("previous_sql_summary")

    checkpoint_turns = _messages_to_recent_turns(state.get("messages", []), limit=RECENT_TURN_WINDOW)
    if checkpoint_turns:
        selected["recent_turns"] = (selected.get("recent_turns") or []) + checkpoint_turns
    # store 턴 + checkpoint 턴 중복 제거 후 윈도우 적용(같은 대화 이중 노출 방지).
    selected["recent_turns"] = _dedup_turns(selected.get("recent_turns") or [])[-RECENT_TURN_WINDOW:]
    # 단일 ContextDecision 1콜(+short-circuit)로 carryover와 resolution을 함께 판단한다.
    decision = decide_context(msg, selected)
    selected["context_resolution"] = decision.to_resolution()
    selected["context_carryover"] = decision.to_carryover()
    merged, warnings = normalize_context(selected)
    packet, agent_ctx = pack_contexts(msg, merged, selected, warnings)

    return {
        "context_packet": packet,
        "context_resolution": selected["context_resolution"],
        "agent_contexts": agent_ctx,
        # 새 턴 runtime artifact는 packet에 이전 요약을 옮긴 뒤 초기화한다.
        "prediction_result": None,
        "evidence_bundle": None,
        "sql_result": None,
        "final_answer": None,
        "execution_plan": None,
        "supervisor_planner_decision": None,
        "supervisor_replanner_decision": None,
        "sql_intent_decision": None,
        "orchestrator_decision": None,
        "active_task_id": None,
        "route": None,
        "intent": None,
        "agent_feedback": {},
        "retry_counts": {},
        "consumed_replan_report_index": None,
    }
print("context_manager 정의 완료")
