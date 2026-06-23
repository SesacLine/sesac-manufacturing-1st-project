from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.agents.evidence_agent import SQL_TABLE_RE
from manufacturing_agent.context.policy import detect_injection
from manufacturing_agent.contracts.context import ContextPacket, DiagnosisContext, EvidenceArtifact, PredictionResult, SQLHistoryArtifact
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.memory.store import conversation_store, run_store

# ---------- nodes/memory_writer_node.py ----------
def _compact_sql_artifact_for_memory(sql: Optional[SQLHistoryArtifact]) -> str:
    if not sql:
        return ""
    lines = [f"status={sql.status}"]
    results = getattr(sql, "results", []) or []
    if results:
        for result in results[:4]:
            tables = [m.group(1).split(".")[-1] for m in SQL_TABLE_RE.finditer(result.sql or "")]
            sample = (result.rows or [])[:2]
            lines.append(
                f"query_type={result.query_type}; tables={tables}; rows={len(result.rows)}; summary={result.summary}; sample_rows={json.dumps(sample, ensure_ascii=False)}"
            )
    elif sql.sql:
        tables = [m.group(1).split(".")[-1] for m in SQL_TABLE_RE.finditer(sql.sql)]
        lines.append(f"query_type={sql.query_type}; tables={tables}; rows={len(sql.rows)}; summary={sql.summary}")
    elif sql.summary:
        lines.append(sql.summary)
    if sql.limitations:
        lines.append(f"limitations={sql.limitations[:3]}")
    # 길이 캡 없이 원문 그대로 보존한다(sample_rows는 query당 2개·query 4개로 이미 구조적으로 제한됨).
    return " | ".join(lines)

def _compact_evidence_artifact_for_memory(ev: Optional[EvidenceArtifact]) -> str:
    if not ev:
        return ""
    sources = [c.get("source_id") for c in (ev.citations or [])[:5]]
    queries = (ev.queries or [])[:3]
    # 길이 캡 없이 원문 그대로 보존한다(evidence_summary는 LLM 생성 요약으로 토큰 상한이 이미 적용됨).
    return (
        f"status={ev.status}; profile={ev.retrieval_profile}; queries={queries}; "
        f"sources={sources}; summary={ev.evidence_summary}; limitations={ev.limitations[:3]}"
    )

def _should_save_diagnosis_context(state: ManufacturingState, pred: Optional[PredictionResult], packet: Optional[ContextPacket]) -> bool:
    if not pred or pred.status not in {"OK", "PARTIAL"}:
        return False
    if not packet or not packet.selected_machine_values:
        return False
    dec = state.get("input_decision")
    if dec and getattr(dec, "blocked", False):
        return False
    if detect_injection(state.get("user_message", "")):
        return False
    return True


def memory_writer_node(state: ManufacturingState) -> dict:
    user_id = state.get("user_id", "?")
    thread_id = state.get("thread_id", "?")
    msg = state.get("user_message", "")
    fa = state.get("final_answer")
    packet = state.get("context_packet")

    conversation_store.add_turn(user_id, "user", msg, thread_id=thread_id)
    if fa:
        conversation_store.add_turn(user_id, "assistant", fa.answer, thread_id=thread_id)

    # turn_id(=request_id)를 함께 저장해 후속 턴이 특정 과거 artifact를 지목 조회할 수 있게 한다.
    turn_id = state.get("request_id") or None
    pred = state.get("prediction_result")
    if pred and pred.status in {"OK", "PARTIAL"} and pred.summary:
        conversation_store.add_summary(user_id, "prediction", pred.summary, thread_id=thread_id, turn_id=turn_id)
    if _should_save_diagnosis_context(state, pred, packet):
        import uuid
        features = {k: v.value for k, v in packet.selected_machine_values.items() if v.value is not None}
        diag = DiagnosisContext(
            id=f"diag-{uuid.uuid4().hex}",
            turn_id=state.get("request_id") or "unknown-turn",
            user_id=user_id,
            thread_id=thread_id,
            features=features,
            failure_types=list(pred.failure_types or []),
            prediction_summary=pred.summary or "",
            created_at=conversation_store._now(),
            is_safe_to_reuse=True,
        )
        conversation_store.save_diagnosis_context(user_id, thread_id, diag)
    ev = state.get("evidence_bundle")
    if ev:
        conversation_store.add_summary(user_id, "evidence", _compact_evidence_artifact_for_memory(ev), thread_id=thread_id, turn_id=turn_id)
    sql = state.get("sql_result")
    if sql:
        conversation_store.add_summary(user_id, "sql", _compact_sql_artifact_for_memory(sql), thread_id=thread_id, turn_id=turn_id)

    # 실행 이력 저장
    run_store.save(state.get("request_id", "?"), user_id, thread_id,
                   {"gate_reports": state.get("gate_reports", []),
                    "retry_counts": state.get("retry_counts", {})})
    return {"messages": [AIMessage(content=fa.answer)]} if fa else {}
print("memory_writer_node 정의 완료")
