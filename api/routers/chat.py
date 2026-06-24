"""Chat and resume endpoints backed by the LangGraph runtime."""
from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.deps import require_user_thread
from api.schemas import ChatRequest, ChatResponse, ResumeRequest
from manufacturing_agent.runtime import resume_turn, run_turn, make_initial_state, make_runnable_config, app

router = APIRouter(tags=["chat"])

# LangGraph 노드 → 사용자에게 보여줄 진행 단계 라벨
_STEP_LABELS = {
    "intake_gate": "입력 안전 점검",
    "context_manager": "맥락 구성",
    "supervisor_planner": "실행 계획 수립",
    "orchestrator_dispatcher": "작업 분배",
    "prediction_agent": "위험 진단 실행",
    "prediction_gate": "진단 검증",
    "evidence_agent": "문서 근거 검색",
    "evidence_gate": "근거 검증",
    "sql_agent": "고장 이력 조회",
    "sql_gate": "이력 검증",
    "supervisor_replanner": "계획 보정",
    "final_answer": "답변 작성",
    "output_safety_gate": "출력 안전 점검",
    "memory_writer": "대화 저장",
}
_TASK_KO = {"prediction": "위험 진단", "sql": "고장 이력", "evidence": "문서 근거", "final_answer": "답변"}


def _step_detail(node: str, delta: dict) -> str:
    """노드 업데이트에서 보여줄 짧은 설명을 뽑는다."""
    try:
        if node == "supervisor_planner":
            plan = delta.get("execution_plan")
            tasks = [t.task_type for t in getattr(plan, "tasks", []) if t.task_type != "final_answer"]
            ko = ", ".join(_TASK_KO.get(t, t) for t in tasks)
            return f"필요 작업: {ko}" if ko else ""
        if node in {"prediction_gate", "evidence_gate", "sql_gate", "intake_gate", "output_safety_gate"}:
            reports = delta.get("gate_reports") or []
            if reports:
                return str(reports[-1].get("status", ""))
    except Exception:
        pass
    return ""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_gen(user_id: str, thread_id: str, message: str, input_features, debug: bool):
    cfg = make_runnable_config(user_id, thread_id, uuid4().hex, recursion_limit=50)
    state_in = make_initial_state(message, user_id, thread_id, uuid4().hex, input_features)
    yield _sse({"type": "start"})
    try:
        for update in app.stream(state_in, config=cfg, stream_mode="updates"):
            for node, delta in (update or {}).items():
                if node not in _STEP_LABELS:
                    continue
                yield _sse({"type": "step", "node": node,
                            "label": _STEP_LABELS.get(node, node),
                            "detail": _step_detail(node, delta or {})})
        final_state = app.get_state(cfg).values or {}
        resp = _build_response(user_id, thread_id, final_state, debug)
        yield _sse({"type": "done", **resp.model_dump()})
    except Exception as exc:  # noqa: BLE001
        if "insufficient_quota" in str(exc):
            yield _sse({"type": "error", "code": "llm_quota_exhausted",
                        "message": "LLM 사용량(쿼터)이 소진되어 답변을 생성할 수 없습니다."})
        else:
            yield _sse({"type": "error", "code": "internal", "message": str(exc)[:300]})


def _sql_payload(sql_result) -> "dict | None":
    """SQL 에이전트가 조회한 failure_history 행을 프론트 카드용으로 평탄화한다.
    SQL을 쓰지 않은 턴이면 None을 반환한다(프론트는 None이면 DB 카드 섹션을 안 그린다)."""
    if sql_result is None:
        return None
    rows = []
    for r in (getattr(sql_result, "results", []) or []):
        rows.extend(getattr(r, "rows", None) or [])
    if not rows:
        rows = list(getattr(sql_result, "rows", None) or [])
    return {
        "status": getattr(sql_result, "status", None),
        "row_count": len(rows),
        "rows": rows[:30],
        "summary": (getattr(sql_result, "summary", "") or "")[:500],
    }


def _evidence_payload(ev) -> "dict | None":
    """evidence(RAG) 에이전트 레벨 메타. 문서별 근거(title/source/snippet)는 top-level citations에 있다."""
    if ev is None:
        return None
    return {
        "status": getattr(ev, "status", None),
        "summary": (getattr(ev, "evidence_summary", "") or "")[:1000],
        "doc_count": len(getattr(ev, "documents", None) or []),
        "citation_count": len(getattr(ev, "citations", None) or []),
    }


def _build_response(user_id: str, thread_id: str, result: dict, debug: bool) -> ChatResponse:
    """Map a runtime result dict to a ChatResponse, defensively handling None."""
    result = result or {}
    final = result.get("final_answer")
    decision = result.get("input_decision")

    blocked = bool(getattr(decision, "blocked", False)) if decision is not None else False

    if final is not None and getattr(final, "answer", None):
        answer = final.answer
    elif decision is not None and getattr(decision, "block_message", None):
        answer = decision.block_message
    else:
        answer = ""

    citations = list(getattr(final, "citations", []) or []) if final is not None else []
    data_refs = list(getattr(final, "data_refs", []) or []) if final is not None else []
    warnings = list(getattr(final, "warnings", []) or []) if final is not None else []
    missing_inputs = list(getattr(final, "missing_inputs", []) or []) if final is not None else []

    trace = None
    if debug:
        gate_reports = result.get("gate_reports") or []
        gates = [[r.get("gate_name"), r.get("status")] for r in gate_reports]
        plan = result.get("execution_plan")
        if plan is not None and getattr(plan, "tasks", None):
            tasks = sorted({t.task_type for t in plan.tasks})
        else:
            tasks = []
        trace = {"gates": gates, "tasks": tasks}

    return ChatResponse(
        user_id=user_id,
        thread_id=thread_id,
        answer=answer,
        citations=citations,
        data_refs=data_refs,
        warnings=warnings,
        missing_inputs=missing_inputs,
        blocked=blocked,
        sql=_sql_payload(result.get("sql_result")),
        evidence=_evidence_payload(result.get("evidence_bundle")),
        trace=trace,
    )


def _map_runtime_error(exc: Exception) -> HTTPException:
    if "insufficient_quota" in str(exc):
        return HTTPException(status_code=503, detail="llm_quota_exhausted")
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, debug: bool = Query(False)):
    require_user_thread(req.user_id, req.thread_id)
    try:
        result = run_turn(
            req.message,
            req.user_id,
            req.thread_id,
            request_id=uuid4().hex,
            input_features=req.input_features,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _map_runtime_error(exc)
    return _build_response(req.user_id, req.thread_id, result, debug)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, debug: bool = Query(False)):
    """진행 단계를 SSE로 흘려보낸다. 이벤트: start → step(여러 번) → done | error.
    프론트는 fetch()+ReadableStream으로 `data: {...}` 라인을 파싱한다(EventSource는 GET만 가능하므로 미사용)."""
    require_user_thread(req.user_id, req.thread_id)
    return StreamingResponse(
        _stream_gen(req.user_id, req.thread_id, req.message, req.input_features, debug),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/resume", response_model=ChatResponse)
def chat_resume(req: ResumeRequest, debug: bool = Query(False)):
    require_user_thread(req.user_id, req.thread_id)
    try:
        result = resume_turn(req.user_id, req.thread_id, request_id="resume")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _map_runtime_error(exc)
    return _build_response(req.user_id, req.thread_id, result, debug)
