from __future__ import annotations
from types import SimpleNamespace
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import FinalAnswer
from manufacturing_agent.contracts.state import ManufacturingState

# ---------- nodes/final_answer_llm_node.py ----------
# 실험용 경량 최종 답변 노드(LLM-자유 버전).
#   기존 final_answer_node의 결정적 조립(종합 판단 배너 / 고장 근거 블록 / 체크리스트 /
#   숫자 환각 가드 / 안전 트레일러 / 용어 현지화 / 폴백 템플릿)을 전부 걷어내고,
#   facts sheet를 준 뒤 LLM(tier="final") 한 콜이 배너·섹션·citation·안전 문구까지 자유롭게 작성한다.
#   유일하게 남기는 결정성: citation 최소 검증(존재하지 않는 [C#] 제거 + 실제 인용된 문서만 [출처] 렌더).
#   facts 조립과 citation 렌더 헬퍼는 기존 노드 것을 재사용한다(입력 정리/출처 포맷은 그대로 결정적).
from manufacturing_agent.nodes.final_answer_node import (
    build_answer_context,
    _format_citations,
    _fallback_final_answer,
    _mark_final_task_pass,
)

FINAL_ANSWER_LLM_SYSTEM_PROMPT = """
너는 제조 설비 진단 AI Agent의 최종 답변 작성자다.
공정·설비 엔지니어가 바로 읽고 판단할 수 있는 하나의 자연스러운 한국어 답변을 작성한다.

원칙:
- 아래 facts sheet에 있는 정보만 사용한다. 없는 수치·근거·고장유형은 절대 지어내지 말고 "확인 필요"로 표현한다.
- 답변의 구조(맨 앞 한 줄 요약, 섹션 구성, 길이, 말투)는 질문 성격에 맞게 네가 자유롭게 정한다.
  진단/이력/근거가 풍부하면 충실히, 단순·일반 질문이면 짧고 대화체로 답한다. 형식 틀을 억지로 채우지 마라.
- 문서 citation이 제공되면 근거가 된 문장 끝에 [C1] 형식으로 표시한다. 제공되지 않은 citation 번호는 만들지 마라.
- raw 스키마 용어(tool_wear, drive_system, query_type, score 값 등) 대신 현장 한국어 용어로 풀어 쓴다.
- 정지/재가동/정비 같은 조치 판단이 걸리면, 최종 승인은 현장 안전 책임자·설비 담당자 몫임을 자연스럽게 덧붙인다.
- 내부 처리 과정, Agent 이름, 라우팅 경로, JSON, SQL row는 출력하지 않는다.

자기검토 과정은 출력하지 말고 사용자에게 보여줄 답변 본문만 작성하라.
""".strip()

FINAL_ANSWER_LLM_USER_PROMPT = """
사용자 질문:
{user_question}

답변 대상(내부 참고용 — 제목에 그대로 복사하지 말 것):
{equipment_id}

아래 facts sheet 안에서만 답하라. 없는 정보는 "확인 필요"로 표현하라.

[현재 위험 진단 요약]
{prediction_summary}

[최근 이력 — 검증된 수치(이 숫자만 사용)]
{history_summary}

[문서 근거 요약]
{evidence_summary}

[안전 판단 요약]
{safety_summary}

[정확 수치 근거(이 수치만 사용)]
{diagnosis_block}

[사용 가능한 Citation 목록]
{citations}
""".strip()


_DATA_REF_ROW_CAP = 50  # 클라이언트로 보내는 행 상한(과다 전송 방지)
_DATA_REF_LABELS = {"detail": "고장 이력 상세", "aggregate": "고장 유형별 집계"}


def _compact_sql(sql) -> str:
    return re.sub(r"\s+", " ", str(sql or "")).strip()


def _date_window(rows: list[dict]):
    dates = sorted(str(r.get("event_date") or "")[:10] for r in rows if r.get("event_date"))
    if not dates:
        return None
    return dates[0] if dates[0] == dates[-1] else f"{dates[0]} ~ {dates[-1]}"


def _build_data_refs(sql) -> list[dict]:
    """실행된 read-only SQL과 반환 행을 결정적으로 [D#] 데이터 출처로 만든다(LLM 비의존, 프론트 drill-down용).
    이미 실행된 쿼리의 스냅샷만 담는다 — 행은 상한으로 자른다."""
    if not sql or getattr(sql, "status", None) != "OK":
        return []
    results = list(getattr(sql, "results", []) or [])
    if not results and getattr(sql, "rows", None):  # 단일 결과 fallback
        results = [SimpleNamespace(query_type=sql.query_type or "detail",
                                   sql=sql.sql, rows=sql.rows, status="OK")]
    refs: list[dict] = []
    for r in results:
        rows = list(getattr(r, "rows", []) or [])
        if getattr(r, "status", "OK") != "OK" or not rows:
            continue
        qt = getattr(r, "query_type", "detail")
        ref = {
            "ref_id": f"D{len(refs) + 1}",
            "label": _DATA_REF_LABELS.get(qt, "고장 이력"),
            "query_type": qt,
            "sql": _compact_sql(getattr(r, "sql", None)),
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows[:_DATA_REF_ROW_CAP],
            "truncated": len(rows) > _DATA_REF_ROW_CAP,
        }
        window = _date_window(rows)
        if window:
            ref["time_window"] = window
        refs.append(ref)
    return refs


def _render_data_refs_block(refs: list[dict]) -> str:
    """답변 본문에 같이 보이는 텍스트 출처 블록([데이터 출처]). 프론트는 data_refs 필드로 칩 렌더."""
    if not refs:
        return ""
    lines = ["[데이터 출처]"]
    for d in refs:
        win = f" ({d['time_window']})" if d.get("time_window") else ""
        more = "+" if d.get("truncated") else ""
        lines.append(f"- [{d['ref_id']}] {d['label']} — {d['row_count']}{more}행{win}")
        if d.get("sql"):
            sql = d["sql"]
            lines.append(f"  - 쿼리: {sql[:300] + ('…' if len(sql) > 300 else '')}")
    return "\n".join(lines)


def _verify_citations(answer: str, citations: list[dict]) -> tuple[str, list[dict]]:
    """citation 최소 검증: (1) 본문의 [C#] 중 실제 제공된 citation에 없는 번호는 제거,
    (2) 실제로 인용된 citation만 추려 [출처] 블록을 렌더. citation이 없으면 본문 그대로."""
    if not citations:
        return answer, []
    allowed = {str(c.get("citation_id") or f"C{i}") for i, c in enumerate(citations, start=1)}
    # 존재하지 않는 [C#] 토큰 제거(앞 공백도 같이 정리)
    answer = re.sub(r"\s*\[(C\d+)\]", lambda m: m.group(0) if m.group(1) in allowed else "", answer)
    used = set(re.findall(r"\[(C\d+)\]", answer))
    cited = [c for c in citations if str(c.get("citation_id") or "") in used] or []
    if "[출처]" in answer:
        answer = re.split(r"\n\s*\[출처\]\s*", answer, maxsplit=1)[0].rstrip()
    if cited:
        answer = answer.rstrip() + "\n\n" + _format_citations(cited)
    return answer, cited


def final_answer_node(state: ManufacturingState) -> dict:
    # Intake Gate 차단 시: 차단 메시지를 그대로 최종 답변으로 반환(기존과 동일)
    dec = state.get("input_decision")
    if dec and dec.blocked:
        return {"final_answer": FinalAnswer(answer=dec.block_message or "요청을 처리할 수 없습니다.")}

    pred = state.get("prediction_result")
    ev = state.get("evidence_bundle")
    sql = state.get("sql_result")
    packet = state.get("context_packet")

    warnings: list[str] = list(packet.context_warnings) if packet else []
    intake = state.get("intake_decision")
    if intake and intake.output_constraints:
        warnings.extend(intake.output_constraints)
    for art in (pred, ev, sql):
        if art and getattr(art, "limitations", None):
            warnings.extend(art.limitations)
    missing = pred.missing_features if (pred and pred.status == "NEEDS_INPUT") else []
    citations = ev.citations if ev and ev.status in {"OK", "LOW_RELEVANCE"} else []

    # facts sheet 조립(결정적, 입력 정리만) → LLM 자유 합성
    ctx = build_answer_context(state)
    try:
        body = call_llm(FINAL_ANSWER_LLM_SYSTEM_PROMPT,
                        FINAL_ANSWER_LLM_USER_PROMPT.format(**ctx), tier="final").strip()
    except Exception as e:
        body = ""
        warnings.append(f"final_answer_llm_error:{type(e).__name__}")
    if not body:
        body = _fallback_final_answer(ctx)
        warnings.append("final_answer_fallback: empty_llm_answer")

    # 결정적: citation 최소 검증 + SQL 데이터 출처(drill-down) 부착
    answer, cited = _verify_citations(body, citations)
    # 폴백: LLM이 자유 합성에서 인라인 [C#]를 하나도 달지 않아도, 검색된 근거가 있으면
    # 통째로 묻히지 않게 전체 citation을 노출하고 [출처] 블록도 부착한다(프론트 칩 보장).
    if not cited and citations:
        cited = citations
        answer = answer.rstrip() + "\n\n" + _format_citations(citations)
    # 데이터 출처는 본문에 텍스트로 붙이지 않는다(원시 SQL 노출 방지 + 칩과 중복 제거).
    # 프론트가 data_refs 필드로 [D#] drill-down 칩을 렌더한다.
    data_refs = _build_data_refs(sql)

    fa = FinalAnswer(answer=answer, citations=cited, data_refs=data_refs,
                     warnings=warnings, missing_inputs=missing)
    updates = _mark_final_task_pass(state)
    updates["final_answer"] = fa
    return updates


print("final_answer_llm_node(LLM 자유 합성 + citation 최소 검증) 정의 완료")
