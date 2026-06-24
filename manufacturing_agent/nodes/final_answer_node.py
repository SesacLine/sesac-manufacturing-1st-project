from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import ContextPacket, EvidenceArtifact, FinalAnswer, PredictionResult, SQLHistoryArtifact
from manufacturing_agent.contracts.state import ManufacturingState

# ---------- nodes/final_answer_node.py ----------
# 답변 생성 전략(하이브리드):
#   1) 코드가 모든 사실(수치/유형/이력 통계/citation/안전)을 facts sheet로 결정적으로 정리한다.
#   2) LLM(tier="final")이 그 facts sheet 안에서만 자연스러운 해설 답변을 작성한다.
#   3) 품질 피드백 + 숫자 hallucination 가드로 검증하고, 1회 보수한다.
#   4) LLM이 없거나 치명적 문제가 남으면 결정적 폴백 답변으로 안전하게 대체한다.
#   5) 고장 종류별 정확 수치 블록/체크리스트/[출처]는 코드가 결정적으로 보장 첨부한다(숫자 hallucination 불가).
def _citation_display_name(citation: dict) -> str:
    import unicodedata
    raw = str(citation.get("title") or citation.get("source") or citation.get("source_id") or "문서 근거")
    name = raw.split("/")[-1]
    name = re.sub(r"_\d+$", "", name)
    name = re.sub(r"\.(html?|md|pdf)$", "", name, flags=re.I)
    name = unicodedata.normalize("NFC", name).replace("_", " ").strip() or "문서 근거"
    return name[:90].rstrip() + ("..." if len(name) > 90 else "")

def _readable_snippet(raw: Any) -> Optional[str]:
    """raw HTML/공백 손상 PDF 추출물은 비개발자에게 안 읽히므로 가독성 있을 때만 노출한다."""
    snippet = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not snippet:
        return None
    ascii_ratio = sum(ch.isascii() for ch in snippet) / max(len(snippet), 1)
    space_ratio = snippet.count(" ") / max(len(snippet), 1)
    if ascii_ratio < 0.5 and space_ratio > 0.03:
        return snippet[:180].rstrip() + ("…" if len(snippet) > 180 else "")
    return None

def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    # 같은 문서의 여러 chunk/인용을 한 항목으로 묶는다(동일 문서가 [출처]에 중복 나열되는 것을 방지).
    docs: dict[str, dict] = {}
    order: list[str] = []
    for idx, c in enumerate(citations, start=1):
        cid = c.get("citation_id") or f"C{idx}"
        source = str(c.get("source") or c.get("source_id") or "").strip()
        key = source or _citation_display_name(c)
        slot = docs.get(key)
        if slot is None:
            slot = docs[key] = {"ids": [], "title": _citation_display_name(c), "source": source, "chunks": [], "snippet": None}
            order.append(key)
        if cid not in slot["ids"]:
            slot["ids"].append(cid)
        chunk = c.get("chunk_index")
        if chunk is not None and chunk not in slot["chunks"]:
            slot["chunks"].append(chunk)
        if slot["snippet"] is None:
            slot["snippet"] = _readable_snippet(c.get("snippet"))
    lines = ["[출처]"]
    for key in order[:6]:
        d = docs[key]
        lines.append(f"- [{', '.join(d['ids'])}] 문서: {d['title']}")
        if d["source"]:
            lines.append(f"  - 원본: {d['source']}")
        if d["chunks"]:
            lines.append(f"  - 위치: chunk={', '.join(str(x) for x in d['chunks'])}")
        if d["snippet"]:
            lines.append(f"  - 원문 근거: {d['snippet']}")
    return "\n".join(lines)

FEATURE_LABELS = {
    "tool_wear": "공구 마모",
    "torque": "토크",
    "rotational_speed": "회전속도",
    "process_temperature": "공정온도",
    "air_temperature": "공기온도",
    "type": "제품 타입",
}
COMPONENT_LABELS = {
    "tooling": "공구",
    "spindle_bearing": "스핀들 베어링",
    "spindle_drive": "스핀들 드라이브",
    "drive_system": "구동 시스템",
    "coolant_system": "쿨런트 시스템",
    "guard_interlock": "가드 인터록",
    "drive_fan": "드라이브 팬",
}
FAILURE_TYPE_LABELS = {
    "TWF": "TWF(공구 마모 계열)",
    "HDF": "HDF(열/냉각 계열)",
    "OSF": "OSF(과부하 계열)",
    "PWF": "PWF(전원/구동 계열)",
    "SAFETY_INTERLOCK": "안전 인터록",
}

def _label_feature(name: Any) -> str:
    return FEATURE_LABELS.get(str(name), str(name))

def _label_component(name: Any) -> str:
    return COMPONENT_LABELS.get(str(name), str(name))

def _label_failure_type(name: Any) -> str:
    return FAILURE_TYPE_LABELS.get(str(name), str(name))

def _risk_level_ko(level: Any) -> str:
    return {"high": "높음", "medium": "중간", "low": "낮음"}.get(str(level).lower(), str(level))

_SHORT_FT = {"OSF": "과부하", "TWF": "공구마모", "HDF": "열/냉각", "PWF": "전원/구동",
             "SAFETY_INTERLOCK": "안전 인터록"}

def _short_failure(code: Any) -> str:
    return _SHORT_FT.get(str(code), _label_failure_type(code))

FINAL_ANSWER_SYSTEM_PROMPT = """
너는 제조 설비 진단 AI Agent의 최종 답변 작성자다.
공정 엔지니어·설비 엔지니어가 바로 읽고 판단할 수 있는 하나의 자연스러운 답변을 작성하는 것이 역할이다.

역할 범위:
- 아래 facts sheet 안의 정보만 사용해 서술 본문을 작성한다. 없는 정보는 "확인 필요"로 표현한다.
- 수치 블록(고장 근거 표·체크리스트·[출처])과 맨 앞 종합 판단 배너는 시스템이 정확한 값으로 자동 첨부한다.
  너는 그것을 직접 만들거나 본문에서 반복하지 않는다.

본문 작성 규칙:
- [섹션 작성 지침]을 최우선으로 따른다. 섹션 개수·구성·문체는 해당 지침만 기준으로 한다.
- facts sheet에 없는 숫자(토크·온도·건수·비율·임계값 등)는 절대 만들지 않는다.
  실제 측정·집계값은 단위를 붙여 구체적으로 쓴다. 예: "토크 62 N·m", "30일 10건 420분".
- 본문 핵심은 두 가지다. ① 가장 심각한 위험 원인 1~2가지가 "왜 위험한지" ② "가장 먼저 확인할 것".
  계산식·임계값 목록·점검 체크리스트는 시스템이 뒤에 붙이므로 본문에 쓰지 않는다.
- [답변 모드]가 NEEDS_INPUT일 때만 입력 부족을 안내한다. 그 외 모드에서 입력 부족 표현을 쓰지 않는다.
- citation이 있으면 관련 문장에 [C1] 형식으로 표시한다. 없는 근거를 새로 만들지 않는다.
- raw 코드(tool_wear, drive_system, score 수치, query_type 등)는 출력하지 않는다.
- [안전 판단 요약]의 현장 확인 권고는 본문 말미에 자연스럽게 녹인다. 승인·지시 표현은 쓰지 않는다.
- 내부 처리 과정, Agent 이름, 라우팅 경로는 설명하지 않는다.

한국어로, 간결하게 작성한다. 최종 출력에는 자기검토 과정을 포함하지 말고 답변 본문만 작성하라.
""".strip()

FINAL_ANSWER_USER_PROMPT = """
사용자 질문:
{user_question}

답변 대상:
{equipment_id}
(답변 대상은 내부 요약 기준이다. 제목에 그대로 복사하지 말고 자연스러운 한국어 제목으로 바꿔라.
 구체 설비명이 없으면 "이 설비", "해당 설비", "최근 설비에서" 같은 표현 대신 "최근 고장 이력에서", "조회된 고장 사례에서"라고 쓴다.)

아래 facts sheet 안에서만 답변하라. 없는 정보는 추정하지 말고 "확인 필요"라고 표현하라.

[답변 모드]
{answer_mode}

[섹션 작성 지침]
{section_guidance}

[현재 위험 진단 요약]
{prediction_summary}

[최근 이력 — 검증된 수치(이 숫자만 사용, 표현·구성은 네가 자연스럽게)]
{history_summary}

[문서 근거 요약]
{evidence_summary}

[안전 판단 요약]
{safety_summary}

[정확 수치 근거(이 수치만 사용; 시스템이 표로 자동 첨부하므로 본문에서는 해석만)]
{diagnosis_block}

[사용 가능한 Citation 목록]
{citations}

위 정보를 바탕으로 사용자에게 보여줄 최종 답변 본문만 작성하라.
artifact 이름, SQL row, JSON, 내부 처리 과정, score 값은 출력하지 마라.
""".strip()

def _answer_equipment_id(state: ManufacturingState, sql: Optional[SQLHistoryArtifact], packet: Optional[ContextPacket]) -> str:
    pred = state.get("prediction_result")
    if pred and sql:
        return "입력 피처와 과거 고장 이력"
    if pred:
        return "입력 피처 샘플"
    if sql:
        return "과거 고장 이력"
    return "제조 설비 점검"

def _answer_title_from_context(ctx: dict) -> str:
    subject = ctx.get("equipment_id") or ""
    if subject == "입력 피처와 과거 고장 이력":
        return "입력 피처 기반 위험 진단과 과거 고장 이력 요약"
    if subject == "입력 피처 샘플":
        return "입력 피처 기반 위험 진단 요약"
    if subject == "과거 고장 이력":
        return "과거 고장 이력 요약"
    return "제조 점검 답변 요약"

def _format_measured_values(machine_values: Optional[dict]) -> str:
    """진단에 사용된 실제 입력 측정값을 '라벨 값단위' 형태로 푼다."""
    if not machine_values:
        return ""
    _units = {"air_temperature": "K", "process_temperature": "K",
              "rotational_speed": "rpm", "torque": "N·m", "tool_wear": "분"}
    parts = []
    for k, v in machine_values.items():
        unit = getattr(v, "unit", None) or _units.get(k, "")
        val = getattr(v, "value", v)
        parts.append(f"{_label_feature(k)} {val}{unit}".strip())
    return ", ".join(parts)

def _prediction_summary_for_answer(pred: Optional[PredictionResult], machine_values: Optional[dict] = None) -> str:
    if not pred:
        return "현재 위험 진단은 이번 요청에서 별도로 수행되지 않음. 이 문장은 위험이 없다는 의미가 아니며, 위험 없음으로 표현하지 말 것."
    if pred.status == "NEEDS_INPUT":
        return "입력 부족: " + ", ".join(pred.missing_features or [])
    levels = [str(r.get("level", "")).lower() for r in (pred.risk_flags or [])]
    if "high" in levels:
        risk_level = "높음"
    elif "medium" in levels:
        risk_level = "중간"
    elif pred.risk_flags:
        risk_level = "낮음"
    else:
        risk_level = "뚜렷한 고위험 신호 없음"
    lines = [f"진단 상태: 완료, 현재 위험 수준: {risk_level}, 신뢰도: {pred.confidence}"]
    measured = _format_measured_values(machine_values)
    if measured:
        lines.append("사용된 입력 측정값: " + measured)
    if pred.summary:
        lines.append(pred.summary)
    if pred.risk_flags:
        brief = [f"{_label_failure_type(r.get('failure_type'))} {_risk_level_ko(r.get('level'))}"
                 f"(영향: {', '.join(_label_feature(x) for x in (r.get('contributing_features') or [])) or '확인 필요'})"
                 for r in pred.risk_flags]
        lines.append("감지된 위험: " + ", ".join(brief))
    if pred.context_mode in {"PATCH_ACTIVE", "USE_ACTIVE", "SELECT_HISTORY"}:
        if pred.changed_features:
            lines.append("변경 입력: " + ", ".join(_label_feature(x) for x in pred.changed_features))
        if pred.reused_features:
            lines.append("재사용한 진단 context feature: " + ", ".join(_label_feature(x) for x in pred.reused_features[:6]))
    if pred.limitations:
        lines.append("한계: " + "; ".join(pred.limitations[:3]))
    return "\n".join(lines)

def _sample_failure_rows(rows: list[dict]) -> list[str]:
    out = []
    for r in rows[:3]:
        dt = r.get("downtime_min")
        dt_s = f" · {dt}분" if str(dt or "").strip() not in ("", "None") else ""
        date = str(r.get("event_date") or "")[5:] or str(r.get("event_date") or "")
        sym = str(r.get("symptom") or r.get("corrective_action") or "").strip()
        out.append(f"{date} {_short_failure(r.get('failure_type'))} {_label_component(r.get('component'))}{dt_s} — {sym}")
    return out

def _format_counter(counter: Counter, unit: str = "건", limit: int = 5) -> str:
    if not counter:
        return "확인 필요"
    return ", ".join(f"{name} {count}{unit}" for name, count in counter.most_common(limit))

def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _dedup_take(values, limit: int) -> list[str]:
    """빈 값 제거 + 중복 제거하며 순서 보존으로 최대 limit개를 취한다."""
    out: list[str] = []
    for v in values:
        s = str(v or "").strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out

def _detail_identity(r: dict):
    """detail 행의 사건 식별 키. 같은 사건을 다른 정렬/투영으로 조회한 중복을 잡는다.
    id가 가장 정확하지만 투영마다 빠질 수 있어(쿼리별 SELECT 컬럼이 다름) 사용하지 않고,
    detail 쿼리에 공통으로 존재하는 사건 컬럼 조합으로 식별한다.
    event_date·failure_type이 없으면 식별 불가로 보고 dedup 대상에서 제외한다(보수적으로 유지)."""
    date = str(r.get("event_date") or "").strip()[:10]
    ftype = str(r.get("failure_type") or "").strip()
    if not date or not ftype:
        return None
    return (date, ftype, str(r.get("component") or "").strip(), str(r.get("downtime_min") or "").strip())


def _dedup_detail_rows(rows: list[dict]) -> list[dict]:
    """여러 detail 쿼리(같은 모집단의 다른 정렬/투영) 결과를 합칠 때 사건 중복을 제거한다.
    먼저 등장한 행(보통 컬럼이 더 풍부한 첫 쿼리)을 보존한다. 식별 불가 행은 그대로 둔다."""
    seen, out = set(), []
    for r in rows:
        key = _detail_identity(r)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out.append(r)
    return out


def _history_facts(sql: Optional[SQLHistoryArtifact]) -> str:
    """SQL 이력에서 '정확한 수치'만 결정적으로 추출한 검증 facts(verified numbers).
    문장 구조/섹션/표현은 최종 답변 LLM이 담당한다 — 여기서는 숫자 환각만 차단한다(고정 틀 없음)."""
    if not sql:
        return "확인된 최근 이력 없음"
    if sql.status == "INVALID_REQUEST":
        return sql.summary or "이력 조회 조건이 부족함"
    if sql.status == "EMPTY":
        return "조건에 맞는 과거 이력은 조회되지 않음"
    if sql.status in {"BLOCKED", "FAIL"}:
        return "이력 조회 실패 또는 정책 차단: " + (sql.error_message or sql.summary or "확인 필요")

    detail_rows: list[dict] = []
    agg_rows: list[dict] = []
    results = getattr(sql, "results", []) or []
    if results:
        for r in results:
            (agg_rows if r.query_type == "aggregate" else detail_rows).extend(r.rows or [])
    elif sql.rows:
        (agg_rows if sql.query_type == "aggregate" else detail_rows).extend(sql.rows)

    # 여러 detail 쿼리가 같은 사건을 다른 정렬/투영으로 조회하면 행이 중복된다 → 집계 전 사건 단위 dedup.
    detail_rows = _dedup_detail_rows(detail_rows)

    facts: list[str] = []
    if detail_rows:
        facts.append(f"조회된 고장 사례: {len(detail_rows)}건")
        downtimes = [_to_int(r.get("downtime_min"), 0) for r in detail_rows
                     if str(r.get("downtime_min") or "").strip() not in ("", "None")]
        if downtimes:
            facts.append(f"다운타임 합계 {sum(downtimes)}분, 평균 {round(sum(downtimes) / len(downtimes))}분")
        facts.append("고장 유형 분포: " + _format_counter(Counter(_short_failure(r.get("failure_type")) for r in detail_rows if r.get("failure_type"))))
        facts.append("영향 영역 분포: " + _format_counter(Counter(_label_component(r.get("component")) for r in detail_rows if r.get("component"))))
        samples = _sample_failure_rows(detail_rows[:3])
        if samples:
            facts.append("대표 사례: " + " / ".join(samples))
        actions = _dedup_take((r.get("corrective_action") for r in detail_rows), 3)
        if actions:
            facts.append("확인된 대응 조치: " + " · ".join(actions))
        preventions = _dedup_take((r.get("preventive_action") for r in detail_rows), 3)
        if preventions:
            facts.append("재발 방지 조치: " + " · ".join(preventions))
    if agg_rows:
        grouped: dict[str, dict[str, Any]] = {}
        for row in agg_rows:
            ft = _short_failure(row.get("failure_type"))
            slot = grouped.setdefault(ft, {"cases": 0, "downtime": 0, "components": Counter()})
            slot["cases"] += _to_int(row.get("case_count"), 1)
            slot["downtime"] += _to_int(row.get("total_downtime_min"), 0)
            if row.get("component"):
                slot["components"][_label_component(row.get("component"))] += _to_int(row.get("case_count"), 1)
        for ft, data in sorted(grouped.items(), key=lambda kv: (-kv[1]["cases"], -kv[1]["downtime"], kv[0]))[:5]:
            facts.append(f"{ft} 집계: {data['cases']}건, 다운타임 {data['downtime']}분, 주요 영역 {_format_counter(data['components'], limit=2)}")
    if not facts:
        facts.append(sql.summary or "조건에 맞는 이력 없음")
    if sql.limitations:
        facts.append("조회 한계: " + "; ".join(sql.limitations[:3]))
    return "\n".join(facts)

def _citation_list_for_answer(citations: list[dict]) -> str:
    if not citations:
        return "사용 가능한 citation 없음"
    return "\n".join(f"[{c.get('citation_id') or f'C{idx}'}] {c.get('title') or _citation_display_name(c)}" for idx, c in enumerate(citations[:5], start=1))

def _evidence_summary_for_answer(ev: Optional[EvidenceArtifact]) -> str:
    if not ev:
        return "확인된 문서 근거 없음"
    if ev.status == "OK":
        return ev.evidence_summary or "문서 근거는 검색됐지만 요약이 비어 있음"
    if ev.status == "LOW_RELEVANCE":
        limited = ev.evidence_summary or "현재 검색된 문서 근거의 관련성이 낮아 단정하기 어려움"
        return limited + " citation은 참고용이며, 추가 문서 확인이 필요함."
    if ev.status == "EMPTY":
        return "현재 검색된 문서 근거만으로는 단정하기 어려움"
    return "문서 근거 조회 실패: " + ("; ".join(ev.limitations[:3]) or "문서 근거를 가져오지 못했습니다.")

def _answer_mode(pred: Optional[PredictionResult], sql: Optional[SQLHistoryArtifact], ev: Optional[EvidenceArtifact]) -> str:
    has_pred = pred is not None and pred.status not in {"SKIPPED"}
    has_sql = sql is not None
    has_evidence = ev is not None and ev.status in {"OK", "LOW_RELEVANCE", "EMPTY"}
    if has_pred and has_sql:
        return "COMBINED"
    if has_sql and has_evidence:
        return "HISTORY_WITH_EVIDENCE"
    if has_sql:
        return "SQL_ONLY"
    if has_pred and has_evidence:
        return "PREDICTION_WITH_EVIDENCE"
    if has_pred:
        return "PREDICTION_ONLY"
    if has_evidence:
        return "EVIDENCE_ONLY"
    return "GENERAL"

def _section_guidance_for_answer(mode: str, ev: Optional[EvidenceArtifact], citations: list[dict]) -> str:
    if mode == "SQL_ONLY":
        return (
            "최근 고장 이력 조회 답변이다. 제공된 '검증된 수치'를 바탕으로 자연스럽게 서술한다. "
            "고정된 섹션 틀이나 불릿 양식을 강제하지 말고, 핵심 수치(건수·다운타임·유형 분포·대표 조치)와 해석상 주의사항을 자연스러운 흐름으로 담는다. "
            "현재 위험 진단·지금 점검할 일·문서 근거 섹션은 만들지 않는다. "
            "점검 권고를 하더라도 SQL 이력에서 확인된 조치 패턴 수준으로만 표현한다. 700자 내외로 간결하게 작성한다."
        )
    if mode in {"COMBINED", "PREDICTION_WITH_EVIDENCE"}:
        return (
            "현재 위험 진단 → 최근 이력 요약 → 문서 근거 → 주의사항 순서로 작성한다. "
            "문서 citation이 있으면 본문에 [C1] 형태로 표시하고, 없는 문서 근거를 새로 만들지 않는다. "
            "'고장 근거 표'와 '지금 점검할 일' 체크리스트는 시스템이 자동 첨부한다 — 본문에서 만들지 않는다."
        )
    if mode == "PREDICTION_ONLY":
        return (
            "입력 피처 기반 위험 진단을 해설한다. 정확한 수치 근거와 점검 항목은 시스템이 본문 뒤에 정확한 값으로 따로 정리해 붙인다. "
            "그러니 본문에서는 표·번호 목록·섹션 제목을 직접 만들지 말고, 같은 계산식·수치를 반복하지 마라. "
            "무엇이 왜 위험한지와 가장 먼저 할 일을 2~4문장으로 자연스럽게 설명한다. 과거 이력이나 문서 근거 섹션은 만들지 않는다."
        )
    if mode == "EVIDENCE_ONLY":
        return "문서 근거와 점검 절차 중심으로 작성한다. 현재 위험 진단이나 과거 이력 섹션은 만들지 않는다."
    if mode == "HISTORY_WITH_EVIDENCE":
        return "고장 이력 요약과 문서 근거를 연결해 작성한다. 현재 위험 진단 섹션은 만들지 않는다."
    return "사용자 질문에 직접 답하되, 없는 artifact를 근거로 한 섹션은 만들지 않는다."

_SAFETY_ACTION_KO = {
    "ALLOW": "일반 제조 질문",
    "ANSWER_SAFELY": "안전 자문 요청 (모델이 재가동·조치 승인을 대신하지 않음)",
    "BLOCK_DANGEROUS_EXECUTION": "위험 실행 요청 — 안내 불가",
    "HUMAN_HANDOFF": "현장 책임자 직접 확인 필요",
}

def _safety_summary_for_answer(state: ManufacturingState, pred: Optional[PredictionResult]) -> str:
    lines = []
    intake = state.get("intake_decision")
    if intake:
        action_ko = _SAFETY_ACTION_KO.get(intake.safety_action, intake.safety_action)
        lines.append(f"요청 성격: {action_ko}. {intake.safety_reason}")
    if pred and pred.safety_hints:
        for h in pred.safety_hints[:3]:
            required = ", ".join(h.required_checks or []) or "현장 확인 필요"
            avoid = ", ".join(h.avoid_actions or []) or "위험 작업 임의 진행 금지"
            lines.append(f"{h.risk_level}: {h.reason}; 필요 확인={required}; 회피={avoid}")
    lines.append("정지/재가동/정비 승인 여부는 현장 안전 책임자와 설비 담당자가 판단해야 함")
    return "\n".join(lines)

_RISK_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}

def _render_diagnosis_block(pred: Optional[PredictionResult]) -> str:
    """고장 종류별 근거(규칙/계산/영향 변수)를 결정적으로 렌더한다. 정상(낮음)은 한 줄로 접는다."""
    if not pred or not pred.risk_flags:
        return ""
    lines = ["고장 종류별 근거"]
    low = []
    for r in pred.risk_flags:
        lvl = str(r.get("level", "")).lower()
        if lvl == "low":
            low.append(_label_failure_type(r.get("failure_type")))
            continue
        emoji = _RISK_EMOJI.get(lvl, "•")
        drivers = " · ".join(_label_feature(x) for x in (r.get("contributing_features") or [])) or "확인 필요"
        lines.append(f"{emoji} {_label_failure_type(r.get('failure_type'))} — {_risk_level_ko(r.get('level'))}")
        if r.get("rule"):
            lines.append(f"   규칙 : {r['rule']}")
        if r.get("formula") or r.get("detail"):
            lines.append(f"   계산 : {r.get('formula') or r.get('detail')}")
        lines.append(f"   영향 변수 : {drivers}")
    if low:
        lines.append("🟢 정상(현재 위험 낮음): " + ", ".join(low))
    return "\n".join(lines)

def _render_checklist(pred: Optional[PredictionResult]) -> str:
    """위험 높은 순으로 권장 점검을 중복 없이 모아 체크리스트로 렌더한다."""
    if not pred or not pred.risk_flags:
        return ""
    seen, items = set(), []
    for r in pred.risk_flags:
        for chk in (r.get("recommended_checks") or []):
            if chk and chk not in seen:
                seen.add(chk)
                items.append(chk)
    if not items:
        return ""
    return "지금 점검할 일\n" + "\n".join(f"{i}. {it}" for i, it in enumerate(items[:4], 1))

def _ensure_diagnosis_block(answer: str, pred: Optional[PredictionResult]) -> str:
    """LLM 답변에 결정적 근거 블록/체크리스트가 없으면 정확한 수치로 덧붙인다(인용 [출처]보다 앞)."""
    if not pred or not pred.risk_flags:
        return answer
    out = (answer or "").rstrip()
    block = _render_diagnosis_block(pred)
    # 이미 렌더된 근거표가 있을 때만 건너뛴다. 헤딩 문구가 아니라 '렌더 전용 마커(영향 변수 :)'로 판정해야
    # LLM이 본문에서 섹션명을 언급해도 실제 표가 누락되지 않는다.
    if block and "영향 변수 :" not in out:
        out += "\n\n" + block
    checklist = _render_checklist(pred)
    if checklist and checklist not in out:
        out += "\n\n" + checklist
    return out

def _verdict_banner(pred, sql, ev) -> str:
    """답변 맨 앞 한 줄 종합 판단(결정적)."""
    if pred and getattr(pred, "status", None) == "NEEDS_INPUT":
        return "ℹ️ 종합 판단: 입력 부족 — 정확한 진단을 위해 추가 데이터가 필요합니다."
    if pred and pred.risk_flags:
        levels = [str(r.get("level", "")).lower() for r in pred.risk_flags]
        lv, emo = ("높음", "🔴") if "high" in levels else ("중간", "🟡") if "medium" in levels else ("낮음", "🟢")
        types = [_short_failure(r.get("failure_type")) for r in pred.risk_flags
                 if str(r.get("level", "")).lower() in {"high", "medium"}]
        ts = " · ".join(dict.fromkeys(types)) or "주의 신호"
        return f"{emo} 종합 판단: 위험 {lv} — {ts}"
    if pred and getattr(pred, "status", None) in {"OK", "PARTIAL"}:
        return "🟢 종합 판단: 입력 기준 뚜렷한 고위험 신호 없음"
    if sql is not None:
        return "🗂 종합 판단: 과거 고장 이력 요약"
    if ev is not None:
        return "📄 종합 판단: 점검 문서 근거 요약"
    return "ℹ️ 종합 판단"

def build_answer_context(state: ManufacturingState) -> dict:
    pred = state.get("prediction_result")
    ev = state.get("evidence_bundle")
    sql = state.get("sql_result")
    packet = state.get("context_packet")
    citations = ev.citations if ev and ev.status in {"OK", "LOW_RELEVANCE"} else []
    mode = _answer_mode(pred, sql, ev)
    machine_values = packet.selected_machine_values if packet else None
    prediction_summary = _prediction_summary_for_answer(pred, machine_values) if pred else "이번 답변 모드에서는 현재 위험 진단 섹션을 만들지 않는다."
    evidence_summary = _evidence_summary_for_answer(ev) if ev else "이번 요청에서 문서 근거 artifact가 없으므로 문서 근거 섹션을 만들지 않는다."
    diagnosis_block = _render_diagnosis_block(pred)
    checklist_block = _render_checklist(pred)
    return {
        "user_question": state.get("user_message", ""),
        "equipment_id": _answer_equipment_id(state, sql, packet),
        "answer_mode": mode,
        "section_guidance": _section_guidance_for_answer(mode, ev, citations),
        "prediction_summary": prediction_summary,
        "history_summary": _history_facts(sql),
        "evidence_summary": evidence_summary,
        "safety_summary": _safety_summary_for_answer(state, pred),
        "diagnosis_block": diagnosis_block or "해당 없음(현재 위험 진단 수치 없음)",
        "checklist_block": checklist_block,
        "citations": _citation_list_for_answer(citations),
    }

# ---------- 품질/숫자 가드 ----------
def _final_answer_quality_feedback(ctx: dict, answer: str) -> list[str]:
    issues: list[str] = []
    mode = ctx.get("answer_mode")
    if mode == "SQL_ONLY":
        banned_sections = ["현재 판단", "지금 점검할 일", "문서 근거"]
        leaked = [s for s in banned_sections if s in answer]
        if leaked:
            issues.append("SQL_ONLY 답변에는 다음 섹션을 만들지 마세요: " + ", ".join(leaked))
    if re.search(r"\bscore\b|점수\s*\(?\d", answer, re.I):
        issues.append("내부 score/점수 값을 노출하지 말고 높음/중간/낮음 정도로 표현하세요.")
    _leak_terms = list(COMPONENT_LABELS) + [t for t in FEATURE_LABELS if t != "type"]
    raw_terms = [t for t in _leak_terms if re.search(rf"\b{re.escape(t)}\b", answer)]
    if raw_terms:
        issues.append("raw schema 용어를 한국어 현장 용어로 풀어 쓰세요: " + ", ".join(sorted(set(raw_terms))[:6]))
    if ctx.get("citations") != "사용 가능한 citation 없음" and not re.search(r"\[C\d+\]", answer):
        issues.append("사용 가능한 citation이 있으면 관련 문장에 [C1] 형식으로 표시하세요.")
    if ctx.get("answer_mode") != "NEEDS_INPUT" and "입력 부족" in answer:
        issues.append("입력 부족 상태가 아니므로 [입력 부족] 섹션이나 표현을 제거하세요.")
    if re.search(r"(?m)^\s*#{1,6}\s+", answer):
        issues.append("markdown # heading marker를 쓰지 말고 짧은 일반 섹션 제목으로 작성하세요.")
    if re.search(r"조정하여\s*테스트|바로\s*재가동|계속\s*운전", answer):
        issues.append("운전 조건 변경이나 테스트 수행을 직접 지시하지 말고 승인된 절차에서 검토할 항목으로 표현하세요.")
    return issues

_UNIT_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(N·m|Nm|rpm|RPM|K|℃|분|건|%|mm|시간|회)")
# 단위 없는 고위험 정량 주장만 좁게 잡는다: 배수(배)와 퍼센트포인트(%p/퍼센트포인트).
# 표준 단위가 없어 _UNIT_NUM_RE로는 못 잡지만 "약 3배", "12%p"처럼 사실상 측정값 주장이다.
_MULT_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(배|%p|퍼센트포인트)")

def _allowed_numbers(ctx: dict) -> set[str]:
    """facts sheet의 '측정값' 키에 등장한 수치 토큰. LLM 본문 수치는 이 집합 안에 있어야 한다.
    citations(예: [C1], [C2], 문서 제목·파일명 속 숫자)는 측정값이 아니라 인용 식별자/색인이므로,
    그 숫자가 본문의 측정 수치(예: 5건)를 잘못 화이트리스트해 hallucination을 통과시키지 않도록 제외한다."""
    allowed: set[str] = set()
    for key in ("prediction_summary", "history_summary", "evidence_summary",
                "safety_summary", "diagnosis_block", "checklist_block"):
        allowed |= set(re.findall(r"\d+(?:\.\d+)?", ctx.get(key, "") or ""))
    return allowed

def _number_guard(answer: str, allowed: set[str]) -> list[str]:
    """단위가 붙은 수치(토크/온도/건수/다운타임 등)가 facts sheet에 없으면 hallucination으로 본다.
    단위 없는 일반 숫자·목록 번호는 오탐 위험이 커서 검사하지 않는다.
    단, 배수(배)·퍼센트포인트(%p)는 단위가 없어도 명백한 정량 주장이라 별도로 검사한다."""
    issues: list[str] = []
    for m in _UNIT_NUM_RE.finditer(answer or ""):
        tok = m.group(1)
        if tok in allowed:
            continue
        if "." not in tok and len(tok) <= 1:   # 목록 번호 등 한 자리 숫자
            continue
        issues.append(f"number_hallucination:{m.group(0).strip()} → facts sheet에 없는 수치이니 제거하거나 facts 값으로 교체하세요.")
        if len(issues) >= 5:
            break
    # 배수/%p는 한 자리 숫자(예: 3배)도 실제 주장이므로 목록 번호 예외를 적용하지 않는다.
    for m in _MULT_NUM_RE.finditer(answer or ""):
        if len(issues) >= 5:
            break
        tok = m.group(1)
        if tok in allowed:
            continue
        issues.append(f"number_hallucination:{m.group(0).strip()} → facts sheet에 없는 수치이니 제거하거나 facts 값으로 교체하세요.")
    return issues

# ---------- 후처리 ----------
def _ensure_citations_visible(answer: str, citations: list[dict]) -> str:
    if not citations:
        return answer
    # 본문에서 실제 인용된 [C#]만 출처로 노출한다(인용 안 된 문서 나열 방지). 하나도 매칭 안 되면 전체로 폴백.
    used = set(re.findall(r"\[(C\d+)\]", answer))
    if used:
        cited = [c for c in citations if (c.get("citation_id") or "") in used]
        if cited:
            citations = cited
    if "[출처]" in answer:
        answer = re.split(r"\n\s*\[출처\]\s*", answer, maxsplit=1)[0].rstrip()
    return answer.rstrip() + "\n\n" + _format_citations(citations)

def _ensure_missing_input_visible(answer: str, missing_inputs: list[str]) -> str:
    if not missing_inputs:
        return answer
    if "입력 부족" in answer or ("입력" in answer and any(term in answer for term in ["부족", "확인 필요", "추가 정보", "추가 입력"])):
        return answer
    missing_text = ", ".join(_label_feature(name) for name in missing_inputs)
    prefix = (
        "[입력 부족]\n"
        f"이번 질문에서 제공된 값만으로는 전체 위험 진단이 제한됩니다. 추가 입력이 필요합니다: {missing_text}.\n\n"
    )
    return prefix + answer.lstrip()

def _remove_false_missing_input_section(answer: str, missing_inputs: list[str]) -> str:
    if missing_inputs or "입력 부족" not in answer:
        return answer
    cleaned = re.sub(r"\n?\[입력 부족\]\s*\n.*?(?=\n(?:#{1,3}\s|[가-힣A-Za-z ]{2,20}\n)|\Z)", "\n", answer, flags=re.S).strip()
    return cleaned or answer

def _localize_answer_terms(answer: str) -> str:
    out = answer
    feature_labels = {k: v for k, v in FEATURE_LABELS.items() if k != "type"}
    for raw, label in {**COMPONENT_LABELS, **feature_labels}.items():
        out = re.sub(rf"\b{re.escape(raw)}\b", label, out)
    phrase_replacements = {
        "Corrective Action": "시정 조치",
        "Preventive Action": "예방 조치",
        "Root Cause": "근본 원인",
        "Failure Type": "고장 유형",
        "Tooling": "공구",
        "tooling": "공구",
    }
    for raw, label in phrase_replacements.items():
        out = out.replace(raw, label)
    return out

def _clean_final_answer_format(answer: str) -> str:
    # Notebook 출력에서 보고서식 markdown heading marker가 과하게 보이지 않도록 정리한다.
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s+", "", answer or "")
    cleaned = cleaned.replace("시정 조치으로", "시정 조치로")
    cleaned = cleaned.replace("예방 조치으로", "예방 조치로")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def _fallback_final_answer(ctx: dict) -> str:
    """LLM 합성이 불가하거나 검증 실패 시의 결정적 답변. 숫자는 facts sheet 값만 사용한다."""
    title = _answer_title_from_context(ctx)
    if ctx.get("answer_mode") == "SQL_ONLY":
        return (
            f"{title}\n\n"
            f"조회 결과 요약\n{ctx['history_summary']}\n\n"
            "해석상 주의사항\n- 이 요약은 저장된 failure_history 샘플 이력 기준입니다. 실제 정비 판단은 현장 점검과 담당자 승인 기준으로 확인해야 합니다."
        )
    parts = [title, "현재 확인된 정보 기준으로 종합하면 다음과 같습니다. 단, 일부 판단은 추가 현장 확인이 필요합니다."]
    if not ctx.get("prediction_summary", "").startswith("이번 답변 모드"):
        parts.append("현재 판단\n" + ctx["prediction_summary"])
    if ctx.get("history_summary") and ctx["history_summary"] != "확인된 최근 이력 없음":
        parts.append("최근 이력 요약\n" + ctx["history_summary"])
    if not ctx.get("evidence_summary", "").startswith("이번 요청에서 문서 근거"):
        parts.append("문서 근거\n" + ctx["evidence_summary"])
    parts.append("주의사항\n" + ctx["safety_summary"])
    return "\n\n".join(parts)

def _mark_final_task_pass(state: ManufacturingState) -> dict:
    plan = state.get("execution_plan")
    active = state.get("active_task_id")
    if not plan:
        return {}
    tasks = [t.model_copy(deep=True) for t in plan.tasks]
    changed = False
    for task in tasks:
        if task.task_type == "final_answer" and (task.task_id == active or task.status == "RUNNING"):
            task.status = "PASS"
            changed = True
    return {"execution_plan": plan.model_copy(update={"tasks": tasks})} if changed else {}

_SAFETY_TRAILER_RISK = "⚠ 규칙 기반 보조 진단이며, 정지·재가동·정비 승인 여부는 현장 안전 책임자와 설비 담당자가 판단해야 합니다."
_SAFETY_TRAILER_INFO = "ℹ 보조 진단·조회 결과이며, 실제 조치는 현장 담당자 확인이 필요합니다."

def _ensure_safety_trailer(answer: str, has_risk: bool) -> str:
    trailer = _SAFETY_TRAILER_RISK if has_risk else _SAFETY_TRAILER_INFO
    core = "현장 안전 책임자" if has_risk else "현장 담당자 확인"
    if core in answer:
        return answer
    return answer.rstrip() + "\n\n" + trailer

def _synthesize_answer(ctx: dict, allowed_numbers: set[str]) -> tuple[str, list[str]]:
    """LLM(tier=final) 합성 → 품질/숫자 가드 → 1회 보수. (answer, 남은 issues) 반환.
    LLM 사용 불가/빈 응답이면 ('', [사유])."""
    user_prompt = FINAL_ANSWER_USER_PROMPT.format(**ctx)
def _verdict_banner(pred, sql, ev) -> str:
    """답변 맨 앞 한 줄 종합 판단(결정적)."""
    if pred and getattr(pred, "status", None) == "NEEDS_INPUT":
        return "ℹ️ 종합 판단: 입력 부족 — 정확한 진단을 위해 추가 데이터가 필요합니다."
    if pred and pred.risk_flags:
        levels = [str(r.get("level", "")).lower() for r in pred.risk_flags]
        lv, emo = ("높음", "🔴") if "high" in levels else ("중간", "🟡") if "medium" in levels else ("낮음", "🟢")
        types = [_short_failure(r.get("failure_type")) for r in pred.risk_flags
                 if str(r.get("level", "")).lower() in {"high", "medium"}]
        ts = " · ".join(dict.fromkeys(types)) or "주의 신호"
        return f"{emo} 종합 판단: 위험 {lv} — {ts}"
    if pred and getattr(pred, "status", None) in {"OK", "PARTIAL"}:
        return "🟢 종합 판단: 입력 기준 뚜렷한 고위험 신호 없음"
    if sql is not None:
        return "🗂 종합 판단: 과거 고장 이력 요약"
    if ev is not None:
        return "📄 종합 판단: 점검 문서 근거 요약"
    return "ℹ️ 종합 판단"

def _missing_block(pred) -> str:
    if not (pred and getattr(pred, "status", None) == "NEEDS_INPUT"):
        return ""
    miss = ", ".join(_label_feature(x) for x in (pred.missing_features or [])) or "추가 입력값"
    return ("추가로 필요한 입력\n"
            f"현재 입력만으로는 정확한 진단이 어렵습니다. 다음 값을 입력해 주세요: {miss}")

def _history_block(sql) -> str:
    if sql is None:
        return ""
    body = _history_summary_for_answer(sql)
    if not body:
        return ""
    return "과거 고장 이력\n" + body

def _evidence_block(ev) -> str:
    if not ev:
        return ""
    status = getattr(ev, "status", None)
    if status == "OK" and ev.evidence_summary:
        _parts = [p.strip() for p in ev.evidence_summary.split("\n") if p.strip()]
        _cited = [p for p in _parts if "[C" in p][:4]
        return "문서 근거\n" + ("\n".join(_cited) if _cited else ev.evidence_summary[:600])
    if status == "LOW_RELEVANCE":
        body = ev.evidence_summary or "검색된 문서의 관련성이 낮아 단정하기 어렵습니다."
        return "문서 근거\n" + body + "\n(관련성이 낮아 참고용입니다. 추가 문서 확인이 필요합니다.)"
    if status == "EMPTY":
        # NO_EVIDENCE 등으로 근거가 없을 때: evidence_agent가 넣은 담당자 확인 안내를 그대로 노출(추측 금지).
        msg = (getattr(ev, "evidence_summary", "") or "").strip()
        return "문서 근거\n" + (msg or "현재 검색된 문서 근거만으로는 단정하기 어렵습니다.")
    return ""  # FAIL → 섹션 생략

HEADLINE_SYS = (
    "너는 제조 진단 답변의 첫 요약 문장 작성자다. 주어진 진단/이력/문서 요지를 바탕으로, "
    "사용자가 가장 먼저 알아야 할 핵심과 가장 먼저 할 일을 1~2문장으로만 쓴다. "
    "표·수치 나열·체크리스트·섹션 제목은 쓰지 마라(뒤에 시스템이 정확한 수치로 붙인다). "
    "위험한 실행 지시(점검 없이 재가동 등)는 절대 하지 말고, 단정 대신 '확인 필요'·'점검 필요'로 표현한다. "
    "한국어 1~2문장만 출력하라."
)

def _headline(pred, sql, ev, user_q: str) -> str:
    facts = []
    if pred and pred.risk_flags:
        top = pred.risk_flags[0]
        facts.append(f"진단: {_short_failure(top.get('failure_type'))} 등 위험 {_risk_level_ko(top.get('level'))}")
    elif pred and getattr(pred, "status", None) == "NEEDS_INPUT":
        facts.append("진단: 입력 부족으로 추가 데이터 필요")
    if sql is not None:
        facts.append("과거 고장 이력 조회됨")
    if ev is not None and getattr(ev, "status", None) in {"OK", "LOW_RELEVANCE"}:
        facts.append("관련 문서 근거 있음")
    try:
        answer = call_llm(FINAL_ANSWER_SYSTEM_PROMPT, user_prompt, tier="final").strip()
    except Exception as e:
        return "", [f"llm_error:{type(e).__name__}"]
    if not answer:
        return "", ["empty_answer"]
    issues = _final_answer_quality_feedback(ctx, answer) + _number_guard(answer, allowed_numbers)
    if issues:
        repair_prompt = user_prompt + "\n\n[수정 지시 — 아래 문제를 모두 고쳐 다시 작성하라]\n- " + "\n- ".join(issues)
        try:
            repaired = call_llm(FINAL_ANSWER_SYSTEM_PROMPT, repair_prompt, tier="final").strip()
            if repaired:
                issues2 = _final_answer_quality_feedback(ctx, repaired) + _number_guard(repaired, allowed_numbers)
                if len(issues2) <= len(issues):
                    answer, issues = repaired, issues2
        except Exception:
            pass
    return answer, issues

def final_answer_node(state: ManufacturingState) -> dict:
    # Intake Gate 차단 시: 차단 메시지를 그대로 최종 답변으로 반환
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

    # ===== facts sheet → LLM 해설 합성(+가드) → 검증 실패 시 결정적 폴백 =====
    ctx = build_answer_context(state)
    allowed_numbers = _allowed_numbers(ctx)
    body, issues = _synthesize_answer(ctx, allowed_numbers)
    used_fallback = False
    # 본문이 비었거나(LLM 불가) 숫자 hallucination이 남으면 결정적 폴백으로 안전하게 대체한다.
    if (not body) or any(i.startswith("number_hallucination") for i in issues):
        body = _fallback_final_answer(ctx)
        used_fallback = True
        if issues:
            warnings.append("final_answer_fallback: " + "; ".join(i.split(":")[0] for i in issues)[:200])

    # ===== 후처리: 입력부족 정합 → 정확 수치 블록 보장 → 현지화/정리 → 안전 트레일러 → 종합 판단 배너 → [출처] =====
    body = _remove_false_missing_input_section(body, missing)
    body = _ensure_missing_input_visible(body, missing)
    body = _ensure_diagnosis_block(body, pred)   # 고장 종류별 근거/체크리스트를 정확 수치로 보장
    _has_risk = bool(pred and pred.risk_flags and any(str(r.get("level", "")).lower() in {"high", "medium"} for r in pred.risk_flags))
    body = _ensure_safety_trailer(body, _has_risk)

    answer = _verdict_banner(pred, sql, ev) + "\n\n" + body
    answer = _clean_final_answer_format(_localize_answer_terms(answer))
    answer = _ensure_citations_visible(answer, citations)

    fa = FinalAnswer(answer=answer, citations=citations, warnings=warnings, missing_inputs=missing)
    updates = _mark_final_task_pass(state)
    updates["final_answer"] = fa
    return updates
print("final_answer_node hybrid(LLM 해설 + 결정적 수치 보장 + 폴백) 정의 완료")
