from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import EvidenceArtifact, FinalAnswer, PredictionResult, SQLHistoryArtifact
from manufacturing_agent.contracts.state import ManufacturingState

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

def _citation_display_name(citation: dict) -> str:
    import unicodedata
    raw = str(citation.get("title") or citation.get("source") or citation.get("source_id") or "문서 근거")
    name = raw.split("/")[-1]
    name = re.sub(r"_\d+$", "", name)
    name = re.sub(r"\.(html?|md|pdf)$", "", name, flags=re.I)
    name = unicodedata.normalize("NFC", name).replace("_", " ").strip() or "문서 근거"
    return name[:90].rstrip() + ("..." if len(name) > 90 else "")

def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["[출처]"]
    for idx, c in enumerate(citations[:6], start=1):
        cid = c.get("citation_id") or f"C{idx}"
        title = _citation_display_name(c)
        source = str(c.get("source") or c.get("source_id") or "").strip()
        chunk = c.get("chunk_index")
        lines.append(f"- [{cid}] 문서: {title}")
        if source:
            lines.append(f"  - 원본: {source}")
        if chunk is not None:
            lines.append(f"  - 위치: chunk={chunk}")
        snippet = re.sub(r"\s+", " ", str(c.get("snippet") or "")).strip()
        if snippet:
            ascii_ratio = sum(ch.isascii() for ch in snippet) / max(len(snippet), 1)
            space_ratio = snippet.count(" ") / max(len(snippet), 1)
            # 영어 raw HTML/공백 손상 PDF 추출물은 비개발자에게 안 읽히므로 가독성 있을 때만 노출
            if ascii_ratio < 0.5 and space_ratio > 0.03:
                if len(snippet) > 180:
                    snippet = snippet[:180].rstrip() + "…"
                lines.append(f"  - 원문 근거: {snippet}")
    return "\n".join(lines)

def _format_counter(counter: Counter, unit: str = "건", limit: int = 5) -> str:
    if not counter:
        return "확인 필요"
    return ", ".join(f"{name} {count}{unit}" for name, count in counter.most_common(limit))

def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _sample_failure_rows(rows: list[dict]) -> list[str]:
    out = []
    for r in rows[:3]:
        dt = r.get("downtime_min")
        dt_s = f" · {dt}분" if str(dt or "").strip() not in ("", "None") else ""
        date = str(r.get("event_date") or "")[5:] or str(r.get("event_date") or "")
        sym = str(r.get("symptom") or r.get("corrective_action") or "").strip()
        out.append(f"{date} {_short_failure(r.get('failure_type'))} {_label_component(r.get('component'))}{dt_s} — {sym}")
    return out

def _history_result_summary(query_type: Optional[str], rows: list[dict], status: str) -> str:
    qtype = query_type or "history"
    if not rows:
        return "• 조건에 맞는 이력 없음"
    if qtype in {"similar_incidents", "failure_history"}:
        failure_types = Counter(_short_failure(r.get("failure_type")) for r in rows if r.get("failure_type"))
        components = Counter(_label_component(r.get("component")) for r in rows if r.get("component"))
        actions = []
        for row in rows:
            a = str(row.get("corrective_action") or "").strip()
            if a and a not in actions:
                actions.append(a)
            if len(actions) >= 3:
                break
        downtimes = [_to_int(r.get("downtime_min"), 0) for r in rows if str(r.get("downtime_min") or "").strip() not in ("", "None")]
        head = f"총 {len(rows)}건"
        if downtimes:
            head += f" · 다운타임 {sum(downtimes)}분(평균 {round(sum(downtimes) / len(downtimes))}분)"
        lines = [head,
                 f"• 유형: {_format_counter(failure_types)}",
                 f"• 영역: {_format_counter(components)}"]
        samples = _sample_failure_rows(rows[:3])
        if samples:
            lines.append("• 대표 사례:")
            lines.extend(f"   - {s}" for s in samples)
        if actions:
            lines.append("• 대표 조치: " + " · ".join(actions))
        preventions = []
        for row in rows:
            p = str(row.get("preventive_action") or "").strip()
            if p and p not in preventions:
                preventions.append(p)
            if len(preventions) >= 3:
                break
        if preventions:
            lines.append("• 재발 방지: " + " · ".join(preventions))
        return "\n".join(lines)
    if qtype == "corrective_actions":
        items = []
        for row in rows:
            it = f"{_short_failure(row.get('failure_type'))}: {row.get('corrective_action')} (예방: {row.get('preventive_action')})"
            if it not in items:
                items.append(it)
            if len(items) >= 4:
                break
        return "유형별 대응 방식\n" + "\n".join(f"• {it}" for it in items)
    if qtype == "repeated_patterns":
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            ft = _short_failure(row.get("failure_type"))
            slot = grouped.setdefault(ft, {"cases": 0, "downtime": 0, "components": Counter()})
            slot["cases"] += _to_int(row.get("case_count"), 1)
            slot["downtime"] += _to_int(row.get("total_downtime_min"), 0)
            if row.get("component"):
                slot["components"][_label_component(row.get("component"))] += _to_int(row.get("case_count"), 1)
        patterns = []
        for ft, data in sorted(grouped.items(), key=lambda kv: (-kv[1]["cases"], -kv[1]["downtime"], kv[0]))[:5]:
            patterns.append(f"{ft}: {data['cases']}건 · 다운타임 {data['downtime']}분 · 주요 영역 {_format_counter(data['components'], limit=2)}")
        return "반복 패턴\n" + "\n".join(f"• {p}" for p in patterns) if patterns else f"• {len(rows)}건 조회됨"
    return f"• {len(rows)}건 조회됨"

def _history_summary_for_answer(sql: Optional[SQLHistoryArtifact]) -> str:
    if not sql:
        return "확인된 최근 이력 없음"
    if sql.status == "INVALID_REQUEST":
        return sql.summary or "이력 조회 조건이 부족함"
    if sql.status == "EMPTY":
        return "조건에 맞는 과거 이력은 조회되지 않음"
    if sql.status in {"BLOCKED", "FAIL"}:
        return "이력 조회 실패 또는 정책 차단: " + (sql.error_message or sql.summary or "확인 필요")
    lines = []
    results = getattr(sql, "results", []) or []
    if results:
        by_type = {}
        for r in results:
            by_type.setdefault(r.query_type, r)
        similar = by_type.get("similar_incidents")
        history = by_type.get("failure_history")
        if similar:
            lines.append(_history_result_summary(similar.query_type, similar.rows or [], similar.status))
            if history and history.rows:
                lines.append(f"(전체 이력 기준 {len(history.rows)}건 포함 — 특정 유형 외 이력은 생략)")
        elif history:
            lines.append(_history_result_summary(history.query_type, history.rows or [], history.status))
        if "repeated_patterns" in by_type:
            rp = by_type["repeated_patterns"]
            lines.append(_history_result_summary("repeated_patterns", rp.rows or [], rp.status))
        if not similar and not history and "corrective_actions" in by_type:
            ca = by_type["corrective_actions"]
            lines.append(_history_result_summary("corrective_actions", ca.rows or [], ca.status))
    elif sql.rows:
        lines.append(_history_result_summary(sql.query_type, sql.rows, sql.status))
    else:
        lines.append(sql.summary or "조건에 맞는 이력 없음")
    if sql.limitations:
        lines.append("조회 한계: " + "; ".join(sql.limitations[:3]))
    return "\n".join(lines)

_RISK_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}

def _render_diagnosis_block(pred: Optional[PredictionResult]) -> str:
    """고장 종류별 근거(규칙/계산/영향 변수)를 결정적으로 렌더한다. 정상(낮음)은 한 줄로 접는다."""
    if not pred or not pred.risk_flags:
        return ""
    lines = ["[고장 종류별 근거]"]
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
    return "[지금 점검할 일]\n" + "\n".join(f"{i}. {it}" for i, it in enumerate(items[:4], 1))

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
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s+", "", answer or "")
    cleaned = cleaned.replace("시정 조치으로", "시정 조치로")
    cleaned = cleaned.replace("예방 조치으로", "예방 조치로")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

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
    return ("[추가로 필요한 입력]\n"
            f"현재 입력만으로는 정확한 진단이 어렵습니다. 다음 값을 입력해 주세요: {miss}")

def _history_block(sql) -> str:
    if sql is None:
        return ""
    body = _history_summary_for_answer(sql)
    if not body:
        return ""
    return "[과거 고장 이력]\n" + body

def _evidence_block(ev) -> str:
    if not ev:
        return ""
    status = getattr(ev, "status", None)
    if status == "OK" and ev.evidence_summary:
        _parts = [p.strip() for p in ev.evidence_summary.split("\n") if p.strip()]
        _cited = [p for p in _parts if "[C" in p][:4]
        return "[문서 근거]\n" + ("\n".join(_cited) if _cited else ev.evidence_summary[:600])
    if status == "LOW_RELEVANCE":
        body = ev.evidence_summary or "검색된 문서의 관련성이 낮아 단정하기 어렵습니다."
        return "[문서 근거]\n" + body + "\n(관련성이 낮아 참고용입니다. 추가 문서 확인이 필요합니다.)"
    if status == "EMPTY":
        return "[문서 근거]\n현재 검색된 문서 근거만으로는 단정하기 어렵습니다."
    return ""

def _headline_fallback(pred, sql, ev) -> str:
    if pred and pred.risk_flags:
        top = pred.risk_flags[0]
        ft = _short_failure(top.get("failure_type"))
        lv = _risk_level_ko(top.get("level"))
        return f"{ft} 위험이 {lv}으로 감지됐습니다. 아래 진단 결과를 확인하고 점검하세요."
    if pred and getattr(pred, "status", None) == "NEEDS_INPUT":
        return "정확한 진단을 위해 추가 입력이 필요합니다. 아래 안내를 확인하세요."
    if sql is not None:
        return "과거 고장 이력을 조회했습니다. 아래 결과를 참고하세요."
    if ev is not None:
        return "관련 문서 근거를 검색했습니다. 아래 내용을 참고하세요."
    return ""

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
        top_fact = (
            f"진단: {_short_failure(top.get('failure_type'))} 위험 {_risk_level_ko(top.get('level'))}"
            + (f" — {top.get('formula')}" if top.get("formula") else "")
        )
        facts.append(top_fact)
        others = [
            _short_failure(r.get("failure_type"))
            for r in pred.risk_flags[1:]
            if str(r.get("level", "")).lower() in {"high", "medium"}
        ]
        if others:
            facts.append("추가 위험: " + " · ".join(others[:2]))
    elif pred and getattr(pred, "status", None) == "NEEDS_INPUT":
        facts.append("진단: 입력 부족으로 추가 데이터 필요")
    if sql is not None:
        facts.append("과거 고장 이력 조회됨")
    if ev is not None and getattr(ev, "status", None) in {"OK", "LOW_RELEVANCE"}:
        facts.append("관련 문서 근거 있음")
    try:
        out = call_llm(HEADLINE_SYS, json.dumps({"질문": user_q, "요지": facts}, ensure_ascii=False), tier="default").strip()
        return out.split("\n")[0].strip() if out else _headline_fallback(pred, sql, ev)
    except Exception:
        return _headline_fallback(pred, sql, ev)

def final_answer_node(state: ManufacturingState) -> dict:
    dec = state.get("input_decision")
    if dec and dec.blocked:
        return {"final_answer": FinalAnswer(answer=dec.block_message or "요청을 처리할 수 없습니다.")}

    pred = state.get("prediction_result")
    ev = state.get("evidence_bundle")
    sql = state.get("sql_result")
    packet = state.get("context_packet")

    warnings: list[str] = []

    # context 수준 경고 (멀티턴 stale 값 사용 등)
    if packet and packet.context_warnings:
        warnings.extend(packet.context_warnings)

    # intake 출력 제약 (안전 관련 응답 지침)
    intake = state.get("intake_decision")
    if intake and intake.output_constraints:
        warnings.extend(intake.output_constraints)

    # artifact별 사용자에게 의미 있는 경고만 선별 (limitations 원문은 노출하지 않음)
    if pred and pred.status == "PARTIAL" and pred.missing_features:
        missing_ko = ", ".join(_label_feature(f) for f in pred.missing_features)
        warnings.append(f"일부 입력값 부족으로 전체 진단이 제한됩니다: {missing_ko}")
    if ev and ev.status == "LOW_RELEVANCE":
        warnings.append("검색된 문서 근거의 관련성이 낮아 참고 수준으로만 제공됩니다.")
    if ev and ev.status == "FAIL":
        warnings.append("문서 근거 검색에 실패했습니다.")
    if sql and sql.status in {"BLOCKED", "FAIL"}:
        warnings.append("고장 이력 조회에 실패했습니다.")

    warnings = list(dict.fromkeys(warnings))  # 중복 제거 (순서 유지)
    missing = pred.missing_features if (pred and pred.status == "NEEDS_INPUT") else []
    citations = ev.citations if ev and ev.status in {"OK", "LOW_RELEVANCE"} else []

    sections = [_verdict_banner(pred, sql, ev)]
    headline = _headline(pred, sql, ev, state.get("user_message", ""))
    if headline:
        sections.append(headline)
    for block in (
        _missing_block(pred),
        _render_diagnosis_block(pred),
        _history_block(sql),
        _evidence_block(ev),
        _render_checklist(pred),
        _format_citations(citations),
    ):
        if block:
            sections.append(block)
    _has_risk = bool(pred and pred.risk_flags and any(str(r.get("level", "")).lower() in {"high", "medium"} for r in pred.risk_flags))
    if _has_risk:
        sections.append("⚠️ 규칙 기반 보조 진단이며, 정지·재가동·정비 승인 여부는 현장 안전 책임자와 설비 담당자가 판단해야 합니다.")
    else:
        sections.append("ℹ️ 보조 진단·조회 결과이며, 실제 조치는 현장 담당자 확인이 필요합니다.")

    answer = _clean_final_answer_format(_localize_answer_terms("\n\n".join(s for s in sections if s)))
    fa = FinalAnswer(answer=answer, citations=citations, warnings=warnings, missing_inputs=missing)
    updates = _mark_final_task_pass(state)
    updates["final_answer"] = fa
    return updates
