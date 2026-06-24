from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import PredictionResult
from manufacturing_agent.rag.pinecone_store import vector_search
#from manufacturing_agent.rag.chroma import vector_search
# Retrieval Layer 전용 rule-based helper (LLM prompt 아님).
# AI4I 변수/고장 모드를 Haas 문서 표현으로 확장하고 우선 검색 문서를 라우팅한다.
from manufacturing_agent.services.manufacturing_taxonomy import (
    build_rag_queries,
    build_search_terms,
    route_documents,
    get_failure_match,
    SYMPTOM_ALIASES,
)

# ---------- services/rag_service.py ----------
# profile -> ChromaDB type 필터 (Haas 문서는 모두 troubleshooting)
## 프로파일이 정의되어야하는 이유 : 각 프로파일에 따라 다른 검색 전략과 문서 필터링을 적용하기 위해
RETRIEVAL_PROFILES = {
    "troubleshooting_rag": "troubleshooting", # mode A: 단순 검색
    "prediction_plus_rag": "troubleshooting", # mode B: 예측 기반 검색
    "fallback_broad": None,                    # 재검색은 type filter 없이 범위 확대
}
HAAS_SOURCE_PREFIXES = ("haas/", "document/haas/")
SOURCE_PREFIX_POLICY = {
    "troubleshooting_rag": HAAS_SOURCE_PREFIXES,
    "prediction_plus_rag": HAAS_SOURCE_PREFIXES,
    "fallback_broad": None,
}

def _source_allowed(source: str, profile: str) -> bool:
    prefixes = SOURCE_PREFIX_POLICY.get(profile)
    if prefixes is None:
        return True
    return (source or "").startswith(prefixes)


def _profile_type_filter(profile: str) -> Optional[str]:
    # OpenAI collection은 type=troubleshooting으로 저장되어 있고,
    # local hash collection은 type=html로 저장되어 있어 type filter를 걸면 전부 탈락한다.
    return RETRIEVAL_PROFILES.get(profile, "troubleshooting") if USE_OPENAI_EMBEDDINGS else None


# ----- routing/retrieval 디버그 로그 -----
import sys as _sys


def _rag_log(event: str, **fields: Any) -> None:
    """RAG_DEBUG가 켜져 있을 때만 routing/retrieval 디버그를 stderr로 남긴다."""
    if not RAG_DEBUG:
        return
    line = " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[RAG {event}] {line}", file=_sys.stderr, flush=True)


def _chunk_ref(h: dict) -> dict:
    """검색 chunk의 식별/위치/score 요약 (debug + dedup 공용)."""
    return {
        "id": h.get("id"),
        "source": h.get("source"),
        "page": h.get("page"),            # PDF page (현재 메타에 없으면 None)
        "chunk_index": h.get("chunk_index"),
        "score": round(float(h.get("score", 0.0)), 4),
    }


def _dedup_key(h: dict) -> tuple:
    """doc_id + source_path + page + chunk_id 복합 중복 제거 키."""
    return (h.get("id"), h.get("source"), h.get("page"), h.get("chunk_index"))


# ----- taxonomy 연동 레이어 -----
# taxonomy의 문서 프로필 키 -> chroma `source` 경로 매칭 토큰.
# (예: "mill_spindle" -> "Mill Spindle" 토큰이 source 경로에 모두 포함되면 해당 문서로 인정)
# 새 제조사 문서가 늘면 taxonomy의 DOC_PROFILES와 이 매핑만 확장하면 된다.
PROFILE_KEY_TO_DOCNAME = {
    "mill_spindle": "Mill Spindle",
    "mill_chatter": "Mill Chatter",
    "mechanical_service": "Mechanical Service",
}

# 증상 alias -> 증상 key 역색인 (자연어 질문에서 증상 추출용, 1회 구성)
_SYMPTOM_LOOKUP: list[tuple[str, str]] = [
    (alias.strip().lower(), key)
    for key, aliases in SYMPTOM_ALIASES.items()
    for alias in aliases
    if alias.strip()
]


def _has_cjk(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _extract_symptoms(question: str) -> list[str]:
    """사용자 자연어 질문에서 taxonomy 증상 key를 추출한다(rule-based).

    - 한글 alias(과열/진동/채터 등 2글자 포함): substring 매칭
    - 영어 alias: 단어경계 매칭 + 3글자 이상 (load→download 같은 과매칭 방지)
    """
    q = (question or "").lower()
    found: list[str] = []
    for alias, key in _SYMPTOM_LOOKUP:
        if key in found:
            continue
        if _has_cjk(alias):
            if len(alias) >= 2 and alias in q:
                found.append(key)
        elif len(alias) >= 3 and re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", q):
            found.append(key)
    return found


#(1) Query Builder------------------------------
def build_query(question: str, profile: str, prediction: Optional[PredictionResult] = None) -> dict:
    """
    Query Builder: 사용자 질문 + Prediction 결과 + 증상을 manufacturing_taxonomy로 확장해
    RAG 검색 계획(Search Plan)을 생성한다.

    Mode A (단순/증상 기반 검색)
        - prediction 정보가 없거나 profile이 prediction_plus_rag가 아닌 경우.
        - 질문에서 추출한 증상(symptoms)만으로 taxonomy 확장/라우팅을 수행한다.

    Mode B (예측 기반 검색)
        - PredictionResult.failure_types(AI4I 코드)와 cause_features(원인 변수)를
          taxonomy alias로 확장하고, priority 문서를 라우팅한다.

    AI4I 변수명을 그대로 검색하지 않고 Haas 문서 표현(tool load, cutting force,
    spindle temperature 등)으로 확장해 query fan-out을 만든다.

    Returns:
        Search Plan(dict). Evidence Agent가 의존하는 키(mode/profile/user_query/
        search_query/tags/doc_whitelist/failure_types/failure_ko)를 그대로 유지하고,
        Retriever 내부 전용 키(queries/symptoms/priority_doc_names)를 추가한다.
    """
    has_pred = bool(prediction and prediction.failure_types)
    symptoms = _extract_symptoms(question)

    if profile != "prediction_plus_rag" or not has_pred:
        mode = "A"
        failure_codes: list[str] = []
        features: list[str] = []
    else:
        mode = "B"
        failure_codes = list(prediction.failure_types)
        features = list(prediction.cause_features or [])

    # taxonomy 기반 확장
    terms = build_search_terms(features, failure_codes, symptoms)          # Haas 표현 검색어
    queries = build_rag_queries(question, features, failure_codes,         # query fan-out
                                symptoms, max_queries=8)
    priority_keys = route_documents(features, failure_codes, symptoms)     # 우선 검색 문서(프로필 키)
    priority_names = [PROFILE_KEY_TO_DOCNAME[k] for k in priority_keys
                      if k in PROFILE_KEY_TO_DOCNAME]
    failure_ko = [m.ko for m in (get_failure_match(c) for c in failure_codes) if m]

    # EvidenceArtifact 표시용 단일 대표 query (질문 + 확장 검색어 상위 일부)
    search_query = " ".join([question.strip(), *terms[:8]]).strip()

    return {
        "mode": mode,
        "profile": profile,
        "user_query": question,
        "search_query": search_query,
        "tags": terms,
        "doc_whitelist": priority_names or None,
        "failure_types": failure_codes,
        "failure_ko": failure_ko,
        # --- Retriever 내부 전용 (Evidence Agent 인터페이스 아님) ---
        "queries": queries,
        "features": features,
        "symptoms": symptoms,
        "priority_doc_names": priority_names,
    }


def _doc_name_matches(source: str, doc_name: str) -> bool:
    """
    화이트리스트 문서명과 실제 source 경로가 일치하는지 확인한다.

    문서명을 공백 기준으로 분리한 뒤,
    모든 토큰이 source 경로에 포함되는지 검사한다.

    Args:
        source:
            검색 결과의 source 경로.

        doc_name:
            화이트리스트에 등록된 문서명.

    Returns:
        True이면 해당 문서로 인정,
        False이면 제외한다.
    """
    s = (source or "").lower()
    return all(tok.lower() in s for tok in doc_name.split())

def _fanout_queries(plan: dict) -> list[str]:
    """build_rag_queries 결과에서 실제 vector search에 쓸 query만 추린다.
    (taxonomy가 붙이는 'preferred_docs: ...' 힌트 라인은 검색어가 아니므로 제외)"""
    raw = plan.get("queries") or [plan.get("search_query", "")]
    out, seen = [], set()
    for q in raw:
        q = (q or "").strip()
        if not q or q.lower().startswith("preferred_docs:"):
            continue
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out or [plan.get("search_query", "")]


#(2) Retriever------------------------------
def retrieve_stage(plan: dict, k: int = 16, top_k: int = 4, debug: Optional[dict] = None) -> list[dict]:
    """
    Retriever (Query Fan-out + Priority Search + Merge + Fallback).
    vector_search 백엔드(Pinecone/Chroma)에서 fan-out query로 문서를 검색한다.

    수행 과정
        1. taxonomy fan-out query 각각에 대해 Vector Search (profile type filter 적용)
        2. doc_id+source+page+chunk_id 복합키 Merge (중복 시 최고 score 유지)
        3. Retrieval Profile별 source policy 적용 (troubleshooting/prediction은 haas-only)
        4. priority_docs 우선: priority 후보가 충분(>= top_k)하면 그것만 사용,
           부족하면 전체 haas 문서로 fallback (기존 RAG가 깨지지 않도록 항상 fallback 유지)

    Args:
        plan: build_query()가 생성한 Search Plan.
        k:    query당 Vector Search 후보 개수.
        top_k: 최종 근거 개수(우선검색 충분 여부 판정 기준).
        debug: 채워줄 디버그 dict(있으면 retrieved/fallback_used 기록).

    Returns:
        merge/우선순위가 적용된 문서 후보 리스트.
    """
    profile = plan["profile"]
    type_filter = _profile_type_filter(profile)
    queries = _fanout_queries(plan)
    _rag_log("route",
             features=plan.get("features"), failure_types=plan.get("failure_types"),
             symptoms=plan.get("symptoms"), priority_docs=plan.get("priority_doc_names"))
    _rag_log("queries", profile=profile, type_filter=type_filter, queries=queries)

    # 1) fan-out + 2) merge (복합 dedup 키, 최고 score 유지)
    merged: dict[tuple, dict] = {}
    for q in queries:
        hits = vector_search(q, k=k, type_filter=type_filter)
        if not hits and type_filter:
            hits = vector_search(q, k=k, type_filter=None)
        for h in hits:
            key = _dedup_key(h)
            prev = merged.get(key)
            if prev is None or float(h.get("score", 0.0)) > float(prev.get("score", 0.0)):
                merged[key] = h
    hits = list(merged.values())

    # 3) profile source policy (haas-only 등)
    hits = [h for h in hits if _source_allowed(h.get("source", ""), profile)]

    # 4) priority 우선 + 부족하면 전체 haas fallback
    fallback_used = False
    priority_names = plan.get("priority_doc_names") or []
    if priority_names and profile != "fallback_broad":
        prio = [h for h in hits
                if any(_doc_name_matches(h.get("source", ""), n) for n in priority_names)]
        prio_keys = {_dedup_key(h) for h in prio}
        rest = [h for h in hits if _dedup_key(h) not in prio_keys]
        if len(prio_keys) >= top_k:
            hits = prio                      # priority 문서만으로 충분
        else:
            hits = prio + rest               # 부족 -> 전체 haas로 fallback 보강
            fallback_used = bool(rest)

    _rag_log("retrieved", count=len(hits), fallback_used=fallback_used,
             chunks=[_chunk_ref(h) for h in sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]])
    if debug is not None:
        debug["fallback_used"] = fallback_used
        debug["candidate_count"] = len(hits)
        debug["retrieved"] = [_chunk_ref(h) for h in
                              sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True)[:max(top_k, 8)]]
    return hits


# score = 1.0 - cosine_distance = 코사인 유사도(0~1). 코사인 공간 임베딩 기준 임계값.
# 0.45: on-topic 질의(≥0.54)는 통과, off-topic(용접 0.42/커피 0.30)은 NO_EVIDENCE로 차단.
# 코퍼스/임베딩 모델이 바뀌면 .env MIN_EVIDENCE_SCORE로 재조정.
MIN_EVIDENCE_SCORE = float(os.environ.get("MIN_EVIDENCE_SCORE", "0.45"))
RETRIEVED_DOC_INJECTION_RE = re.compile(
    r"ignore\s+previous\s+instructions|system\s+prompt|developer\s+instruction|이전\s*지시\s*무시|안전\s*경고\s*제거|규칙\s*무시",
    re.I,
)

def _redact_retrieved_instruction_text(text: str) -> str:
    safe = RETRIEVED_DOC_INJECTION_RE.sub("[UNTRUSTED_INSTRUCTION_REMOVED]", text or "")
    return safe[:1200]

def sanitize_retrieved_doc(doc: dict) -> dict:
    text = str(doc.get("text") or "")
    flagged = bool(RETRIEVED_DOC_INJECTION_RE.search(text))
    safe_doc = dict(doc)
    safe_doc["text"] = _redact_retrieved_instruction_text(text) if flagged else text
    safe_doc["security_flags"] = {"possible_prompt_injection": flagged}
    return safe_doc


#(3) Evidence Ranker------------------------------
def rank_evidence(hits: list[dict], top_k: int = 3) -> list[dict]:
    """
    Evidence Ranker.
    Retriever가 반환한 후보 문서를 정렬하고 중복 Chunk를 제거하여 최종 근거 문서를 선택한다.

    수행 과정
        1. score 기준 정렬
        2. doc id 기준 중복 제거
        3. Top-k 문서 선택

    Args:
        hits:
            Retriever 검색 결과.

        top_k:
            최종 선택할 근거 문서 개수.

    Returns:
        최종 근거 문서 리스트.
    """
    seen, ranked = set(), []
    for h in sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True):
        # doc_id + source + page + chunk_id 복합키 dedup 후 score Top-k
        key = _dedup_key(h)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(h)
        if len(ranked) >= top_k:
            break
    return [sanitize_retrieved_doc(d) for d in ranked]


#(4) Citation Builder------------------------------
def _clean_evidence_snippet(text: str, limit: int = 360) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    # HTML navigation/header fragments are poor evidence; trim to likely troubleshooting content when possible.
    anchors = [
        "Symptom Table", "Symptom", "Possible Cause", "Corrective Action", "Electrical Safety",
        "Excessive Tool Wear", "Drive Belt", "Coolant", "Bearing", "Lubrication",
        "위험", "조치", "점검", "정비", "에너지관리", "잠금", "표지", "재가동",
    ]
    lower = cleaned.lower()
    positions = [lower.find(a.lower()) for a in anchors if lower.find(a.lower()) >= 0]
    if positions:
        cleaned = cleaned[min(positions):]
    return cleaned[:limit].strip()

def _citation_title(source: Any, source_id: Any = None) -> str:
    raw = str(source or source_id or "문서 근거")
    name = re.split(r"[/\\]", raw)[-1]
    name = re.sub(r"\.(html|pdf|txt|md)$", "", name, flags=re.I)
    return name.replace("_", " ").strip() or "문서 근거"

def build_citations(docs: list[dict]) -> list[dict]:
    """최종 선택 문서를 citation metadata로 변환한다."""
    citations = []
    for idx, d in enumerate(docs, start=1):
        snippet = _clean_evidence_snippet(str(d.get("text") or ""), limit=420)
        citations.append({
            "citation_id": f"C{idx}",
            "source_id": d.get("id"),
            "source": d.get("source"),
            "title": _citation_title(d.get("source"), d.get("id")),
            "type": d.get("type"),
            "chunk_index": d.get("chunk_index"),
            "snippet": snippet,
            "score": round(float(d.get("score", 0)), 3),
            "security_flags": d.get("security_flags", {"possible_prompt_injection": False}),
        })
    return citations

def build_citation_aware_docs(docs: list[dict], citations: list[dict]) -> list[dict]:
    items = []
    for idx, doc in enumerate(docs):
        c = citations[idx] if idx < len(citations) else {}
        items.append({
            "citation_id": c.get("citation_id", f"C{idx + 1}"),
            "title": c.get("title") or _citation_title(doc.get("source"), doc.get("id")),
            "source": c.get("source") or doc.get("source"),
            "chunk_index": c.get("chunk_index", doc.get("chunk_index")),
            "score": c.get("score", round(float(doc.get("score", 0)), 3)),
            "snippet": c.get("snippet") or _clean_evidence_snippet(str(doc.get("text") or "")),
            "text": str(doc.get("text") or "")[:1800],
            "security_flags": doc.get("security_flags", {"possible_prompt_injection": False}),
        })
    return items


#----------------- RAG Search Pipeline (Entry Point) ------------------------------
def rag_search(question: str, profile: str, prediction: Optional[PredictionResult] = None,
               retrieve_k: int = 16, top_k: int = 4) -> dict:
    """
    RAG Search Pipeline.

    Evidence Agent가 호출하는 RAG 서비스의 진입점이다.

    내부 수행 순서
        1. Query Builder
        2. Retriever
        3. Evidence Ranker
        4. Citation Builder

    Note:
        문서 요약 및 자연어 답변 생성은 수행하지 않는다.
        Evidence Agent가 반환된 documents와 citations를
        이용하여 최종 답변을 생성한다.

    Args:
        question:
            사용자 질문.

        profile:
            Retrieval Profile.

        prediction:
            Prediction Agent 결과.
            Mode B에서만 사용된다.

        retrieve_k:
            Retriever 후보 문서 개수.

        top_k:
            최종 근거 문서 개수.

    Returns:
        {
            "plan": Search Plan,
            "documents": Ranked Documents,
            "citations": Citation List,
            "status": "OK" | "NO_EVIDENCE",
            "limitations": [...],
            "guidance": <NO_EVIDENCE일 때 담당자 안내 문구>,
            "debug": {features, failure_types, symptoms, priority_docs, queries,
                      retrieved[], fallback_used, status}
        }

    NO_EVIDENCE 정책:
        - top_k 결과가 없거나, score가 threshold(MIN_EVIDENCE_SCORE) 미만이면
          documents/citations를 비우고 status=NO_EVIDENCE로 반환한다.
        - 이렇게 하면 Evidence Agent가 LLM 요약(추측)을 호출하지 않고,
          사용자에게는 담당자 확인 안내를 노출한다.
    """
    plan = build_query(question, profile, prediction)   # (1)
    debug: dict[str, Any] = {
        "features": plan.get("features"),
        "failure_types": plan.get("failure_types"),
        "symptoms": plan.get("symptoms"),
        "priority_docs": plan.get("priority_doc_names"),
        "queries": _fanout_queries(plan),
    }
    hits = retrieve_stage(plan, k=retrieve_k, top_k=top_k, debug=debug)  # (2) fan-out + priority + fallback
    ranked = rank_evidence(hits, top_k=top_k)            # (3)
    relevant = [d for d in ranked if float(d.get("score", 0.0)) >= MIN_EVIDENCE_SCORE]

    if not relevant:
        reason = ("검색된 문서가 없습니다." if not ranked
                  else f"검색된 문서 score가 threshold({MIN_EVIDENCE_SCORE}) 미만입니다.")
        guidance = (
            "관련 문서 근거를 찾지 못했습니다. 추측으로 답변하지 않으며, "
            f"정확한 점검·조치는 {support_contact_text()} 확인이 필요합니다."
        )
        debug["status"] = "NO_EVIDENCE"
        _rag_log("no_evidence", reason=reason, ranked=len(ranked))
        return {
            "plan": plan, "documents": [], "citations": [],
            "status": "NO_EVIDENCE", "limitations": [reason],
            "guidance": guidance, "debug": debug,
        }

    debug["status"] = "OK"
    return {
        "plan": plan, "documents": relevant, "citations": build_citations(relevant),
        "status": "OK", "limitations": [], "debug": debug,
    }


