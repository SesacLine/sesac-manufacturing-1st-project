from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import PredictionResult
from manufacturing_agent.rag.pinecone_store import vector_search

# ---------- services/rag_service.py ----------
# profile -> ChromaDB type 필터 (Haas 문서는 모두 troubleshooting)
## 프로파일이 정의되어야하는 이유 : 각 프로파일에 따라 다른 검색 전략과 문서 필터링을 적용하기 위해
RETRIEVAL_PROFILES = {
    "troubleshooting_rag": "troubleshooting", # mode A: 단순 검색
    "prediction_plus_rag": "troubleshooting", # mode B: 예측 기반 검색
    "safety_procedure_rag": None,              # 안전/LOTO/재가동은 safety/html 문서를 함께 검색
    "fallback_broad": None,                    # 재검색은 type filter 없이 범위 확대
}
HAAS_SOURCE_PREFIXES = ("haas/", "document/haas/")
SOURCE_PREFIX_POLICY = {
    "troubleshooting_rag": HAAS_SOURCE_PREFIXES,
    "prediction_plus_rag": HAAS_SOURCE_PREFIXES,
    "safety_procedure_rag": ("osha/", "kosha/", "haas/", "document/osha/", "document/kosha/", "document/haas/"),
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

# ----- 매핑 레이어 -----
# 고장 유형(한글) -> 검색 태그 + 문서 화이트리스트
FAILURE_RAG_MAP = {
    "공구마모고장": {"query_tags": ["tool wear", "chatter", "surface finish", "cutting load",
                              "tool condition", "tool vibration"],
                "documents": ["Mill Chatter", "Mill Spindle"]},
    "열방출실패": {"query_tags": ["overheating", "spindle temperature", "lubrication", "cooling",
                             "air pressure", "thermal issue"],
               "documents": ["Mill Spindle", "Vector Drive"]},
    "과부하파손": {"query_tags": ["overload", "high torque", "cutting force", "spindle load",
                             "chatter", "rpm", "feed speed"],
               "documents": ["Mill Chatter", "Mill Spindle", "Vector Drive"]},
    "전력실패": {"query_tags": ["vector drive", "DC bus", "input voltage", "regen",
                            "electrical failure", "spindle motor", "drive alarm"],
              "documents": ["Vector Drive NGC", "Vector Drive CHC"]},
    "무작위오류": {"query_tags": ["unknown failure", "alarm code", "symptom clarification"],
               "documents": []},
}

#질의에 아래 키워드가 포함되면 이 태그를 추가해 검색 범위를 넓힌다
VARIABLE_TAG_MAP = {
    "기온": ["ambient temperature", "cooling", "overheating"],                 
    "공정온도": ["process temperature", "spindle overheating", "thermal issue", "lubrication"],
    "회전속도": ["rpm", "spindle speed", "chatter", "vibration"],
    "토크": ["torque", "high load", "overload", "cutting force", "spindle load"],
    "공구마모": ["tool wear", "tool condition", "surface finish", "cutting load", "chatter"],
}
# PredictionResult.failure_types의 AI4I 코드 <-> 매핑 테이블(한글 키) 브리지
FAILURE_CODE_TO_KO = {"TWF": "공구마모고장", "HDF": "열방출실패",
                      "OSF": "과부하파손", "PWF": "전력실패", "RNF": "무작위오류"}
FEATURE_TO_KO = {"air_temperature": "기온", "process_temperature": "공정온도",
                 "rotational_speed": "회전속도", "torque": "토크", "tool_wear": "공구마모"}


#(1) Query Builder------------------------------
def build_query(question: str, profile: str, prediction: Optional[PredictionResult] = None) -> dict:
    """
    Query Builder: 사용자 질문과 Prediction 결과를 기반으로 RAG 검색 계획(Search Plan)을 생성한다.

    Mode A (단순 문서 검색)
        - prediction 정보가 없거나 profile이 troubleshooting_rag인 경우
        - 사용자 질의를 그대로 검색 Query로 사용한다.

    Mode B (예측 기반 문서 검색)
        - Prediction 결과의 고장 유형(failure_types)과
          원인 변수(cause_features)를 이용하여
          검색 태그와 문서 화이트리스트를 생성한다.

    Args:
        question: 사용자의 원본 질문.
        profile: Retrieval Profile. ("troubleshooting_rag", "prediction_plus_rag")
        prediction: Prediction Agent 결과.Mode B에서만 사용된다.

    Returns:
        Search Plan(dict)
            {
                mode,
                profile,
                user_query,
                search_query,
                tags,
                doc_whitelist,
                failure_types,
                failure_ko
            }
    """

    has_pred = bool(prediction and prediction.failure_types) #
    if profile != "prediction_plus_rag" or not has_pred:
        return {"mode": "A", "profile": profile, "user_query": question, "search_query": question,
                "tags": [], "doc_whitelist": None, "failure_types": [], "failure_ko": []}

    # mode B: 도출된 고장 유형을 모두 반영 + 원인 변수 태그 확장
    failure_types, failure_ko, tags, docs = [], [], [], []
    for code in prediction.failure_types:
        failure_types.append(code)
        ko = FAILURE_CODE_TO_KO.get(code)
        if ko:
            failure_ko.append(ko)
        fmap = FAILURE_RAG_MAP.get(ko, {})
        tags.extend(fmap.get("query_tags", []))
        docs.extend(fmap.get("documents", []))
    for feat in (prediction.cause_features or []):
        tags.extend(VARIABLE_TAG_MAP.get(FEATURE_TO_KO.get(feat, ""), []))
    tags = list(dict.fromkeys(tags))
    docs = list(dict.fromkeys(docs))
    return {"mode": "B", "profile": profile, "user_query": question,
            "search_query": " ".join([question, *tags]).strip(), "tags": tags,
            "doc_whitelist": docs or None, "failure_types": failure_types, "failure_ko": failure_ko}


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

#(2) Retriever------------------------------
def retrieve_stage(plan: dict, k: int = 8) -> list[dict]:
    """
    Retriever.
    Query Builder가 생성한 Search Plan을 이용하여 Pinecone에서 문서를 검색한다.

    수행 과정
        1. Retrieval Profile에 맞는 type filter 적용
        2. Vector Search 수행
        3. Retrieval Profile별 source policy 적용
        4. (Mode B인 경우) 문서 화이트리스트 적용

    Args:
        plan:
            build_query()가 생성한 Search Plan.

        k:
            Vector Search 후보 문서 개수.

    Returns:
        검색된 문서 후보 리스트.
    """
    type_filter = _profile_type_filter(plan["profile"])
    hits = vector_search(plan["search_query"], k=k, type_filter=type_filter)
    if not hits and type_filter:
        hits = vector_search(plan["search_query"], k=k, type_filter=None)
    profile = plan["profile"]
    hits = [h for h in hits if _source_allowed(h.get("source", ""), profile)]
    whitelist = plan.get("doc_whitelist")
    if whitelist:
        filtered = [h for h in hits
                    if any(_doc_name_matches(h.get("source", ""), n) for n in whitelist)]
        if filtered:
            hits = filtered
    return hits


# score = 1.0 - cosine_distance = 코사인 유사도(0~1). 코사인 공간 임베딩 기준 임계값.
MIN_EVIDENCE_SCORE = float(os.environ.get("MIN_EVIDENCE_SCORE", "0.2"))
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
        2. (source, chunk_index) 기준 중복 제거
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
        key = (h.get("source"), h.get("chunk_index"))
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
            "citations": Citation List
        }
    """
    plan = build_query(question, profile, prediction)   # (1)
    hits = retrieve_stage(plan, k=retrieve_k)             # (2)
    ranked = rank_evidence(hits, top_k=top_k)            # (3)
    if not ranked:
        return {"plan": plan, "documents": [], "citations": [], "status": "EMPTY", "limitations": ["검색된 문서가 없습니다."]}
    relevant = [d for d in ranked if float(d.get("score", 0.0)) >= MIN_EVIDENCE_SCORE]
    if not relevant:
        return {
            "plan": plan,
            "documents": ranked,
            "citations": build_citations(ranked),
            "status": "LOW_RELEVANCE",
            "limitations": [f"검색된 문서의 score가 낮아 근거 품질이 제한됩니다. threshold={MIN_EVIDENCE_SCORE}"],
        }
    return {"plan": plan, "documents": relevant, "citations": build_citations(relevant), "status": "OK", "limitations": []}


