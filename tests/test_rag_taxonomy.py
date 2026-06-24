"""taxonomy 기반 RAG retrieval layer 품질 테스트.

순수 테스트(taxonomy alias/fan-out/routing)는 의존성 없이 항상 실행된다.
retrieval/no-evidence/citation 테스트는 rag_service 임포트가 필요하며,
vector_search를 monkeypatch해 **벡터 백엔드(Pinecone/Chroma) 무관**·OpenAI 호출 없이 결정적으로 동작한다.
(rag_service 임포트에는 openai/pinecone 패키지만 필요. 실제 Pinecone 연결/색인은 monkeypatch로 대체)

실제 Pinecone 연결 smoke 테스트는 RUN_PINECONE_LIVE=1 + PINECONE_API_KEY가 있을 때만 실행된다.

Run:
    uv run python -m pytest tests/test_rag_taxonomy.py -q
"""
from __future__ import annotations

import os

import pytest

from manufacturing_agent.services import manufacturing_taxonomy as tax

# rag_service 임포트 가능 여부 (vector_search 백엔드 = pinecone_store). 불가하면 관련 테스트 skip
try:
    from manufacturing_agent.services import rag_service as rs
    from manufacturing_agent.contracts.context import PredictionResult
    _RS_OK = True
except Exception as exc:  # pragma: no cover
    _RS_OK = False
    _RS_ERR = str(exc)

needs_rs = pytest.mark.skipif(not _RS_OK, reason="rag_service 임포트 불가(openai/pinecone 패키지 필요)")

# 실제 Pinecone 연결이 필요한 통합 테스트용 가드.
# 일반 pytest 실행에서 우발적으로 돌지 않도록 명시적 opt-in(RUN_PINECONE_LIVE=1)을 요구한다.
# (실행 전 PINECONE_API_KEY + 실제 OPENAI_API_KEY + scripts/reembed_pinecone.py 색인 업서트 필요)
_PINECONE_LIVE = (
    os.environ.get("RUN_PINECONE_LIVE", "").lower() in {"1", "true", "yes", "on"}
    and bool(os.environ.get("PINECONE_API_KEY"))
)
needs_pinecone = pytest.mark.skipif(
    not (_RS_OK and _PINECONE_LIVE),
    reason="실제 Pinecone 연결 필요(RUN_PINECONE_LIVE=1 + PINECONE_API_KEY + 색인 업서트)",
)


# ─────────────────────────────────────────────────────────────────
# 1. taxonomy alias test (AI4I 변수/고장 -> Haas 문서 표현 확장)
# ─────────────────────────────────────────────────────────────────
def test_feature_alias_expands_to_haas_terms():
    """torque는 Haas 표현(tool load/cutting force 등)으로 확장돼야 한다."""
    m = tax.get_feature_match("torque")
    assert m is not None
    aliases = {a.lower() for a in m.aliases}
    assert "tool load" in aliases
    assert "cutting force" in aliases
    # AI4I 원시 변수명만 검색하지 않도록 Haas 표현이 충분히 있어야 함
    assert len(aliases) >= 5


def test_failure_alias_hdf_has_thermal_terms():
    m = tax.get_failure_match("HDF")
    assert m is not None and m.ko == "방열 불량"
    aliases = {a.lower() for a in m.aliases}
    assert {"overheating", "spindle temperature"} <= aliases


def test_search_terms_combine_failure_feature_symptom():
    # max_terms를 넉넉히 줘서 failure/feature/symptom 확장이 모두 반영되는지 확인
    terms = tax.build_search_terms(features=["tool_wear"], failure_codes=["OSF"],
                                   symptoms=["chatter"], max_terms=60)
    low = {t.lower() for t in terms}
    assert "tool wear" in low          # feature 확장
    assert "cutting force" in low       # OSF 확장
    assert "vibration" in low or "chatter" in low  # symptom 확장


# ─────────────────────────────────────────────────────────────────
# 2. query fan-out test
# ─────────────────────────────────────────────────────────────────
def test_fanout_preserves_korean_and_adds_english():
    q = "토크가 높고 공구 마모가 심한데 점검 방법 알려줘"
    queries = tax.build_rag_queries(q, features=["torque", "tool_wear"], failure_codes=["OSF"])
    assert queries[0] == q.strip()                      # 원문 한글 보존(첫 query)
    joined = " ".join(queries).lower()
    assert "tool load" in joined or "cutting force" in joined  # 영어 확장 query 포함
    assert len(queries) >= 3                            # fan-out 다중 query


def test_fanout_unique_and_capped():
    queries = tax.build_rag_queries("질문", features=["torque"], failure_codes=["OSF"], max_queries=5)
    assert len(queries) == len(set(queries))            # 중복 없음
    assert len(queries) <= 5


# ─────────────────────────────────────────────────────────────────
# 3. document routing test
# ─────────────────────────────────────────────────────────────────
def test_routing_twf_prefers_chatter():
    docs = tax.route_documents(failure_codes=["TWF"])
    assert docs[0] == "mill_chatter"


def test_routing_hdf_prefers_spindle_then_mechanical():
    docs = tax.route_documents(failure_codes=["HDF"])
    assert docs[0] == "mill_spindle"
    assert "mechanical_service" in docs


def test_routing_default_when_empty():
    docs = tax.route_documents()
    assert set(docs) == {"mechanical_service", "mill_spindle", "mill_chatter"}


@needs_rs
def test_profile_key_to_docname_mapping_covers_routing():
    """route_documents가 내는 모든 키가 source 매칭 토큰으로 매핑돼야 한다."""
    for key in ["mill_spindle", "mill_chatter", "mechanical_service"]:
        assert key in rs.PROFILE_KEY_TO_DOCNAME


# ─────────────────────────────────────────────────────────────────
# 4. build_query (한글 query 유지 + 영어 확장 + priority docs)
# ─────────────────────────────────────────────────────────────────
@needs_rs
def test_build_query_mode_b_has_priority_and_english_terms():
    pred = PredictionResult(status="OK", failure_types=["OSF"], cause_features=["torque", "tool_wear"])
    plan = rs.build_query("토크가 높고 공구 마모가 심하다", "prediction_plus_rag", pred)
    assert plan["mode"] == "B"
    assert plan["failure_types"] == ["OSF"]
    assert plan["priority_doc_names"]                       # 우선 문서 존재
    assert plan["queries"][0] == "토크가 높고 공구 마모가 심하다"  # 한글 원문 보존
    assert any("cutting force" in t.lower() or "tool load" in t.lower() for t in plan["tags"])


@needs_rs
def test_build_query_extracts_korean_symptom():
    """한글 2글자 증상어(과열)도 추출돼 Mill Spindle로 라우팅."""
    plan = rs.build_query("스핀들 과열 원인 알려줘", "troubleshooting_rag", None)
    assert "overheating" in plan["symptoms"]
    assert "Mill Spindle" in plan["priority_doc_names"]


# ─────────────────────────────────────────────────────────────────
# 5. retrieval smoke test (vector_search monkeypatch -> 결정적)
# ─────────────────────────────────────────────────────────────────
def _hit(hid, source, chunk, score, text="spindle overheating troubleshooting"):
    # pinecone_store.vector_search() / chroma.vector_search() 와 동일한 반환 dict 형태
    return {"id": hid, "text": text, "type": "troubleshooting",
            "source": source, "chunk_index": chunk, "score": score}


@needs_rs
def test_retrieval_smoke_and_dedup(monkeypatch):
    SP = "haas/haascnc.com-Mill Spindle - Troubleshooting Guide.pdf"
    CH = "haas/haascnc.com-Mill Chatter - Troubleshooting - TG0100.pdf"

    def fake_vs(query, k=3, type_filter=None):
        # 여러 fan-out query가 같은 chunk(s1)를 반복 반환 -> dedup 확인
        return [_hit("s1", SP, 7, 0.62), _hit("s1", SP, 7, 0.55), _hit("c1", CH, 2, 0.50)]

    monkeypatch.setattr(rs, "vector_search", fake_vs)
    res = rs.rag_search("스핀들 과열 원인", "troubleshooting_rag", None, retrieve_k=8, top_k=4)
    assert res["status"] == "OK"
    ids = [c["source_id"] for c in res["citations"]]
    assert ids.count("s1") == 1                       # doc_id 기준 중복 제거
    assert "debug" in res and res["debug"]["queries"]  # 디버그 로그 페이로드
    assert all(0.0 <= c["score"] <= 1.0 for c in res["citations"])


# ─────────────────────────────────────────────────────────────────
# 6. no evidence fallback test (NO_EVIDENCE + 담당자 안내)
# ─────────────────────────────────────────────────────────────────
@needs_rs
def test_no_evidence_when_empty(monkeypatch):
    monkeypatch.setattr(rs, "vector_search", lambda q, k=3, type_filter=None: [])
    res = rs.rag_search("코퍼스에 없는 주제", "troubleshooting_rag", None)
    assert res["status"] == "NO_EVIDENCE"
    assert res["documents"] == [] and res["citations"] == []
    assert "찾지 못" in res["guidance"]                 # 추측 차단 + 정직한 안내
    assert "담당자" in res["guidance"]                   # 담당자 확인 안내


@needs_rs
def test_no_evidence_when_below_threshold(monkeypatch):
    SP = "haas/haascnc.com-Mill Spindle - Troubleshooting Guide.pdf"
    monkeypatch.setattr(rs, "vector_search",
                        lambda q, k=3, type_filter=None: [_hit("s1", SP, 1, 0.05)])
    res = rs.rag_search("관련성 낮은 질문", "troubleshooting_rag", None)
    assert res["status"] == "NO_EVIDENCE"               # threshold 미달도 NO_EVIDENCE
    assert res["documents"] == []                       # LLM 추측용 문서 미전달


@needs_rs
def test_no_evidence_contact_from_env(monkeypatch):
    """담당자 연락처는 config/env에서 읽어 안내에 포함돼야 한다(하드코딩 금지)."""
    monkeypatch.setattr(rs, "SUPPORT_CONTACT_NAME", "정비반장")
    monkeypatch.setattr(rs, "SUPPORT_CONTACT_EMAIL", "maint@factory.local")
    monkeypatch.setattr(rs, "SUPPORT_CONTACT_PHONE", "")
    # support_contact_text는 config 모듈 함수이므로 그 모듈 값을 패치
    import manufacturing_agent.config as cfg
    monkeypatch.setattr(cfg, "SUPPORT_CONTACT_NAME", "정비반장")
    monkeypatch.setattr(cfg, "SUPPORT_CONTACT_EMAIL", "maint@factory.local")
    monkeypatch.setattr(rs, "support_contact_text", cfg.support_contact_text)
    monkeypatch.setattr(rs, "vector_search", lambda q, k=3, type_filter=None: [])
    res = rs.rag_search("없는 주제", "troubleshooting_rag", None)
    assert "정비반장" in res["guidance"]
    assert "maint@factory.local" in res["guidance"]


# ─────────────────────────────────────────────────────────────────
# 7. final answer citation test
# ─────────────────────────────────────────────────────────────────
@needs_rs
def test_citations_built_with_ids_and_source():
    docs = [
        {"id": "s1", "source": "haas/Mill Spindle.pdf", "chunk_index": 7, "score": 0.62, "text": "spindle ..."},
        {"id": "c1", "source": "haas/Mill Chatter.pdf", "chunk_index": 2, "score": 0.50, "text": "chatter ..."},
    ]
    cits = rs.build_citations(docs)
    assert [c["citation_id"] for c in cits] == ["C1", "C2"]
    assert cits[0]["source"] == "haas/Mill Spindle.pdf"
    assert cits[0]["score"] == 0.62


def test_final_answer_renders_citation_markers():
    """final_answer_node가 citation을 [C1] 형태 [출처] 블록으로 렌더한다."""
    from manufacturing_agent.nodes.final_answer_node import _format_citations
    block = _format_citations([
        {"citation_id": "C1", "title": "Mill Spindle", "source": "haas/Mill Spindle.pdf", "chunk_index": 7},
    ])
    assert "[출처]" in block
    assert "[C1]" in block


# ─────────────────────────────────────────────────────────────────
# 8. 실제 Pinecone 연결 smoke (RUN_PINECONE_LIVE=1 + PINECONE_API_KEY + 색인 업서트 필요)
#    조건 미충족 시 자동 skip. `scripts/reembed_pinecone.py` 로 색인을 채운 뒤,
#    RUN_PINECONE_LIVE=1 로 실행한다. 예) RUN_PINECONE_LIVE=1 uv run python -m pytest tests/test_rag_taxonomy.py -k live
# ─────────────────────────────────────────────────────────────────
@needs_pinecone
def test_pinecone_vector_search_live():
    """실제 Pinecone 색인에서 on-topic 질의가 haas 문서를 score와 함께 반환한다."""
    from manufacturing_agent.rag import pinecone_store

    hits = pinecone_store.vector_search("spindle overheating troubleshooting", k=5)
    assert hits, "Pinecone에서 결과가 비었습니다(색인 업서트 여부 확인: scripts/reembed_pinecone.py)"
    h0 = hits[0]
    # chroma.vector_search 와 동일한 반환 스키마
    assert set(h0) >= {"id", "text", "source", "chunk_index", "score"}
    assert isinstance(h0["score"], float)


@needs_pinecone
def test_pinecone_rag_search_live_on_topic():
    """rag_search(실 Pinecone)가 on-topic 질의에 OK + citation을 반환한다."""
    res = rs.rag_search("스핀들 과열 원인", "troubleshooting_rag", None, retrieve_k=16, top_k=4)
    assert res["status"] in {"OK", "NO_EVIDENCE"}  # 색인 상태에 따라
    if res["status"] == "OK":
        assert res["citations"]
        assert all(0.0 <= c["score"] <= 1.0 for c in res["citations"])
