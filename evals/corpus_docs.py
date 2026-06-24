"""RAG 평가/시나리오 공용 — 현재 코퍼스(PDF 기준) 문서 식별 단일 소스(드리프트 방지).

목적
- golden 라벨(evals/golden/rag_retrieval.jsonl)과 검색 결과 source를 **하나의 doc_key**로 정규화한다.
- 파일명이 바뀌어도(html→pdf, 접두어 변경 등) **여기 tokens만 고치면** golden/eval이 따라간다.
- 코퍼스에 없는 라벨(드리프트)을 조기에 감지(label_to_key가 None을 반환).

규칙
- tokens: 모두 소문자 source에 포함되면 그 문서로 인정(부분일치). 코드 토큰(tg/m-번호, haascnc.com)이
  문서를 변별하도록 고른다. html 전용 옛 문서는 일부러 매칭 안 되게 둔다(PDF만).
"""
from __future__ import annotations

# doc_key -> (사람이 읽는 이름, 매칭 토큰[전부 포함되면 매칭])
# 코퍼스 = haas PDF 전용 (osha/kosha 안전문서는 제거됨).
CORPUS_DOCS: dict[str, tuple[str, list[str]]] = {
    "haas_mechanical_service": ("Haas Mechanical Service Manual (PDF)", ["mechanical", "service", "manual"]),
    "haas_mill_spindle":       ("Haas Mill Spindle Troubleshooting (PDF)", ["haascnc.com", "mill", "spindle"]),
    "haas_mill_chatter":       ("Haas Mill Chatter Troubleshooting (PDF)", ["haascnc.com", "mill", "chatter"]),
}


def source_to_key(source: str) -> str | None:
    """검색 결과 source(또는 정규화된 source)를 코퍼스 doc_key로 변환. 미등록이면 None."""
    s = (source or "").lower()
    for key, (_, tokens) in CORPUS_DOCS.items():
        if tokens and all(t in s for t in tokens):
            return key
    return None


def label_to_key(label: str) -> str | None:
    """golden 라벨을 doc_key로 변환. 라벨이 곧 doc_key면 그대로, 아니면 source처럼 토큰 매칭."""
    if label in CORPUS_DOCS:
        return label
    return source_to_key(label)


def human_name(key: str) -> str:
    return CORPUS_DOCS.get(key, (key, []))[0]
