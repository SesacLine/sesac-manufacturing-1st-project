"""RAG 검색 평가 — golden/rag_retrieval.jsonl 기준 Recall@k / Precision@k / MRR.

설계 메모(개선 반영):
- 답변단(top_k=4, MIN_EVIDENCE_SCORE) 게이트의 영향을 배제하기 위해 **리트리버 단계
  (build_query + retrieve_stage)** 를 직접 평가한다. → 'R@k'가 실제 top-k를 의미.
- doc 매칭은 **토큰 부분일치**(접두어 'haascnc.com-' / 확장자 차이에 견고).
- 코퍼스는 **PDF 기준**. 검색 결과의 .html source는 평가에서 제외한다(html 미고려).
- relevant_doc_ids가 빈 케이스(코퍼스 미보유)는 status만 관찰(생성단 정직성으로 평가).

run_golden.py 도 이 모듈의 함수를 import해 동일 로직을 쓴다(중복 제거).

실행: PYTHONUTF8=1 PYTHONPATH=. python evals/rag_retrieval_eval.py
"""
from __future__ import annotations
import os, sys, json, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # evals/ (corpus_docs import)
from manufacturing_agent.services.rag_service import build_query, retrieve_stage, rag_search
from corpus_docs import source_to_key, label_to_key  # 코퍼스 doc_key 단일 소스(드리프트 방지)

GOLD = os.path.join(os.path.dirname(__file__), "golden", "rag_retrieval.jsonl")
PASS_RECALL = 0.5
IGNORE_HTML = True   # PDF 코퍼스 기준 — .html source는 평가에서 제외


def _norm(source: str) -> str:
    """source에서 확장자 제거 + 소문자화하지 않은 원형 키."""
    return re.sub(r"\.(html?|pdf)$", "", str(source or "").strip(), flags=re.I)


def _is_html(source: str) -> bool:
    return str(source or "").strip().lower().endswith((".html", ".htm"))


def _tokens(doc_id: str) -> list[str]:
    return [t for t in re.split(r"[ /_]+", _norm(doc_id).lower()) if len(t) >= 3]


def _doc_match(retrieved_norm: str, gold_id: str) -> bool:
    """같은 문서인지 판정. 1순위: 공유 doc_key 레지스트리(corpus_docs)로 정규화 비교
    (파일명 드리프트에 견고). 둘 다 키 해석되면 키 동등성, 아니면 토큰 부분일치로 폴백."""
    rk, gk = source_to_key(retrieved_norm), label_to_key(gold_id)
    if rk and gk:
        return rk == gk
    r = (retrieved_norm or "").lower()
    toks = _tokens(gold_id)
    return bool(toks) and all(t in r for t in toks)


def validate_golden(cases: list[dict]) -> list[str]:
    """golden 라벨이 현재 코퍼스 doc_key로 해석되는지 검사. 미해석=코퍼스 불일치(드리프트)."""
    unknown = []
    for c in cases:
        for g in (c.get("relevant_doc_ids") or []):
            if not label_to_key(g):
                unknown.append(f"{c.get('id')}: {g}")
    return unknown


def ranked_sources(query: str, k: int, profile: str = "troubleshooting_rag") -> list[str]:
    """리트리버 후보를 score 내림차순 top-k 정규화 source로(답변 top_k/score 게이트 우회).
    PDF 기준 — html source는 제외한다."""
    plan = build_query(query, profile)                       # prediction=None → mode A
    hits = retrieve_stage(plan, k=k)
    if IGNORE_HTML:
        hits = [h for h in hits if not _is_html(h.get("source"))]
    hits = sorted(hits, key=lambda h: float(h.get("score", 0.0)), reverse=True)
    return [_norm(h.get("source")) for h in hits[:k]]


def metrics(ranked: list[str], relevant: list[str], k: int) -> dict:
    topk = ranked[:k]
    hit = {g for g in relevant if any(_doc_match(r, g) for r in topk)}
    recall = len(hit) / len(relevant) if relevant else None
    distinct = list(dict.fromkeys(topk))
    matched = [r for r in distinct if any(_doc_match(r, g) for g in relevant)]
    precision = (len(matched) / len(distinct)) if distinct else 0.0
    mrr = 0.0
    for i, s in enumerate(topk, 1):
        if any(_doc_match(s, g) for g in relevant):
            mrr = 1.0 / i
            break
    return {"recall": recall, "precision": precision, "mrr": mrr,
            "hit": sorted(hit), "missed": sorted(set(relevant) - hit)}


def _preflight() -> bool:
    backend = os.environ.get("VECTOR_BACKEND", "chroma")
    index = os.environ.get("PINECONE_INDEX_NAME", "-")
    print(f"[backend={backend} index={index} ignore_html={IGNORE_HTML}]")
    if not ranked_sources("스핀들 점검 절차", 5, "troubleshooting_rag"):
        print("⚠ 검색 결과가 비었습니다 — 색인 미적재/코퍼스 불일치. "
              "scripts/reembed_pinecone.py 로 색인 확인 후 재실행. 평가 중단.")
        return False
    return True


def main() -> int:
    if not _preflight():
        return 1
    cases = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    drift = validate_golden(cases)
    if drift:
        print("⚠ 코퍼스 미등록 golden 라벨(드리프트 — corpus_docs.py/golden 점검 필요):")
        for d in drift:
            print("   -", d)
    rows, rec, mrr, prec, scored = [], [], [], [], 0
    seen = set()  # 자기 케이스 top-k에 한 번이라도 등장한 golden id
    for c in cases:
        relevant = list(c.get("relevant_doc_ids") or [])
        k = int(c.get("k", 16))
        ranked = ranked_sources(c["query"], k, c.get("profile", "troubleshooting_rag"))
        if not relevant:                       # 코퍼스 미보유 → status만 관찰
            status = rag_search(c["query"], profile=c.get("profile", "troubleshooting_rag")).get("status")
            rows.append(f"~ {c['id']:<24} [no-gold] status={status} top1={ranked[0] if ranked else '-'}")
            continue
        m = metrics(ranked, relevant, k)
        scored += 1; rec.append(m["recall"]); mrr.append(m["mrr"]); prec.append(m["precision"])
        seen |= set(m["hit"])
        ok = m["recall"] is not None and m["recall"] >= PASS_RECALL
        rows.append(f"{'✓' if ok else '✗'} {c['id']:<24} R@{k}={m['recall']:.2f} P={m['precision']:.2f} MRR={m['mrr']:.2f}"
                    + (f"  missed={m['missed']}" if m["missed"] else ""))

    print("\n=== RAG 검색 평가 (golden/rag_retrieval.jsonl) ===")
    print("\n".join(rows))
    if scored:
        print(f"\n평균 Recall@k = {sum(rec)/scored:.2f} | 평균 Precision = {sum(prec)/scored:.2f} "
              f"| 평균 MRR = {sum(mrr)/scored:.2f} | 채점 {scored}건 (pass 기준 R>={PASS_RECALL})")
    all_gold = {g for c in cases for g in (c.get("relevant_doc_ids") or [])}
    never = sorted(all_gold - seen)
    if never:
        print(f"\n⚠ 한 번도 검색 안 된 golden id(라벨/코퍼스 불일치 의심): {never}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
