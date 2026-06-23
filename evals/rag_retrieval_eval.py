"""RAG 검색 평가 — golden/rag_retrieval.jsonl 기준 Recall@k / Precision@k / MRR.

doc-level 매칭: 검색 결과 document의 source(예: 'haas/...TG0101.html')에서 확장자를 떼고
golden relevant_doc_ids(예: 'haas/...TG0101')와 비교한다.
relevant_doc_ids가 빈 케이스(코퍼스 미보유)는 retrieval status만 관찰(생성단에서 정직성 평가).

실행: PYTHONUTF8=1 PYTHONPATH=. python evals/rag_retrieval_eval.py
"""
from __future__ import annotations
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manufacturing_agent.services.rag_service import rag_search

GOLD = os.path.join(os.path.dirname(__file__), "golden", "rag_retrieval.jsonl")


def _norm(source: str) -> str:
    return re.sub(r"\.(html?|pdf)$", "", str(source or "").strip(), flags=re.I)


def _retrieved_sources(query: str, k: int, profile: str = "troubleshooting_rag"):
    """검색 결과 document를 rank 순서대로 정규화된 source 리스트로(중복 포함).
    profile은 케이스가 지정한 것을 쓴다 — agent가 의도별로 고르는 프로파일이 검색 대상 문서군을 가르기 때문."""
    r = rag_search(query, profile=profile, retrieve_k=k)
    return [_norm(d.get("source")) for d in (r.get("documents") or [])], r.get("status")


def _metrics(ranked: list[str], relevant: set[str], k: int) -> dict:
    topk = ranked[:k]
    hit_docs = set(topk) & relevant
    recall = len(hit_docs) / len(relevant) if relevant else None
    distinct_topk = list(dict.fromkeys(topk))
    precision = (len(hit_docs) / len(distinct_topk)) if distinct_topk else 0.0
    mrr = 0.0
    for i, s in enumerate(topk, 1):
        if s in relevant:
            mrr = 1.0 / i
            break
    return {"recall": recall, "precision": precision, "mrr": mrr,
            "hit": sorted(hit_docs), "missed": sorted(relevant - hit_docs)}


def main() -> int:
    cases = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    rows, rec, mrr, scored = [], [], [], 0
    for c in cases:
        relevant = set(c.get("relevant_doc_ids") or [])
        k = int(c.get("k", 16))
        ranked, status = _retrieved_sources(c["query"], k, c.get("profile", "troubleshooting_rag"))
        if not relevant:                       # 코퍼스 미보유 → 관찰만
            rows.append(f"~ {c['id']:<22} [no-gold] status={status} top1={ranked[0] if ranked else '-'}")
            continue
        m = _metrics(ranked, relevant, k)
        scored += 1; rec.append(m["recall"]); mrr.append(m["mrr"])
        ok = m["recall"] and m["recall"] >= 0.5
        tag = "✓" if ok else "✗"
        rows.append(f"{tag} {c['id']:<22} R@{k}={m['recall']:.2f} P={m['precision']:.2f} MRR={m['mrr']:.2f}"
                    + (f"  missed={m['missed']}" if m["missed"] else ""))
    print("\n=== RAG 검색 평가 (golden/rag_retrieval.jsonl) ===")
    print("\n".join(rows))
    if scored:
        print(f"\n평균 Recall@k = {sum(rec)/scored:.2f} | 평균 MRR = {sum(mrr)/scored:.2f} | 채점 {scored}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
