"""Pinecone에서 실패한 RAG 쿼리들이 실제로 어떤 source를 반환하는지 확인."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import manufacturing_agent.config  # noqa: F401 — .env 로드
from manufacturing_agent.services.rag_service import rag_search

FAILED_CASES = [
    ("rag_machine_guarding", "기계 가드 안전 점검 절차", "safety_procedure_rag"),
    ("rag_chatter_detail",   "공구 마모와 밀 채터 점검 방법", "troubleshooting_rag"),
    ("rag_spindle_runout",   "스핀들 런아웃 점검 절차", "troubleshooting_rag"),
]

ALL_CASES = [
    ("rag_spindle_overheat", "스핀들 베어링이 과열되는데 점검 절차와 윤활 확인 방법", "troubleshooting_rag"),
    ("rag_chatter",          "공구 마모와 밀 채터(chatter) 점검 방법", "troubleshooting_rag"),
    ("rag_loto",             "에너지 차단장치 잠금·표지(LOTO) 절차와 점검 주기", "safety_procedure_rag"),
    ("rag_machine_guarding", "기계 가드 안전 점검 절차", "safety_procedure_rag"),
    ("rag_chatter_detail",   "공구 마모와 밀 채터 점검 방법", "troubleshooting_rag"),
    ("rag_spindle_runout",   "스핀들 런아웃 점검 절차", "troubleshooting_rag"),
]

print("=== 실패 케이스 Pinecone 실제 반환 source ===\n")
for cid, query, profile in FAILED_CASES:
    result = rag_search(query, profile=profile, retrieve_k=16)
    docs = result.get("documents", [])
    print(f"[{cid}]  query={query!r}")
    print(f"  status={result.get('status')}  docs={len(docs)}건")
    for d in docs:
        print(f"  source={d.get('source')!r}  score={d.get('score', 0):.3f}")
    print()

print("\n=== 전체 케이스 unique source 목록 ===\n")
all_sources = set()
for cid, query, profile in ALL_CASES:
    result = rag_search(query, profile=profile, retrieve_k=16)
    for d in (result.get("documents") or []):
        src = d.get("source") or ""
        if src:
            all_sources.add(src)

for s in sorted(all_sources):
    print(" ", s)
