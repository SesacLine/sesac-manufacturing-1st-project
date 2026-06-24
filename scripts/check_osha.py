import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import manufacturing_agent.config  # noqa: F401
from manufacturing_agent.services.rag_service import rag_search

queries = [
    ("machine_guarding KO", "기계 방호장치 설치 기준", "safety_procedure_rag"),
    ("machine_guarding EN", "machine guarding installation requirements", "safety_procedure_rag"),
    ("loto KO",             "잠금 표지 절차 LOTO", "safety_procedure_rag"),
]

for label, query, profile in queries:
    result = rag_search(query, profile=profile, retrieve_k=20)
    docs = result.get("documents", [])
    print(f"\n[{label}] status={result.get('status')}  docs={len(docs)}")
    for d in docs:
        print(f"  score={d.get('score', 0):.3f}  source={d.get('source')}")
