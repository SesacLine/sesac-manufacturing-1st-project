# 편의 명령. 예: make test-rag
.PHONY: test test-rag test-rag-trace rag-nb

# 단위/회귀 (no-LLM) 테스트
test:
	uv run python -m pytest tests/ -q

# RAG 전용 시나리오 러너 (실제 LLM 호출 — 비용 발생)
test-rag:
	uv run python scripts/run_rag_scenarios.py

# RAG 시나리오 + trace 덤프 (fan-out/priority/source/score)
test-rag-trace:
	uv run python scripts/run_rag_scenarios.py --dump-dir traces/rag

# RAG 테스트 노트북 재생성 (scripts/manufacturing_rag_scenarios.ipynb)
rag-nb:
	uv run python scripts/_gen_rag_scenarios_nb.py
