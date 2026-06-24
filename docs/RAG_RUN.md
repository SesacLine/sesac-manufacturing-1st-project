# RAG 실행 프로세스 (짧은 가이드)

RAG(문서 근거 검색) 파트만 ingest → 검색 → 테스트 하는 절차다.
설계/내부 동작은 [RAG_RETRIEVAL_ARCHITECTURE.md](RAG_RETRIEVAL_ARCHITECTURE.md) 참고.

```
코퍼스(document/) → 임베딩/색인(ingest) → vector_search(백엔드) → taxonomy fan-out → 답변+citation
```

---

## 1. 백엔드 선택 (.env)

벡터 백엔드는 `chroma`(로컬) 또는 `pinecone`(클라우드) 중 선택한다.

```bash
# 공통
OPENAI_API_KEY=sk-...
# 담당자 안내(NO_EVIDENCE 시 노출, 하드코딩 금지)
SUPPORT_CONTACT_NAME=설비 정비 담당자
SUPPORT_CONTACT_EMAIL=
SUPPORT_CONTACT_PHONE=

# Pinecone를 쓸 때만
VECTOR_BACKEND=pinecone
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=sesacline-agent-docs   # (선택, 기본값 동일)
PINECONE_CLOUD=aws                          # (선택)
PINECONE_REGION=us-east-1                   # (선택)
```

> 런타임 `vector_search`는 `manufacturing_agent/rag/pinecone_store.py`(Pinecone) /
> `manufacturing_agent/rag/chroma.py`(Chroma)에 있고, `rag_service.py`가 이를 import한다.

---

## 2. 코퍼스 ingest (1회, 문서 바뀌면 재실행)

대상 코퍼스: `document/haas` 3종(Mechanical Service / Mill Spindle / Mill Chatter) + `osha`/`kosha` 안전문서.
(haas_backup, Mill Accuracy는 제외)

**Pinecone:**
```bash
uv run python scripts/reembed_pinecone.py            # 색인 자동 생성 + 업서트
uv run python scripts/reembed_pinecone.py --reset    # 색인 재생성
uv run python scripts/reembed_pinecone.py --dry-run  # 대상 청크만 출력
```

**Chroma:**
```bash
uv run python scripts/reembed_corpus.py              # agent_data/chroma 재구축(reset 포함)
# 또는 노트북 01_embed_documents_chroma.ipynb 실행
```

---

## 3. RAG 검색 테스트 실행

### (a) 시나리오 러너 (실제 LLM 호출 — 비용 발생)
```bash
uv run python scripts/run_rag_scenarios.py                       # 전체
uv run python scripts/run_rag_scenarios.py --group cause         # 그룹만
uv run python scripts/run_rag_scenarios.py --scenario RAG_CAUSE_01 --trace
uv run python scripts/run_rag_scenarios.py --full-answer --dump-dir traces/rag
```
그룹: `cause`, `inspection`, `maintenance`, `preventive`,
`prediction_rag_single`, `prediction_rag_multiturn`, `empty`.
출력에 **소요 시간 + 유사도 score + rag_trace(fan-out/priority/source/score)** 포함.
(`make test-rag` 도 동일)

### (b) 노트북 (시나리오=셀, 하나씩 관찰)
`scripts/manufacturing_rag_scenarios.ipynb` 열고 → 부트스트랩 셀 먼저 → 원하는 셀 실행.

### (c) 단위 테스트 (no-LLM, CI용)
```bash
uv run python -m pytest tests/test_rag_taxonomy.py -q
```
taxonomy/fan-out/routing/dedup/NO_EVIDENCE/citation 검증. `vector_search`를 monkeypatch해 백엔드 무관.

**실제 Pinecone 연결 smoke (색인 채운 뒤, 명시 opt-in):**
```bash
RUN_PINECONE_LIVE=1 uv run python -m pytest tests/test_rag_taxonomy.py -k live -q
```

---

## 4. 디버그 / 자주 막히는 곳

- 검색 라우팅·쿼리·score 로그: `RAG_DEBUG=true` 환경변수 (stderr로 출력).
- **결과가 비거나 score 낮음 → `NO_EVIDENCE`**: 추측 답변을 막고 담당자 확인 안내를 노출(임계값 `MIN_EVIDENCE_SCORE`, 기본 0.45).
- Pinecone인데 결과가 0건: 색인 업서트 안 됨 → `scripts/reembed_pinecone.py` 실행 확인.
- `PINECONE_API_KEY` 누락: 검색 첫 호출 시 에러 → `.env` 확인.
