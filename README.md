# Manufacturing Agent

LangGraph 기반 **제조 설비 진단 AI 에이전트**. 구조는 **Gate-driven Plan-and-Execute**다.
코어 로직은 `manufacturing_agent/` Python 패키지이고, **FastAPI 백엔드(`api/`) + React 프론트엔드(`frontend/`)** 로 서비스한다.

```
intake_gate → context_manager → supervisor_planner → orchestrator_dispatcher
  → prediction_agent / sql_agent / evidence_agent  (+ worker gates)
  → (optional) supervisor_replanner
  → final_answer → output_safety_gate → memory_writer
```

**주요 기능:**
- 설비 feature 기반 rule-based 위험 진단 (OSF/TWF/PWF/HDF)
- `failure_history` SQLite Text-to-SQL 과거 고장 이력 조회 (SELECT-only)
- Pinecone 기반 정비·안전 문서 RAG + citation 생성 (ChromaDB 로컬 대체 가능)
- 멀티턴 `DiagnosisContext` 관리 (CURRENT_ONLY / USE_ACTIVE / PATCH_ACTIVE / SELECT_HISTORY / REFER_ACTIVE_RESULT)
- SQLite checkpointer 기반 실패 후 resume
- 입력·출력 deterministic 안전 게이트 (intake_gate / output_safety_gate)

---

## 0. 구성 한눈에

```
manufacturing_agent/   ← 코어 로직 Python 패키지 (진단/RAG/SQL/게이트/그래프)
api/                   ← FastAPI 서버. 진입점: api/main.py:app
frontend/              ← React(Vite) 화면
scripts/               ← 임베딩·시나리오 스크립트
evals/                 ← 평가 프레임워크 (golden 데이터셋 + 러너 + calibration)
tests/                 ← 단위·통합·LLM judge 테스트
document/              ← RAG 원본 문서 (KOSHA PDF / OSHA HTML / Haas HTML)
sql/                   ← failure_history 스키마 + seed
agent_data/            ← 런타임 데이터 (SQLite). 자동 생성, git 제외
docs/                  ← 설계 문서 (아키텍처·평가·마이그레이션 등)
```

흐름: **의존성 설치 → `.env` 설정 → DB 생성 → 문서 임베딩(1회) → 백엔드 → 프론트**

> 모든 파이썬 명령은 `uv run ...` 으로 실행한다. `python`/`uvicorn`을 직접 호출하면 "인식되지 않습니다" 에러가 난다.

---

## 1. 사전 준비 (최초 1회)

### 1-1. 도구
- **Python 3.12+ / uv** — 의존성 관리 (`uv --version` 확인)
- **Node.js 18+ / npm** — 프론트엔드 (`node -v`, `npm -v` 확인)

### 1-2. 의존성 설치

```bash
uv sync --extra llm
```

> `--all-extras` 사용 금지 — jupyter 포함 시 tornado 버전 충돌로 서버 기동 불가.

### 1-3. `.env` 설정

`.env.example`을 복사해 `.env`를 만들고 필요한 키를 채운다.

```bash
cp .env.example .env
```

```env
# 필수
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-small

# 벡터 DB 선택 (기본: pinecone)
VECTOR_BACKEND=pinecone
PINECONE_API_KEY=<팀 채널에서 받은 키>
PINECONE_INDEX_NAME=sesacline-agent-docs

# LangSmith 트레이싱 (선택 — 로컬 검증 시 false 권장)
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=manufacturing-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# 분류 전용 저비용 모델 (선택)
CLASSIFIER_MODEL=gpt-4o-mini
```

> `.env`는 git에 커밋되지 않는다. `OPENAI_API_KEY` 없이는 LLM 기능이 동작하지 않는다.

### 1-4. failure_history DB 생성

```bash
uv run python -c "
import sqlite3; from pathlib import Path
Path('agent_data').mkdir(exist_ok=True)
conn = sqlite3.connect('agent_data/failure_history.sqlite')
conn.executescript(Path('sql/failure_history_schema.sql').read_text(encoding='utf-8'))
conn.close(); print('done')
"
```

또는 sqlite3 CLI가 있으면:
```bash
sqlite3 agent_data/failure_history.sqlite < sql/failure_history_schema.sql
```

### 1-5. 문서 임베딩 (벡터 DB 초기화)

**[기본] Pinecone** — 팀 공유 인덱스. API 키만 있으면 이미 빌드된 인덱스를 그대로 사용 가능.
새로 빌드하거나 문서가 바뀌었을 때만 실행:

```bash
python scripts/build_pinecone.py           # 증분 업로드 (이미 있는 청크 건너뜀)
python scripts/build_pinecone.py --reset   # 인덱스 전체 삭제 후 재빌드
python scripts/build_pinecone.py --dry-run # 업로드 계획만 출력 (실제 업로드 없음)
```

**[대안] ChromaDB 로컬** — Pinecone API 키 없이 오프라인 사용:

```bash
uv run python scripts/build_chroma.py --local    # 로컬 해시 임베딩 (API 키·비용 없음, 검색 품질 낮음)
uv run python scripts/build_chroma.py            # OpenAI 임베딩 (API 키 필요, 검색 품질 좋음)
uv run python scripts/build_chroma.py --reset    # 전체 재빌드
uv run python scripts/build_chroma.py --dry-run  # 계획만 확인
```

> ChromaDB 로컬 사용 시 `.env`에 `VECTOR_BACKEND=chroma` 추가.

---

## 2. 실행

백엔드와 프론트는 **각각 다른 터미널**에서 동시에 띄운다. 백엔드를 먼저 켠다.

### 2-1. 백엔드 (FastAPI) — 터미널 A

```bash
uv run uvicorn api.main:app --reload --port 8000
```

- API: http://localhost:8000
- Swagger 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/healthz → `{"status":"ok"}`

> 처음 기동 시 벡터스토어·그래프 로딩으로 몇 초 걸린다 (정상).

### 2-2. 프론트엔드 (React + Vite) — 터미널 B

```bash
cd frontend
npm install     # 최초 1회
npm run dev     # http://localhost:5173
```

배포용 빌드가 필요하면:
```bash
npm run build   # frontend/dist 생성
```

---

## 3. 사용 흐름

브라우저에서 **사용자 생성 → 대화(thread) 생성 → 질문** 순서다.

1. 사이드바에서 **"새 사용자 생성"** 클릭 (user_id가 브라우저에 저장됨)
2. **"새 대화 생성"** 후 선택
3. 질문 입력:
   - **자연어**: `토크 62, 공구마모 215, 회전속도 1320, 공기온도 298, 공정온도 309, 타입 M인데 고장 위험 진단해줘`
   - **구조화 입력**: 수치를 칸에 직접 입력 (자연어가 우선 — 비워도 됨)
4. 답변에 **종합 판단 / 고장 유형별 계산 근거 / 과거 이력 / 문서 citation / 점검 항목** 포함. debug 토글로 내부 게이트·태스크 확인 가능.

---

## 4. REST API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/users` | 사용자 생성 → `{user_id}` |
| `DELETE` | `/users/{user_id}` | 사용자 + 모든 대화/메모리/체크포인트 삭제 |
| `GET` | `/users/{user_id}/threads` | 대화 목록 |
| `POST` | `/users/{user_id}/threads` | 대화 생성 → `{thread_id}` |
| `DELETE` | `/users/{user_id}/threads/{thread_id}` | 대화 삭제 |
| `POST` | `/chat` | 한 턴 실행 (질문 → 진단 답변 + artifact) |
| `POST` | `/chat/stream` | SSE 스트리밍 응답 |
| `POST` | `/chat/resume` | checkpoint에서 중단 턴 재개 |
| `GET` | `/users/{uid}/threads/{tid}/history` | 대화 이력 |
| `GET` | `/usage` | LLM 사용량·토큰·추정 비용 |
| `GET` | `/healthz`, `/readyz` | 헬스체크 |

멀티턴 식별자: **`user_id`** (ConversationStore 네임스페이스) + **`thread_id`** (LangGraph checkpointer 복구 기준).

**cURL 예시:**
```bash
# 1) 사용자 생성
curl -X POST http://localhost:8000/users -H "Content-Type: application/json" -d "{}"

# 2) 대화 생성
curl -X POST http://localhost:8000/users/<USER_ID>/threads -H "Content-Type: application/json" -d "{}"

# 3) 채팅 (debug 모드)
curl -X POST "http://localhost:8000/chat?debug=true" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"<USER_ID>\",\"thread_id\":\"<THREAD_ID>\",\"message\":\"토크 62 공구마모 215 타입 M 진단해줘\"}"

# 4) 구조화 입력 + 채팅
curl -X POST "http://localhost:8000/chat?debug=true" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"<USER_ID>\",\"thread_id\":\"<THREAD_ID>\",\"message\":\"\",\"input_features\":{\"type\":\"M\",\"rotational_speed\":1320,\"torque\":62,\"tool_wear\":215,\"air_temperature\":298,\"process_temperature\":309}}"
```

---

## 5. 평가 결과

3-Layer 전략으로 평가한다. **Layer 1·2는 API 키 없이 누구나 재현 가능**하다.

### Layer 1 — 결정론적 단위 테스트 (API 키 불필요)

```bash
uv run pytest tests/ -q --ignore=tests/test_llm_judge.py --ignore=tests/test_sql_quality.py
```

| 테스트 파일 | 케이스 | 결과 | 내용 |
|---|---|---|---|
| `test_prediction_service.py` | 28 | ✅ 28/28 | TWF/OSF/HDF/PWF 임계값·경계값, 물리 공식 기반 |
| `test_intake_gate_regex.py` | 28 | ✅ 28/28 | 위험 운전·LOTO 무시 차단 14 / 오차단 방지 12 / 기타 |
| `test_output_safety_regex.py` | 23 | ✅ 23/23 | 위험 지시 차단 + 안전 권고 오차단 방지 |
| `test_quality_gates.py` | 18 | ✅ 18/18 | OK/FAIL/EMPTY 등 상태별 분기 |
| `test_rag_taxonomy.py` | 6 | ✅ 6/6 | RAG taxonomy 분류 |
| `test_regression.py` | — | ✅ | 회귀 보호 |

### Layer 2 — SQLite 통합 테스트 (API 키 불필요)

```bash
uv run pytest tests/test_sql_quality.py -q
```

| 테스트 파일 | 케이스 | 결과 | 내용 |
|---|---|---|---|
| `test_sql_quality.py` | 27 | ✅ 27/27 | SELECT-only 강제, DML/DDL/다중문 차단, SQLite 실행 포함 |

**Layer 1 + Layer 2 합산: 141/141 통과 (API 키 불필요 · 약 2.3초)**

### Layer 3 — LLM 품질 평가 (API 키 필요)

#### Supervisor 라우팅 평가

```bash
uv run python evals/scripts/run_dispatcher_eval.py
uv run python evals/scripts/run_replanner_eval.py
uv run python evals/scripts/run_sql_intent_eval.py
```

| 평가 항목 | 케이스 | 결과 |
|---|---|---|
| Dispatcher 라우팅 | 12 | ✅ 12/12 = **1.00** |
| Replanner 액션 | 12 | ✅ 12/12 = **1.00** |
| SQL Intent 분류 | 12 | ✅ 12/12 = **1.00** |

#### RAG 품질 평가 (RAGAS)

```bash
uv run python evals/ragas/run_ragas.py
```

| 지표 | 전체 | 해석 |
|---|---|---|
| **faithfulness** | **0.92** | 검색 발동 시 근거 충실도 높음 (환각 적음) |
| **context_precision** | **0.81** | 관련 문맥이 상위에 검색됨 |
| context_recall | 0.49 | top-K·청크 크기 점검 대상 |
| answer_relevancy | 0.13 | 포맷 아티팩트 (citation 불릿·헤징을 RAGAS가 회피로 오판 — faithfulness가 실제 지표) |
| **예측 정확도 (1-2 트랙)** | **8/8 = 1.00** | OSF/HDF 고장 유형 전부 정확 분류 |

#### LLM-as-a-Judge (Reference-Free 4축)

```bash
# 개별 케이스 실행
uv run python tests/test_llm_judge.py --id fa_prediction_only

# Calibration 검증 (judge 신뢰도 교정)
uv run python evals/scripts/run_calibration_eval.py
```

- 평가 방법: **Reference-Free** — gold answer 대신 `facts_sheet`(시스템이 계산한 사실)에 서술 본문이 충실한지 직접 채점
- 4축 루브릭: **G**roundedness · **C**ompleteness · **N**arrative Clarity · **N**o Overreach
- Calibration: 결함을 일부러 심은 15건(clean 5 / 결함 10)으로 judge가 실제로 해당 결함을 깎는지 검증

---

## 6. 평가 프레임워크 (`evals/`)

```bash
# 통합 golden 평가 대시보드 (routing·intake·multiturn·sql·rag)
PYTHONUTF8=1 uv run python evals/run_golden.py

# LLM judge — 최종 답변 품질 (결정적 체크 + LLM-as-judge 4축 루브릭)
PYTHONUTF8=1 uv run python evals/final_answer_judge.py

# 종단 멀티턴 — 실제 그래프를 같은 thread로 순차 실행
PYTHONUTF8=1 uv run python evals/multiturn_seq_eval.py
```

### Golden 데이터셋 (`evals/golden/`)

| 파일 | 컴포넌트 | 지표 |
|---|---|---|
| `routing.jsonl` | supervisor_planner | EM + per-label F1 |
| `intake.jsonl` | intake_gate | 위험 Recall + False-Block률 |
| `multiturn.jsonl` | context_manager | mode 정확도 + carryover F1 |
| `multiturn_seq.jsonl` | context_manager | 순서 의존 멀티턴 |
| `text_to_sql.jsonl` | sql_agent | Execution Accuracy + 안전 거절률 |
| `rag_retrieval.jsonl` | evidence_agent | Recall@k + MRR |
| `final_answer.jsonl` | final_answer_node | LLM judge 루브릭 |
| `llm_judge_cases.jsonl` | LLM judge | Reference-Free 4축, 12건 |

### Calibration 데이터셋 (`evals/calibration/`)

| 파일 | 케이스 | 내용 |
|---|---|---|
| `calibration_cases.jsonl` | 15건 | clean 5 / 결함 10, 5개 모드(예측·종합·이력·문서·입력부족) 커버 |

---

## 7. 아키텍처

| 컴포넌트 | 설명 |
|---|---|
| **SupervisorPlanner** | 요청을 분석해 `ExecutionPlan` 생성 (prediction/sql/evidence task 조합 결정) |
| **OrchestratorDispatcher** | LLM 없는 deterministic 라우터. GateReport 반영 → task 상태/retry 관리 → 다음 worker / replanner / final_answer 결정 |
| **SupervisorReplanner** | `PLAN_REPAIR_REQUIRED` gate 실패 task만 targeted patch — 전체 재계획 아님 |
| **PredictionAgent** | 물리 공식 기반 rule-only 진단 (OSF/TWF/PWF/HDF). ML 없음, API 키 불필요 |
| **SQLAgent** | Text-to-SQL. `failure_history` 단일 테이블, SELECT-only, DDL/DML/PRAGMA 금지, LIMIT 필수, readonly 검증 실행 |
| **EvidenceAgent** | Pinecone RAG. OK/EMPTY/LOW_RELEVANCE/FAIL 구분, citation metadata, prompt injection sanitize |
| **ContextManager** | 입력 우선순위: 구조화 입력 > 이번 턴 NL > 이전 맥락 carry. 사용자 명시 참조 시에만 이전 DiagnosisContext 사용 |
| **intake_gate** | 위험 운전 지속·LOTO 우회 요청 차단 (정책 기반, LLM 판단 없음) |
| **output_safety_gate** | 최종 답변의 위험 문구 차단 (정책 기반, LLM 판단 없음) |

더 자세한 설명: `docs/manufacturing_agent_v6.md`, `docs/evaluation_framework.md`

---

## 8. 개발 원칙

- 특정 시나리오 문장을 하드코딩하지 않는다
- `orchestrator_dispatcher`에 LLM 호출을 넣지 않는다
- Worker가 final answer를 직접 만들지 않는다
- Gate가 worker를 직접 재실행하지 않는다
- SQL 실패를 template fallback으로 숨기지 않는다
- Evidence가 부족하면 부족하다고 표시한다
- FinalAnswer에 raw SQL row·JSON·내부 state·debug trace를 출력하지 않는다
- 위험 실행·안전장치 우회·점검 없는 재가동 승인 표현은 `output_safety_gate`에서 차단한다

---

## 9. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `uvicorn ... 인식되지 않습니다` | 전역에 없음 → `uv run uvicorn api.main:app --reload --port 8000` |
| `OPENAI_API_KEY` 오류 | `.env` 존재·값 확인 |
| RAG 결과가 안 나옴 | Pinecone: `PINECONE_API_KEY` 확인. ChromaDB: `uv run python scripts/build_chroma.py` 후 `ls agent_data/chroma` |
| SQL 결과 FAIL | DB 확인: `sqlite3 agent_data/failure_history.sqlite ".tables"` → `failure_history` 보여야 함 |
| 프론트가 API 못 부름 | 백엔드(8000) 먼저 실행. 포트가 다르면 `CORS_ORIGINS` 환경변수에 추가 |
| LangSmith 403/경고 | `.env`에 `LANGSMITH_TRACING=false` 또는 유효한 `LANGSMITH_API_KEY` 입력 |
| `/chat` 503 llm_quota_exhausted | OpenAI Billing → Credit balance 및 프로젝트 예산 한도 확인 |
| 첫 응답이 느림 | 정상. 기동 시 벡터스토어·그래프 로딩 + LLM 호출 (수 초) |
| `모듈 manufacturing_agent 없음` | 프로젝트 루트 폴더에서 `uv run ...` 으로 실행 |
