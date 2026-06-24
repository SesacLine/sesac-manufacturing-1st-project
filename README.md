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
scripts/               ← 임베딩·회귀 시나리오 스크립트
evals/                 ← 평가 프레임워크 (golden 데이터셋 + 러너)
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

## 5. 평가 프레임워크 (`evals/`)

golden 데이터셋 기반 컴포넌트별 자동 평가. 자세한 지표 정의는 `docs/evaluation_framework.md`.

### 평가 스크립트

```bash
# [통합] routing · intake · multiturn · text_to_sql · rag 종합 대시보드
PYTHONUTF8=1 uv run python evals/run_golden.py

# [LLM judge] 최종 답변 품질 (결정적 체크 + LLM-as-judge 5축 루브릭)
PYTHONUTF8=1 uv run python evals/final_answer_judge.py

# [종단 멀티턴] 실제 그래프를 같은 thread로 순차 실행
PYTHONUTF8=1 uv run python evals/multiturn_seq_eval.py
```

### golden 데이터셋 (`evals/golden/`)

| 파일 | 컴포넌트 | 지표 |
|---|---|---|
| `routing.jsonl` | supervisor_planner | EM + per-label F1 |
| `intake.jsonl` | intake_gate | 위험 Recall + False-Block률 |
| `multiturn.jsonl` | context_manager | mode 정확도 + carryover F1 |
| `text_to_sql.jsonl` | sql_agent | Execution Accuracy + 안전 거절률 |
| `rag_retrieval.jsonl` | evidence_agent (검색) | Recall@k + MRR |
| `final_answer.jsonl` | final_answer_node | LLM judge 루브릭 + 숫자 환각률 |

### 회귀 테스트 (API 키 불필요)

```bash
uv run pytest -q
```

### 시나리오 배치 실행

```bash
LANGSMITH_TRACING=false uv run python scripts/run_manufacturing_scenarios_v2.py --json
LANGSMITH_TRACING=false uv run python scripts/run_manufacturing_scenarios_v2.py --json --full-answer
```

---

## 6. 아키텍처

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

## 7. 개발 원칙

- 특정 시나리오 문장을 하드코딩하지 않는다
- `orchestrator_dispatcher`에 LLM 호출을 넣지 않는다
- Worker가 final answer를 직접 만들지 않는다
- Gate가 worker를 직접 재실행하지 않는다
- SQL 실패를 template fallback으로 숨기지 않는다
- Evidence가 부족하면 부족하다고 표시한다
- FinalAnswer에 raw SQL row·JSON·내부 state·debug trace를 출력하지 않는다
- 위험 실행·안전장치 우회·점검 없는 재가동 승인 표현은 `output_safety_gate`에서 차단한다

---

## 8. 트러블슈팅

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
