# Manufacturing Agent

LangGraph 기반 **제조 설비 진단 AI 에이전트**. 구조는 ReAct가 아니라 **Gate-driven Plan-and-Execute**다.
코어 로직은 `manufacturing_agent/` **Python 패키지**이고, **FastAPI 백엔드(`api/`) + React 프론트엔드(`frontend/`)** 로 서비스한다.
(노트북 `*.ipynb`은 더 이상 실행 경로가 아니라 참고/실험용이다.)

```text
intake_gate → context_manager → supervisor_planner → orchestrator_dispatcher
   → prediction_agent / sql_agent / evidence_agent  (+ worker gates)
   → (optional) supervisor_replanner
   → final_answer → output_safety_gate → memory_writer
```

주요 기능:
- 설비 feature 기반 rule-based 위험 진단 (OSF/TWF/PWF/HDF)
- `failure_history` SQLite 기반 과거 고장 이력 / 조치 / 반복 패턴 조회 (Text-to-SQL, SELECT-only)
- ChromaDB 기반 정비/안전 문서 RAG + citation 생성
- 멀티턴 `DiagnosisContext` 관리 (CURRENT_ONLY / USE_ACTIVE / PATCH_ACTIVE / SELECT_HISTORY / REFER_ACTIVE_RESULT)
- SQLite checkpointer 기반 실패 후 resume
- output safety deterministic backstop

---

## 0. 구성 한눈에

```text
manufacturing_agent/   ← 코어 로직 Python 패키지 (진단/RAG/SQL/게이트/그래프)
api/                   ← FastAPI 서버 (REST API). 진입점: api/main.py:app
frontend/              ← React(Vite) 화면
scripts/               ← 임베딩·회귀 시나리오 실행 스크립트
evals/                 ← 평가 프레임워크(golden 데이터셋 + 러너)
document/              ← 임베딩할 원본 문서(KOSHA/OSHA/Haas)
sql/                   ← failure_history 스키마 + seed
agent_data/            ← 런타임 데이터(ChromaDB 벡터·SQLite). 자동 생성, git 제외
```

흐름: **의존성 설치 → `.env` → DB 생성 → 문서 임베딩(1회) → 백엔드 → 프론트**

> 모든 파이썬 명령은 `uv run ...` 으로 실행한다 (가상환경 자동 사용 — `python`/`uvicorn`을 직접 부르면 "인식되지 않습니다" 에러).

---

## 1. 사전 준비 (최초 1회)

### 1-1. 도구
- **Python 3.12+ / uv** — 의존성 관리 (`uv --version`). 없으면: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 18+ / npm** — 프론트엔드 (`node -v`, `npm -v`)

### 1-2. 의존성 설치
```bash
uv sync --all-extras
```

### 1-3. `.env` 설정
`.env.example`을 복사해 `.env`를 만들고 OpenAI 키를 넣는다.
```bash
cp .env.example .env
```
```env
OPENAI_API_KEY=sk-...your-key...
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-small

# (선택) LangSmith 추적 — 로컬 검증에선 꺼두는 게 편하다
LANGSMITH_TRACING=false
LANGCHAIN_TRACING_V2=false
```
> 키는 코드에 적지 말고 `.env`에만. `.env`는 git에 커밋되지 않는다. `OPENAI_API_KEY`가 없으면 실행이 정상 완료되지 않는다.

### 1-4. failure_history DB 생성
```bash
uv run python - <<'PY'
import sqlite3
from pathlib import Path
Path("agent_data").mkdir(exist_ok=True)
sql = Path("sql/failure_history_schema.sql").read_text(encoding="utf-8")
conn = sqlite3.connect("agent_data/failure_history.sqlite")
conn.executescript(sql); conn.close()
print("failure_history.sqlite created")
PY
```
> `sqlite3` CLI가 있으면: `sqlite3 agent_data/failure_history.sqlite < sql/failure_history_schema.sql`

### 1-5. 문서 임베딩 (ChromaDB)
`document/` 문서를 벡터로 임베딩한다. **OpenAI 임베딩 API 비용이 든다.**
```bash
uv run python scripts/build_chroma.py
```

---

## 2. 실행 (백엔드 + 프론트)

> 백엔드와 프론트는 **각각 다른 터미널**에서 동시에 띄운다. 프론트가 백엔드 API를 호출하므로 백엔드를 먼저 켠다.

### 2-1. 백엔드 (FastAPI) — 터미널 A
```bash
uv run uvicorn api.main:app --reload --port 8000
```
- API: http://localhost:8000 · 스웨거 문서: http://localhost:8000/docs
- ⚠️ `uvicorn ...` 직접 호출 금지 → 반드시 `uv run uvicorn ...`

### 2-2. 프론트엔드 (React + Vite) — 터미널 B
```bash
cd frontend
npm install        # 최초 1회만
npm run dev        # http://localhost:5173
```
브라우저에서 **http://localhost:5173** 접속.

### (배포용) 프론트 빌드
```bash
cd frontend
npm run build      # frontend/dist 생성
npm run preview    # 빌드 결과 미리보기
```

---

## 3. REST API 요약 (`api/`)

| 메서드 · 경로 | 역할 |
|---|---|
| `POST /chat` | 한 턴 실행 (질문 → 최종 답변 + artifact) |
| `POST /chat/stream` | 스트리밍 응답 |
| `POST /chat/resume` | checkpoint에서 실패/중단 턴 resume |
| `POST /users`, `GET /users/{id}/threads`, `POST /users/{id}/threads` | 사용자·스레드 관리 |
| `GET /users/{user_id}/threads/{thread_id}/history` | 대화 이력 |
| `GET /usage` | 사용량 |
| `GET /healthz`, `GET /readyz` | 헬스체크 |

멀티턴 식별자: **`user_id`**(ConversationStore namespace) + **`thread_id`**(LangGraph checkpointer 복구 기준). `session_id`는 쓰지 않는다.

---

## 4. 코드에서 직접 한 턴 실행 (.py)

API/프론트 없이 패키지만으로도 실행된다.
```bash
uv run python - <<'PY'
import manufacturing_agent.runtime as rt
uid, tid, turn = "u1", "t1", "r1"
res = rt.app.invoke(
    rt.make_initial_state("토크 64, 공구마모 215, 회전속도 1380으로 진단해줘", uid, tid, turn),
    config=rt.make_runnable_config(uid, tid, turn),
)
print(res["final_answer"].answer)
PY
```

---

## 5. 회귀 시나리오 & 평가

### 5-1. 시나리오 회귀 (`scripts/`)
```bash
# 전체
LANGSMITH_TRACING=false uv run python scripts/run_manufacturing_scenarios_v2.py --json
# 답변까지 자세히
LANGSMITH_TRACING=false uv run python scripts/run_manufacturing_scenarios_v2.py --json --full-answer
```

### 5-2. 평가 프레임워크 (`evals/`)
컴포넌트별 golden 데이터셋 + 러너. 자세한 지표 정의는 `docs/evaluation_framework.md`.
```bash
# 분류/검색 통합 대시보드 (routing·intake·multiturn·sql·rag)
uv run python evals/run_golden.py
# RAG 검색 (Recall@k / MRR)
uv run python evals/rag_retrieval_eval.py
# 최종 답변 (결정적 체크 + LLM-as-judge)
uv run python evals/final_answer_judge.py
# 진짜 멀티턴(종단, 같은 thread로 실제 그래프 순차 실행)
uv run python evals/multiturn_seq_eval.py
```

### 5-3. 유닛 테스트
```bash
uv run pytest -q
```

---

## 6. 아키텍처 요약

- **SupervisorPlanner** — 요청을 보고 `ExecutionPlan`(prediction/sql/evidence/final_answer task)을 만든다. task별 `params`/`success_criteria` 보유.
- **OrchestratorDispatcher** — LLM 없는 deterministic state machine. 직전 `GateReport` 반영 → task status/retry 관리 → 의존성 확인 → 다음 worker / `supervisor_replanner` / `final_answer`로 route.
- **SupervisorReplanner** — 전체 재계획이 아니라, gate가 `PLAN_REPAIR_REQUIRED`를 남긴 실패 task만 patch해 targeted rerun.
- **SQLAgent** (`manufacturing_agent/agents/sql_agent.py`) — Text-to-SQL. 대상 `failure_history` 단일 테이블, **SELECT-only**, DDL/DML/PRAGMA/다중 statement 금지, `LIMIT` 필수, `EXPLAIN QUERY PLAN` 검증 후 readonly 실행.
- **EvidenceAgent** — ChromaDB RAG. `OK`/`EMPTY`/`LOW_RELEVANCE`/`FAIL` 구분, docs=0이면 요약 LLM 호출 안 함, citation metadata 포함, 문서 prompt injection sanitize.
- **ContextManager** — 이전 feature를 자동 병합하지 않고, 사용자가 명시적으로 참조할 때만 모드(CURRENT_ONLY/USE_ACTIVE/PATCH_ACTIVE/SELECT_HISTORY/REFER_ACTIVE_RESULT)에 따라 active/recent `DiagnosisContext`를 사용. 입력 우선순위: **구조화 입력 > 이번 턴 NL 추출 > 이전 맥락 carry**.

더 자세한 설명: `docs/` (예: `manufacturing_agent_v6.md`, `manufacturing_agent_v6_flow.md`, `evaluation_framework.md`).

---

## 7. 개발 원칙
- 특정 시나리오 문장을 하드코딩하지 않는다.
- `orchestrator_dispatcher`에 LLM 호출을 넣지 않는다.
- Worker가 final answer를 직접 만들지 않는다.
- Gate가 worker를 직접 재실행하지 않는다.
- SQL 실패를 template fallback으로 숨기지 않는다.
- Evidence가 부족하면 부족하다고 표시한다.
- FinalAnswer에 raw SQL row/JSON/내부 state/debug trace를 출력하지 않는다.
- 위험 실행·안전장치 우회·점검 없는 재가동 승인 표현은 `output_safety_gate`에서 차단한다.

---

## 8. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `uvicorn ... 인식되지 않습니다` | 전역에 없음 → `uv run uvicorn api.main:app --reload --port 8000` |
| `OPENAI_API_KEY` 오류 | `.env` 존재/값 확인 (`cat .env`) |
| RAG citation이 안 나옴 | 임베딩 먼저: `uv run python scripts/build_chroma.py` → `ls agent_data/chroma` |
| SQL 결과 FAIL/INVALID_REQUEST | DB 확인: `sqlite3 agent_data/failure_history.sqlite '.tables'` → `failure_history` 보여야 함 |
| 프론트가 API 못 부름 | 백엔드(8000) 먼저 실행 + CORS는 5173/3000 허용(필요 시 `CORS_ORIGINS` 환경변수) |
| LangSmith 경고 많음 | `export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false` |
