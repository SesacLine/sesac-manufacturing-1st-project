# Manufacturing Agent 멀티턴/컨텍스트 코드 분석

이 문서는 `manufacturing_agent` 패키지 안에서 멀티턴 대화와 컨텍스트 재사용에 관여하는 코드의 동작 방식과 아쉬운 점을 정리한다. 기준 코드는 현재 저장소의 Python 모듈 기준이며, FastAPI/프론트엔드는 범위에서 제외하고 `manufacturing_agent` 내부 런타임, 그래프, 메모리, 컨텍스트, worker 사용 지점만 다룬다.

## 1. 전체 구조

멀티턴 처리는 크게 세 계층으로 나뉜다.

1. 저장 계층: `ConversationStore`, `RunStore`, `UserThreadRegistry`
2. 컨텍스트 해석 계층: `context_manager`, `selector`, `packer`, `normalizer`, `policy`
3. 실행 계층: LangGraph state/checkpoint, planner, worker, gate, memory writer

핵심 흐름은 다음과 같다.

```text
run_turn()
  -> make_runnable_config(user_id, thread_id)
  -> LangGraph invoke
    -> intake_gate
    -> context_manager
      -> select_context
      -> _llm_context_carryover
      -> resolve_context
      -> normalize_context
      -> pack_contexts
    -> supervisor_planner
    -> worker agents
    -> final_answer
    -> output_safety_gate
    -> memory_writer_node
      -> turns/summaries/diagnosis_contexts 저장
      -> LangGraph messages에 AI 답변 추가
```

멀티턴의 식별자는 `user_id`와 `thread_id`다. `thread_id`가 LangGraph checkpoint key이자 장기 메모리 조회 범위이므로, 같은 사용자의 다른 대화 thread와 섞이지 않도록 대부분의 조회가 `user_id + thread_id` 조건으로 묶여 있다.

## 2. 상태 계약

파일: `manufacturing_agent/contracts/state.py`, `manufacturing_agent/contracts/context.py`

`ManufacturingState`는 LangGraph의 `MessagesState`를 상속한다. 따라서 `messages`는 LangGraph checkpoint에 누적될 수 있고, 별도로 `user_id`, `thread_id`, `user_message`, `context_packet`, worker artifact들이 들어간다.

```python
class ManufacturingState(MessagesState, total=False):
    # (상속) messages: Annotated[list[BaseMessage], add_messages]
    request_id: str
    thread_id: str
    user_id: str
    user_message: str
    input_features: Optional[MachineFeatureInput]

    context_packet: Optional[ContextPacket]
    agent_contexts: dict

    prediction_result: Optional[PredictionResult]
    evidence_bundle: Optional[EvidenceArtifact]
    sql_result: Optional[SQLHistoryArtifact]

    final_answer: Optional[FinalAnswer]
```

컨텍스트 관련 핵심 모델은 다음이다.

```python
class DiagnosisContext(BaseModel):
    """진단에 실제 사용된 feature 묶음의 재사용 가능한 snapshot."""
    id: str
    turn_id: str
    user_id: str
    thread_id: str
    features: dict[str, Any] = Field(default_factory=dict)
    failure_types: list[str] = Field(default_factory=list)
    prediction_summary: str = ""
    created_at: str
    is_safe_to_reuse: bool = True
```

`DiagnosisContext`는 "대화 전체"가 아니라 "진단에 실제 사용된 feature 묶음"을 저장한다. 이 설계 덕분에 후속 질문에서 "아까 조건으로", "토크만 60으로 바꿔서" 같은 요청을 처리할 수 있다.

```python
class ContextResolution(BaseModel):
    """이번 턴에서 이전 진단 context를 어떻게 사용할지에 대한 결정."""
    mode: ContextMode = "CURRENT_ONLY"
    current_values: dict[str, Any] = Field(default_factory=dict)
    base_context_id: Optional[str] = None
    patch_values: dict[str, Any] = Field(default_factory=dict)
    resolved_features: dict[str, Any] = Field(default_factory=dict)
    changed_features: list[str] = Field(default_factory=list)
    reused_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason: str = ""
```

`ContextResolution.mode`는 멀티턴 입력값 재사용 방식의 중심이다.

- `CURRENT_ONLY`: 현재 턴에서 직접 제공한 값만 사용
- `USE_ACTIVE`: active diagnosis context 전체 재사용
- `PATCH_ACTIVE`: active diagnosis context에 현재 변경값만 덮어쓰기
- `SELECT_HISTORY`: 최근 context 중 특정 과거 context 하나 선택
- `REFER_ACTIVE_RESULT`: 재진단이 아니라 이전 결과/artifact만 참조

```python
class ContextCarryoverDecision(BaseModel):
    """멀티턴 후속 질문이 이전 artifact를 어떻게 참조하는지 LLM이 판단한 결과."""
    is_followup: bool = False
    uses_previous_prediction: bool = False
    uses_previous_evidence: bool = False
    uses_previous_sql: bool = False
    inferred_time_range: Optional[dict] = None
    referenced_artifacts: list[Literal["prediction", "sql", "evidence"]] = Field(default_factory=list)
    reason_summary: str = ""
```

`ContextResolution`이 "이전 feature snapshot을 어떻게 쓸지"를 결정한다면, `ContextCarryoverDecision`은 "이전 prediction/evidence/sql artifact를 후속 질문에 참고할지"를 결정한다.

## 3. 장기 메모리 저장소

파일: `manufacturing_agent/memory/store.py`

`ConversationStore`는 SQLite 기반 장기 메모리다. 테이블은 다음 역할을 가진다.

```sql
CREATE TABLE IF NOT EXISTS turns(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT, thread_id TEXT, role TEXT, content TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS summaries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT, thread_id TEXT, kind TEXT, content TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS diagnosis_contexts(
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    features_json TEXT NOT NULL,
    failure_types_json TEXT,
    prediction_summary TEXT,
    is_safe_to_reuse INTEGER DEFAULT 1,
    created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS context_state(
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    active_context_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, thread_id));
```

역할은 다음과 같다.

- `turns`: 사용자/assistant 원문 대화 저장
- `summaries`: prediction/evidence/sql artifact의 compact summary 저장
- `diagnosis_contexts`: 재사용 가능한 feature snapshot 저장
- `context_state`: 현재 thread의 active diagnosis context 포인터 저장

중요한 조회는 모두 `thread_id`를 받는다.

```python
def recent_turns(self, user_id, limit=8, thread_id=None) -> list[dict]:
    # thread_id가 주어지면 그 대화로만 한정한다(다른 thread 대화 누수 방지).
    # thread_id가 없을 때만 user 전체에서 조회한다.
```

```python
def latest_summary(self, user_id, kind, thread_id=None) -> Optional[str]:
    # thread_id가 주어지면 그 대화로만 한정한다(다른 thread 요약 누수 방지).
```

`save_diagnosis_context()`는 새 진단 context를 저장하면서 active context를 즉시 갱신한다.

```python
def save_diagnosis_context(self, user_id: str, thread_id: str, context: DiagnosisContext) -> None:
    ...
    c.execute(
        """INSERT INTO context_state(user_id,thread_id,active_context_id,updated_at)
           VALUES(?,?,?,?)
           ON CONFLICT(user_id,thread_id) DO UPDATE SET
               active_context_id=excluded.active_context_id,
               updated_at=excluded.updated_at""",
        (user_id, thread_id, context.id, self._now()),
    )
    self._prune_recent_contexts(c, user_id, thread_id, keep=5)
```

최근 diagnosis context는 thread당 5개만 유지한다. active context도 이 테이블 안에 있으므로, 오래된 context는 삭제될 수 있다.

## 4. user/thread 레지스트리와 삭제

파일: `manufacturing_agent/memory/registry.py`

`UserThreadRegistry`는 `users`, `threads`를 관리하고, 삭제 시 장기 메모리와 LangGraph checkpoint를 함께 지운다.

```python
_CONVERSATION_TABLES = ("turns", "machine_values", "summaries", "diagnosis_contexts", "context_state")
_CHECKPOINT_TABLES = ("checkpoints", "writes")
```

```python
def delete_thread(self, user_id: str, thread_id: str) -> bool:
    ...
    for table in _CONVERSATION_TABLES:
        c.execute(
            f"DELETE FROM {table} WHERE user_id=? AND thread_id=?",
            (user_id, thread_id),
        )
    self._delete_checkpoints(thread_id)
```

이 구조는 멀티턴 격리를 위해 중요하다. 장기 메모리만 지우고 checkpoint를 남기면 LangGraph `messages`나 runtime state가 되살아날 수 있고, 반대로 checkpoint만 지우면 저장된 summaries/context가 후속 질문에 남을 수 있다.

## 5. 런타임과 체크포인트

파일: `manufacturing_agent/runtime.py`, `manufacturing_agent/graph/build.py`

런타임은 SQLite `SqliteSaver`를 사용해 LangGraph checkpoint를 활성화한다.

```python
sql_saver = make_sqlite_saver(CHECKPOINT_DB)
app = build_graph(checkpointer=sql_saver)
```

`thread_id`는 checkpoint의 핵심 configurable 값이다.

```python
def make_runnable_config(user_id: str, thread_id: str, request_id: Optional[str] = None,
                         *, checkpoint_ns: str = "", recursion_limit: int = 50,
                         source: str = "notebook") -> RunnableConfig:
    configurable = {"thread_id": thread_id, "user_id": user_id}
```

새 턴은 매번 `make_initial_state()`로 초기 상태를 만든다.

```python
def make_initial_state(...):
    return {
        "request_id": request_id, "user_id": user_id, "thread_id": thread_id,
        "user_message": effective_msg, "input_features": input_features or None,
        "messages": [], "agent_contexts": {}, "gate_reports": [], "retry_counts": {},
        ...
    }
```

초기 상태의 `messages`는 빈 배열이지만, LangGraph checkpoint의 `messages`는 `add_messages` reducer로 누적될 수 있다. 이 때문에 `context_manager`와 `intake_gate`는 저장소의 `turns`뿐 아니라 checkpoint의 `messages`도 함께 읽는다.

재개 흐름은 `resume_turn()`이 `app.invoke(None, config=config)`를 호출하는 방식이다.

```python
def resume_turn(user_id: str, thread_id: str, request_id: str = "resume", ...):
    config = make_runnable_config(user_id, thread_id, request_id, ...)
    result = _invoke_from_checkpoint(app, config, max_resume_attempts=max_resume_attempts)
```

`graph/build.py`의 checkpoint serializer allowlist에는 컨텍스트 관련 Pydantic 모델이 명시되어 있다.

```python
CHECKPOINT_SAFE_TYPES = (
    MachineValue, DiagnosisContext, ContextState, ContextResolution,
    ContextCarryoverDecision, SupervisorPlannerDecision, SQLIntentDecision,
    ContextPacket, AgentContextPacket, PredictionResult, EvidenceArtifact, SQLQueryResult,
    SQLHistoryArtifact, FinalAnswer, ...
)
```

여기서 중요한 점은 checkpointer가 "LLM에게 넘길 최근 대화 문자열"을 직접 만드는 장치가 아니라는 것이다. checkpointer는 해당 `thread_id`의 LangGraph state snapshot을 저장하고 복원한다. 이 state 안에는 `messages`, `context_packet`, `execution_plan`, worker artifact, gate reports 같은 런타임 객체가 들어갈 수 있다. 어떤 내용을 LLM prompt로 넘길지는 각 노드가 state에서 필요한 부분을 골라 별도로 만든다.

```text
LangGraph checkpointer
  = GraphState 저장/복원
  = 노드 간 state 전달과 실패 후 resume용
  != LLM prompt에 그대로 들어가는 최근 대화 원문 묶음

ContextManager / Planner / Worker
  = 복원된 state + ConversationStore에서 필요한 부분을 읽음
  = recent_turns_summary, previous_*_summary, selected_machine_values 등으로 가공
  = 각 LLM/PydanticAI/RAG에 서로 다른 형태로 전달
```

이 allowlist는 checkpoint에 Pydantic 객체를 저장/복원하기 위한 직렬화 정책이다. PydanticAI Text-to-SQL에 넘기는 schema나 deps와는 다른 층이다. PydanticAI는 LangGraph checkpoint DB를 직접 읽지 않고, `sql_agent()`가 만든 `user_message`, `context_summary`, `task_params`, `SQLTextToSQLDeps`만 받는다.

## 5.1 checkpointer와 "최근 N턴 전달"의 실제 관계

현재 코드 기준으로 "최근 8턴을 모두 LLM에게 넘긴다"는 설명은 정확하지 않다. 비슷한 흔적은 있지만 실제 실행 경로는 다르다.

`ConversationStore.recent_turns()`의 기본값은 `limit=8`이다.

```python
def recent_turns(self, user_id, limit=8, thread_id=None) -> list[dict]:
```

하지만 `context_selector`는 이 기본값을 쓰지 않고 `RECENT_TURN_WINDOW`를 명시한다.

```python
recent = store.recent_turns(user_id, limit=RECENT_TURN_WINDOW, thread_id=thread_id)
```

그리고 `RECENT_TURN_WINDOW`는 현재 50이다.

```python
RECENT_TURN_WINDOW = 50
ASSISTANT_TURN_LIMIT = 4
```

따라서 현재 context 경로의 정책은 다음에 가깝다.

- Store에서 최근 최대 50개 turn을 가져온다.
- checkpoint `messages`에서도 최대 50개 turn을 추출한다.
- 두 출처를 합친 뒤 `(role, content)` 기준 중복 제거한다.
- 최종적으로 최근 50개 turn만 남긴다.
- prompt용 문자열을 만들 때 사용자 turn은 가능한 한 유지한다.
- assistant turn은 최근 4개만 유지한다.
- 각 turn의 글자 수 cap은 현재 없다.

정리하면, 과거의 "8턴"은 `ConversationStore.recent_turns()` 기본값 또는 이전 설계 설명의 흔적으로 볼 수 있고, 현재 멀티턴 context manager의 실제 상한은 `RECENT_TURN_WINDOW=50`이다. 다만 assistant 답변은 `ASSISTANT_TURN_LIMIT=4`로 잘린다.

주의할 점은 checkpointer 자체가 이 50/4 정책을 적용하는 것이 아니라는 점이다. checkpointer는 state를 복원하고, `context_manager`와 `intake_gate`가 복원된 `messages`를 읽어 `_messages_to_recent_turns()`와 `_summarize_recent_turns()` 정책으로 prompt용 요약을 만든다.

## 6. context_manager의 책임

파일: `manufacturing_agent/context/manager.py`

`context_manager()`는 멀티턴 컨텍스트의 중심 노드다. 주요 작업은 다음 순서다.

```python
selected = select_context(msg, user_id, conversation_store, structured, thread_id=thread_id)
selected["previous_prediction_result"] = prev_pred
selected["previous_prediction_summary"] = _summary_from_artifact("prediction", prev_pred) or selected.get("previous_prediction_summary")
selected["previous_evidence_summary"] = _summary_from_artifact("evidence", prev_ev) or selected.get("previous_evidence_summary")
selected["previous_sql_summary"] = _summary_from_artifact("sql", prev_sql) or selected.get("previous_sql_summary")

checkpoint_turns = _messages_to_recent_turns(state.get("messages", []), limit=RECENT_TURN_WINDOW)
if checkpoint_turns:
    selected["recent_turns"] = (selected.get("recent_turns") or []) + checkpoint_turns
selected["recent_turns"] = _dedup_turns(selected.get("recent_turns") or [])[-RECENT_TURN_WINDOW:]
selected["context_carryover"] = _llm_context_carryover(msg, selected)
selected["context_resolution"] = resolve_context(msg, selected)
merged, warnings = normalize_context(selected)
packet, agent_ctx = pack_contexts(msg, merged, selected, warnings)
```

특징:

- 장기 메모리 `ConversationStore`에서 가져온 최근 턴과 checkpoint `messages`에서 복원된 최근 턴을 합친다.
- `(role, content)` 기준으로 중복 제거한다.
- `ContextCarryoverDecision`으로 이전 artifact 참조 여부를 판단한다.
- `ContextResolution`으로 이전 feature snapshot 재사용 여부를 판단한다.
- `ContextPacket`과 agent별 `AgentContextPacket`을 생성한다.
- 새 턴의 artifact는 초기화한다.

```python
return {
    "context_packet": packet,
    "context_resolution": selected["context_resolution"],
    "agent_contexts": agent_ctx,
    # 새 턴 runtime artifact는 packet에 이전 요약을 옮긴 뒤 초기화한다.
    "prediction_result": None,
    "evidence_bundle": None,
    "sql_result": None,
    "final_answer": None,
    ...
}
```

이 초기화는 중요하다. 이전 턴의 runtime artifact가 현재 턴 결과처럼 남는 것을 막고, 필요한 이전 정보는 `ContextPacket.previous_*_summary`로만 전달한다.

## 7. 컨텍스트 후보 선택

파일: `manufacturing_agent/context/selector.py`

`select_context()`는 판단하지 않고 후보만 모은다.

```python
def select_context(user_message: str, user_id: str, store: ConversationStore,
                   structured: Optional[dict] = None, thread_id: Optional[str] = None) -> dict:
    """현재 입력값과 재사용 가능한 진단 context 후보만 선택한다.

    feature별 최신값을 자동으로 가져오지 않는다. 이전 feature 재사용 여부는
    ContextResolution 단계에서 하나의 base DiagnosisContext를 선택한 뒤 결정한다.
    """
```

현재 값은 자연어 추출값과 구조화 입력을 합쳐 만든다.

```python
nl_vals = extract_machine_values(user_message)
structured = structured or {}
current_vals = {**nl_vals, **structured}
```

후보로 가져오는 것은 다음이다.

```python
recent = store.recent_turns(user_id, limit=RECENT_TURN_WINDOW, thread_id=thread_id)
clean_recent = [t for t in recent if not detect_injection(t["content"])]
active_context = store.get_active_context(user_id, thread_id) if thread_id else None
recent_contexts = store.get_recent_contexts(user_id, thread_id, limit=5) if thread_id else []
```

주목할 점은 `latest_machine_values()`를 쓰지 않는다는 점이다. 코드 주석도 "ContextManager는 이 값을 prediction input 보완에 사용하지 않는다"라고 되어 있다. 즉, 과거 feature별 최신값을 임의로 조합하지 않고, 반드시 하나의 `DiagnosisContext`를 기준으로만 재사용한다.

## 8. 최근 대화 요약 정책

파일: `manufacturing_agent/context/packer.py`

최근 대화는 다음 정책을 따른다.

```python
RECENT_TURN_WINDOW = 50
ASSISTANT_TURN_LIMIT = 4
```

```python
def _summarize_recent_turns(..., user_all: bool = False,
                            assistant_limit: Optional[int] = ASSISTANT_TURN_LIMIT) -> str:
    """최근 대화를 'role:content' 한 줄 형태로 이어붙인다.
    chars=None(기본)이면 원문 전체를 자르지 않고 그대로 넣는다(개행만 공백으로 평탄화).
    user_all=True이면 사용자(user) 턴은 모두 유지하고, AI(assistant) 답변만 최근 assistant_limit개로 제한한다(순서 보존)."""
```

`user_all=True`일 때 사용자 발화는 윈도우 내 전부 유지하고 assistant 답변만 최근 4개로 제한한다. 의도 추적에는 사용자 발화가 중요하고, assistant 답변은 길어 토큰을 많이 쓰기 때문이다.

아쉬운 점은 `chars=None`이라 원문이 길면 그대로 들어간다는 것이다. `RECENT_TURN_WINDOW=50`이고 사용자 발화 전체를 유지하므로, 긴 사용자 입력이 반복되면 planner/context LLM 프롬프트가 커질 수 있다.

또 하나의 주의점은 "최근 대화 요약"이라는 이름이 실제로는 LLM summary가 아니라 문자열 concat이라는 점이다.

```python
return " | ".join(f"{t['role']}:{_body(t['content'])}" for t in selected)
```

즉, 현재 구현은 다음처럼 작동한다.

- 의미 요약을 새로 생성하지 않는다.
- role/content를 한 줄 문자열로 평탄화한다.
- user turn은 많이 남기고 assistant turn만 4개로 줄인다.
- turn별 길이 제한은 없다.

따라서 "RAG 품질 때문에 대화 내용을 잘라서 보낸다"는 말은 반은 맞고 반은 다르다. 최근 대화 전체를 RAG 검색 쿼리에 그대로 넣지는 않는다. 하지만 planner/context/sql에는 `recent_turns_summary`라는 concat 문자열이 들어갈 수 있다. Evidence RAG는 검색 품질을 위해 이 문자열을 검색 쿼리에는 넣지 않고, 요약 생성 LLM에만 참고로 넣는다.

## 9. artifact carryover 판단

파일: `manufacturing_agent/context/packer.py`

`_llm_context_carryover()`는 현재 발화가 이전 prediction/sql/evidence artifact를 참조하는지 판단한다.

프롬프트의 핵심 제한은 다음이다.

```python
"너는 task planner가 아니다. SQL 조회 필요 여부, 문서 검색 필요 여부, worker task 분해는 SupervisorPlanner가 담당한다. "
"현재 질문이 이전 prediction/sql/evidence artifact를 참조하는지만 referenced_artifacts와 uses_previous_* 필드로 표시한다. "
```

입력 payload는 현재 질문, 최근 대화 요약, 이전 artifact summary다.

```python
payload = {
    "current_user_message": user_message,
    "recent_turns_summary": recent_summary,
    "previous_prediction_summary": selected.get("previous_prediction_summary"),
    "previous_evidence_summary": selected.get("previous_evidence_summary"),
    "previous_sql_summary": selected.get("previous_sql_summary"),
}
```

후처리에서 `is_followup=false`면 모든 이전 artifact 사용 플래그를 강제로 false로 만든다.

```python
if not decision.is_followup:
    decision = decision.model_copy(update={
        "uses_previous_prediction": False,
        "uses_previous_evidence": False,
        "uses_previous_sql": False,
        "referenced_artifacts": [],
    })
```

이 결정은 planner, evidence, sql, final answer에 전달된다.

## 10. feature context resolution 판단

파일: `manufacturing_agent/context/packer.py`

`resolve_context()`는 현재 입력값과 기존 `DiagnosisContext` 후보를 보고 이번 턴에서 사용할 feature 묶음을 결정한다.

LLM에 주는 prompt는 중요한 제약을 명시한다.

```python
"CURRENT_ONLY는 현재 사용자가 직접 말한 값만 쓴다. 이전 feature 자동 보완은 금지다. "
"USE_ACTIVE는 사용자가 방금/아까/같은 조건/이전 입력값 기준이라고 명시한 경우 active context 전체를 쓴다. "
"PATCH_ACTIVE는 사용자가 특정 값만 바꾸라고 명시한 경우 active context 하나에 현재 변경값만 덮어쓴다. "
"SELECT_HISTORY는 recent_contexts 중 사용자가 특정 과거 조건 하나를 지칭한 경우만 쓴다. 여러 context를 섞지 않는다. "
"REFER_ACTIVE_RESULT는 재진단이 아니라 방금 결과/고장 유형/근거/이력만 참조하는 경우다. "
```

LLM 결정 후 코드가 한 번 더 안전하게 보정한다.

```python
if mode in {"USE_ACTIVE", "PATCH_ACTIVE"} and not base:
    warnings.append("재사용할 active 진단 context가 없어 현재 입력만 사용합니다.")
    mode = "CURRENT_ONLY"
```

`PATCH_ACTIVE`와 `SELECT_HISTORY`는 현재 값이 있는데 LLM이 patch 값을 비워두면 현재 값을 patch로 사용한다.

```python
if mode in {"PATCH_ACTIVE", "SELECT_HISTORY"} and current_values and not patch_values:
    patch_values = current_values
```

모드별 최종 `resolved_features`는 다음 로직으로 만들어진다.

```python
if mode == "CURRENT_ONLY":
    resolved = current_values
elif mode == "REFER_ACTIVE_RESULT":
    resolved = {}
elif mode == "USE_ACTIVE" and base:
    resolved = dict(base.features or {})
elif mode in {"PATCH_ACTIVE", "SELECT_HISTORY"} and base:
    resolved = dict(base.features or {})
    for key, value in patch_values.items():
        resolved[key] = value
```

좋은 점은 "feature별 최신값을 섞지 않는다"는 점이다. 제조 진단에서 서로 다른 시점의 센서값을 섞는 것은 위험할 수 있는데, 현재 코드는 active context 하나 또는 selected context 하나만 base로 삼는다.

## 11. normalizer와 MachineValue source

파일: `manufacturing_agent/context/normalizer.py`

`normalize_context()`는 `ContextResolution.resolved_features`를 `MachineValue`로 바꾼다.

```python
for name, val in (resolution.resolved_features or {}).items():
    is_current = name in current_keys and (resolution.mode == "CURRENT_ONLY" or name in changed)
    if is_current:
        source = "current"
    elif name in reused:
        source = "active_context" if resolution.mode in {"USE_ACTIVE", "PATCH_ACTIVE"} else "history_context"
    else:
        source = "context"
    merged[name] = _machine_value_from_context(name, val, is_current=is_current, source=source)
```

이 결과가 prediction input과 final answer의 "사용된 입력값" 설명에 쓰인다.

아쉬운 점은 `is_stale`가 항상 `False`라는 것이다.

```python
return MachineValue(name=name, value=val, source=source, is_current=is_current, is_stale=False)
```

`CONTEXT_RULES`에는 "오래된 센서값은 stale 표시한다"가 있지만, 실제 `DiagnosisContext`에는 timestamp가 context 단위로만 있고 feature별 timestamp가 없어서 stale 판단이 구현되지 않았다.

## 12. ContextPacket과 agent별 context

파일: `manufacturing_agent/context/packer.py`

`pack_contexts()`는 공통 `ContextPacket`과 agent별 `AgentContextPacket`을 만든다.

```python
packet = ContextPacket(
    current_question=user_message,
    recent_turns_summary=recent_summary,
    current_values=selected.get("current_values") or {},
    context_resolution=resolution,
    selected_machine_values=merged,
    previous_prediction_result=selected.get("previous_prediction_result"),
    previous_prediction_summary=selected.get("previous_prediction_summary"),
    previous_evidence_summary=selected.get("previous_evidence_summary"),
    previous_sql_summary=selected.get("previous_sql_summary"),
    context_carryover=carry,
    user_constraints=user_constraints,
    context_warnings=warnings,
)
```

agent별로 전달되는 정보는 다르다.

```python
"prediction_agent": AgentContextPacket(
    selected_context={"features": feats, "missing": missing,
                      "sources": {k: v.source for k, v in merged.items()},
                      "stale": [k for k, v in merged.items() if v.is_stale], **context_meta})
```

```python
"evidence_agent": AgentContextPacket(
    selected_context={"warnings": warnings, "recent_summary": recent_summary, **context_meta},
    prior_results=prior_results)
```

```python
"sql_agent": AgentContextPacket(
    selected_context={"recent_summary": recent_summary, "failure_history_only": True, **context_meta},
    prior_results=prior_results)
```

prediction agent는 feature 중심, evidence/sql/final은 최근 대화 요약과 previous artifact 중심으로 받는다.

## 13. planner에서 context 사용

파일: `manufacturing_agent/graph/planner.py`

`SupervisorPlanner`는 컨텍스트를 보고 필요한 worker task를 고른다. 다만 `context_carryover`와 previous summaries는 참고 맥락이며, 실제 task 판단은 planner가 별도로 한다.

```python
def _supervisor_planner_payload(state: ManufacturingState) -> dict:
    packet = state.get("context_packet")
    carry = packet.context_carryover if packet else None
    ...
    return {
        "user_message": state.get("user_message", ""),
        "has_structured_input_features": bool(structured),
        "input_features": structured or None,
        "recent_turns_summary": packet.recent_turns_summary if packet else "",
        "available_previous_prediction_summary": packet.previous_prediction_summary if packet else None,
        "available_previous_evidence_summary": packet.previous_evidence_summary if packet else None,
        "available_previous_sql_summary": packet.previous_sql_summary if packet else None,
        "previous_prediction_summary": packet.previous_prediction_summary if (packet and carry and carry.uses_previous_prediction) else None,
        "previous_evidence_summary": packet.previous_evidence_summary if (packet and carry and carry.uses_previous_evidence) else None,
        "previous_sql_summary": packet.previous_sql_summary if (packet and carry and carry.uses_previous_sql) else None,
        "current_constraints": packet.user_constraints if packet else {},
        "context_carryover": carry.model_dump() if carry else None,
    }
```

좋은 점:

- `available_previous_*_summary`와 `previous_*_summary`를 나눠 전달한다.
- carryover가 true인 artifact만 명시적 previous로 전달한다.
- 구조화 입력이 있으면 prediction task를 강제한다.

아쉬운 점:

- Planner LLM과 ContextManager LLM이 모두 멀티턴 의도를 해석한다. `ContextCarryoverDecision`이 planner 판단을 돕지만, 두 LLM 판단이 엇갈릴 수 있다.
- `available_previous_*_summary`가 항상 들어가므로, prompt상 "참고 맥락"이라고 해도 LLM이 과거 artifact를 과도하게 반영할 가능성이 있다.

## 14. prediction agent에서 context 사용

파일: `manufacturing_agent/agents/prediction_agent.py`

prediction agent는 `agent_contexts["prediction_agent"]`의 `features`를 기본 입력으로 사용한다.

```python
ctx = state["agent_contexts"]["prediction_agent"]
packet = state.get("context_packet")
resolution = packet.context_resolution if packet else None
feats = dict(ctx.selected_context.get("features", {}))
```

구조화 센서 입력은 context 해석보다 우선한다.

```python
_structured = state.get("input_features")
if _structured is not None:
    _sd = _structured.to_features() if hasattr(_structured, "to_features") else (...)
    feats.update({k: v for k, v in _sd.items() if v is not None})
```

결과에는 context mode와 재사용/변경 feature가 들어간다.

```python
PredictionResult(
    ...
    context_mode=context_mode,
    base_context_id=base_context_id,
    changed_features=changed_features,
    reused_features=reused_features,
)
```

또한 limitations에 context 사용 방식을 사용자에게 설명할 수 있는 문구를 붙인다.

```python
elif context_mode == "PATCH_ACTIVE":
    limitations.append(f"이전 진단 context를 기준으로 {changed}만 변경해 판단했습니다.")
elif context_mode == "USE_ACTIVE":
    limitations.append("사용자가 명시적으로 참조한 이전 진단 context를 기준으로 판단했습니다.")
```

## 15. evidence agent에서 context 사용

파일: `manufacturing_agent/agents/evidence_agent.py`

evidence agent는 이전 evidence/sql summary가 후속 질문에서 참조된 경우 검색 질문에 넣는다.

```python
prior = ctx.prior_results or {}
prior_context = []
if prior.get("is_followup") and prior.get("evidence_summary"):
    prior_context.append(f"이전 문서 근거 요약: {prior['evidence_summary']}")
if prior.get("is_followup") and prior.get("sql_summary"):
    prior_context.append(f"이전 SQL 이력 요약: {prior['sql_summary']}")
if prior_context:
    question = f"{question}\n\n[이전 턴 컨텍스트]\n" + "\n".join(prior_context)
```

반면 최근 대화 요약은 검색 쿼리가 아니라 evidence summary LLM에만 넣는다.

```python
# 최근 대화 원문은 요약 생성 LLM에만 참고로 주입한다.
# 검색 쿼리(question)에는 넣지 않아 retrieval 임베딩 오염을 막는다.
recent_summary = (getattr(ctx, "selected_context", None) or {}).get("recent_summary") or ""
conversation_block = (
    "\n\n[최근 대화 맥락 — 사용자 의도 파악용 참고. 근거 인용은 아래 citation 문서에서만 한다]\n" + recent_summary
    if recent_summary else ""
)
```

이 부분은 좋은 설계다. 검색 임베딩에 전체 대화가 들어가면 검색이 흐려질 수 있는데, 현재 코드는 prior artifact summary만 검색 질문에 넣고 최근 대화는 요약 생성 단계에서만 참고시킨다.

아쉬운 점은 prior artifact summary가 길이 제한 없이 compact 함수 결과 그대로 들어갈 수 있다는 것이다. 특히 SQL summary는 sample rows를 포함하므로 복합 질의가 많으면 검색 질문이 길어질 수 있다.

### 15.1 RAG에 전달되는 것과 잘리는 것

Evidence RAG 경로는 LangGraph 내부 state 전달과 다르다. `evidence_agent()`는 GraphState 전체나 checkpointer 전체를 RAG에 넘기지 않는다. RAG에는 `question`, `profile`, `prediction`, `retrieve_k`만 들어간다.

```python
result = rag_search(question=question, profile=profile, prediction=pred, retrieve_k=k)
```

`question`에는 기본적으로 현재 사용자 질문이 들어간다. 다만 후속 질문이 이전 artifact를 참조한다고 판단되면 이전 evidence/sql summary가 덧붙을 수 있다.

```python
if prior.get("is_followup") and prior.get("evidence_summary"):
    prior_context.append(f"이전 문서 근거 요약: {prior['evidence_summary']}")
if prior.get("is_followup") and prior.get("sql_summary"):
    prior_context.append(f"이전 SQL 이력 요약: {prior['sql_summary']}")
if prior_context:
    question = f"{question}\n\n[이전 턴 컨텍스트]\n" + "\n".join(prior_context)
```

반대로 `recent_turns_summary`는 검색 쿼리에 넣지 않는다. 코드 주석이 이 의도를 명확히 말한다.

```python
# 최근 대화 원문은 요약 생성 LLM에만 참고로 주입한다.
# 검색 쿼리(question)에는 넣지 않아 retrieval 임베딩 오염을 막는다.
```

RAG 내부의 절단/품질 정책은 `services/rag_service.py`에 있다.

```python
def rag_search(question: str, profile: str, prediction: Optional[PredictionResult] = None,
               retrieve_k: int = 16, top_k: int = 4) -> dict:
```

`evidence_agent()`는 profile에 따라 후보 검색 개수를 바꾼다.

```python
k = 20 if profile == "safety_procedure_rag" else 16
if feedback:
    profile = "fallback_broad"
    k = 8
```

RAG pipeline은 다음 순서로 품질을 제한한다.

```text
build_query()
  - Mode A: 현재 question 그대로 search_query
  - Mode B: prediction failure_types/cause_features 기반 tag 확장

retrieve_stage(k)
  - Chroma vector search 후보 k개
  - profile별 type/source policy 적용
  - prediction 기반 doc whitelist 적용 가능

rank_evidence(top_k=4)
  - score 내림차순
  - (source, chunk_index) 중복 제거
  - 최종 최대 4개 문서

rag_search()
  - MIN_EVIDENCE_SCORE 미만이면 LOW_RELEVANCE
```

관련 코드:

```python
def rank_evidence(hits: list[dict], top_k: int = 3) -> list[dict]:
    ...
    if len(ranked) >= top_k:
        break
```

```python
MIN_EVIDENCE_SCORE = float(os.environ.get("MIN_EVIDENCE_SCORE", "0.2"))
relevant = [d for d in ranked if float(d.get("score", 0.0)) >= MIN_EVIDENCE_SCORE]
if not relevant:
    return {"status": "LOW_RELEVANCE", ...}
```

문서 내용도 LLM에 그대로 무제한 전달되지 않는다.

```python
def _clean_evidence_snippet(text: str, limit: int = 360) -> str:
    ...
    return cleaned[:limit].strip()
```

```python
def build_citation_aware_docs(docs: list[dict], citations: list[dict]) -> list[dict]:
    ...
    "text": str(doc.get("text") or "")[:1800],
```

prompt injection 의심 문서 텍스트는 더 짧게 sanitize된다.

```python
def _redact_retrieved_instruction_text(text: str) -> str:
    safe = RETRIEVED_DOC_INJECTION_RE.sub("[UNTRUSTED_INSTRUCTION_REMOVED]", text or "")
    return safe[:1200]
```

따라서 "RAG 품질 때문에 잘라서 보낸다"는 부분은 주로 다음을 뜻한다.

- 최근 대화 전체를 검색 쿼리에 넣지 않는다.
- 검색 후보는 profile에 따라 16/20/8개로 제한한다.
- 최종 근거 문서는 기본 최대 4개로 제한한다.
- 낮은 score 문서는 `LOW_RELEVANCE`로 처리한다.
- citation snippet과 LLM 전달용 문서 text를 길이 제한한다.

## 16. SQL agent에서 context 사용

파일: `manufacturing_agent/agents/evidence_agent.py`

SQL 관련 코드는 같은 파일 안의 `sql_agent()`에 있다. `_build_sql_context_summary()`가 `ContextPacket`을 SQL Text-to-SQL용 context로 바꾼다.

```python
def _build_sql_context_summary(packet: Optional[ContextPacket], state: ManufacturingState) -> str:
    if not packet:
        return ""
    blocks = []
    carry = packet.context_carryover
    blocks.append("현재 SQL DB는 failure_history 단일 테이블이다. 설비/자산 식별자 조건은 사용하지 않는다.")
    if packet.recent_turns_summary:
        blocks.append(f"참고용 최근 대화(thread context): {packet.recent_turns_summary}")
```

이전 artifact는 carryover 여부에 따라 label만 달라진다.

```python
if packet.previous_sql_summary:
    label = "현재 질문이 참조한 이전 SQL 이력 artifact" if (carry and carry.uses_previous_sql) else "참고용 이전 SQL 이력 artifact"
    blocks.append(f"{label}: {packet.previous_sql_summary}")
```

현재 prediction이 있으면 failure type을 SQL 문맥에 넣어 유사 사례 조회에 활용할 수 있게 한다.

```python
if state.get("prediction_result"):
    pred = state.get("prediction_result")
    blocks.append(f"현재 prediction failure_types: {getattr(pred, 'failure_types', [])}; cause_features: {getattr(pred, 'cause_features', [])}")
```

time range carryover는 `_sanitize_time_range()`로 기준일과 맞지 않는 날짜 환각을 보정한다.

```python
if constraints.get("time_range"):
    constraints["time_range"] = _sanitize_time_range(
        constraints["time_range"], SQL_REFERENCE_DATE, DEFAULT_SQL_DEPS.default_time_window_days)
```

좋은 점:

- SQL DB가 `failure_history` 단일 테이블이라는 제약을 context summary와 Text-to-SQL system prompt 양쪽에 넣는다.
- 기준일 `SQL_REFERENCE_DATE`를 명시하고 time range를 보정한다.
- 설비/자산 식별자 조건을 만들지 말라는 제한이 반복해서 들어간다.

아쉬운 점:

- `SQL_REFERENCE_DATE = "2026-06-21"`이 코드 상수로 고정되어 있다. 현재 날짜나 데이터셋 기준일을 명확히 분리하지 않으면 시간이 지나면서 "최근 30일" 해석이 어긋날 수 있다.
- previous summary가 carryover가 false여도 "참고용"으로 들어간다. Text-to-SQL LLM이 참고용 artifact를 실제 조건처럼 사용할 위험이 있다.

### 16.1 PydanticAI에 전달되는 context는 LangGraph state와 다르다

SQL worker는 LangGraph node이지만, PydanticAI agent에게 LangGraph state 전체를 넘기지는 않는다. `sql_agent()`가 state에서 필요한 조각을 뽑아 별도 prompt/deps로 변환한다.

LangGraph node 내부에서 만드는 값:

```python
packet = state.get("context_packet")
task_params = get_active_task_params(state, expected_type="sql")
context_summary = _build_sql_context_summary(packet, state)
msg = state.get("user_message", "")
planned_query_types = [q for q in (task_params.get("query_types") or []) if q in allowed_qtypes]
text_deps = _text_to_sql_deps_from_agent_deps(deps, planned_query_types)
```

PydanticAI runner에 실제로 넘어가는 값:

```python
response = _normalize_text_to_sql_response(runner(
    user_message=msg,
    context_summary=context_summary,
    task_params=task_params,
    deps=text_deps,
))
```

PydanticAI prompt는 `_text_to_sql_prompt()`에서 만들어진다.

```python
def _text_to_sql_prompt(user_message: str, context_summary: str, task_params: dict) -> str:
    return (
        f"[사용자 질문]\n{user_message}\n\n"
        f"[Context summary]\n{context_summary or '(none)'}\n\n"
        f"[Supervisor SQL task params]\n{json.dumps(task_params or {}, ensure_ascii=False)}\n\n"
        "중요: failure_history는 고장 사례 단위 테이블이다. 식별자 조건을 만들지 말고, failure_type/component/symptom/root_cause/action 중심으로 조회하라.\n\n"
        "SQLSuccess 또는 SQLInvalidRequest structured output으로 답하라. "
        "SQLSuccess라면 queries에 실행할 SELECT SQL을 모두 담아라."
    )
```

즉, PydanticAI가 받는 것은 다음뿐이다.

- 현재 사용자 질문 `user_message`
- `_build_sql_context_summary()`가 만든 문자열 context
- SupervisorPlanner가 만든 SQL task params
- `SQLTextToSQLDeps`: schema text, allowed tables, reference date, max rows, readonly, planned query types

PydanticAI가 받지 않는 것:

- LangGraph checkpoint DB
- 전체 `ManufacturingState`
- `messages` 원본 리스트 전체
- RAG documents 전체
- final answer
- gate reports 전체

`SQLTextToSQLDeps`는 PydanticAI system prompt와 output validator에서 쓰인다.

```python
class SQLTextToSQLDeps(BaseModel):
    db_uri: str
    schema_text: str
    allowed_tables: list[str]
    reference_date: str
    default_time_window_days: int = 30
    max_rows: int = 50
    readonly: bool = True
    supervisor_query_types: list[str] = Field(default_factory=list)
```

그리고 PydanticAI output은 다시 LangGraph artifact로 돌아오기 전에 검증된다.

```python
cleaned_sql = _validate_text_to_sql_query(q.sql_query, ctx.deps)
```

이 차이가 중요하다. LangGraph 내부에서는 state object들이 노드 사이를 이동하지만, PydanticAI Text-to-SQL은 별도 agent 호출이며 typed deps/prompt만 받는다. 따라서 checkpointer에 저장된 전체 대화가 PydanticAI에 자동 전달되는 일은 없다.

## 17. memory_writer_node의 저장 로직

파일: `manufacturing_agent/nodes/memory_writer_node.py`

`memory_writer_node()`는 최종 답변 뒤에 대화와 artifact summary를 저장한다.

```python
conversation_store.add_turn(user_id, "user", msg, thread_id=thread_id)
if fa:
    conversation_store.add_turn(user_id, "assistant", fa.answer, thread_id=thread_id)
```

prediction summary 저장:

```python
if pred and pred.status in {"OK", "PARTIAL"} and pred.summary:
    conversation_store.add_summary(user_id, "prediction", pred.summary, thread_id=thread_id)
```

재사용 가능한 diagnosis context 저장 조건:

```python
def _should_save_diagnosis_context(state: ManufacturingState, pred: Optional[PredictionResult], packet: Optional[ContextPacket]) -> bool:
    if not pred or pred.status not in {"OK", "PARTIAL"}:
        return False
    if not packet or not packet.selected_machine_values:
        return False
    dec = state.get("input_decision")
    if dec and getattr(dec, "blocked", False):
        return False
    if detect_injection(state.get("user_message", "")):
        return False
    return True
```

저장되는 feature는 이번 턴에서 최종 선택된 `selected_machine_values`다.

```python
features = {k: v.value for k, v in packet.selected_machine_values.items() if v.value is not None}
diag = DiagnosisContext(
    id=f"diag-{uuid.uuid4().hex}",
    turn_id=state.get("request_id") or "unknown-turn",
    user_id=user_id,
    thread_id=thread_id,
    features=features,
    failure_types=list(pred.failure_types or []),
    prediction_summary=pred.summary or "",
    created_at=conversation_store._now(),
    is_safe_to_reuse=True,
)
conversation_store.save_diagnosis_context(user_id, thread_id, diag)
```

evidence/sql은 compact summary로 저장된다.

```python
if ev:
    conversation_store.add_summary(user_id, "evidence", _compact_evidence_artifact_for_memory(ev), thread_id=thread_id)
if sql:
    conversation_store.add_summary(user_id, "sql", _compact_sql_artifact_for_memory(sql), thread_id=thread_id)
```

마지막으로 LangGraph checkpoint의 `messages`에 assistant 답변을 추가한다.

```python
return {"messages": [AIMessage(content=fa.answer)]} if fa else {}
```

아쉬운 점:

- user turn은 `intake_gate`에서 checkpoint messages에 들어가고, assistant turn은 `memory_writer_node`에서 들어간다. 반면 장기 저장소에는 둘 다 `memory_writer_node`에서 저장된다. 두 저장 경로가 달라 중복/불일치 가능성이 있다.
- `memory_writer_node`가 실행되기 전에 실패하면 장기 저장소에는 user turn이 남지 않을 수 있지만 checkpoint에는 남을 수 있다.
- diagnosis context 저장 시 현재 context가 `REFER_ACTIVE_RESULT`라면 `selected_machine_values`가 비어 있어 저장되지 않는다. 의도상 맞지만, "이전 결과만 참조한 후속 질문"도 active context 사용 흔적은 남지 않는다.

## 18. intake gate와 checkpoint context

파일: `manufacturing_agent/gates/intake_gate.py`

intake gate는 대화 초반에 service/safety 판단을 한다. 짧은 후속 질문을 제조 도메인으로 인정하기 위해 checkpoint messages 요약을 LLM intake에 넘긴다.

```python
checkpoint_context = _summarize_recent_turns(
    _messages_to_recent_turns(state.get("messages", []), limit=RECENT_TURN_WINDOW),
    user_all=True,
)
intake = _llm_intake(msg, context_summary=checkpoint_context)
```

좋은 점은 "그건 왜?" 같은 짧은 후속 질문을 out-of-scope로 오판하지 않도록 recent context를 제공한다는 것이다.

아쉬운 점은 여기서는 `ConversationStore.recent_turns()`를 보지 않고 checkpoint `messages`만 본다는 것이다. checkpoint가 삭제됐지만 장기 메모리가 남아 있거나, 반대로 checkpoint만 있고 장기 메모리 저장이 실패한 상황에서 intake/context_manager가 서로 다른 맥락을 보게 된다.

## 19. 그래프 배선상 멀티턴 위치

파일: `manufacturing_agent/graph/build.py`

컨텍스트는 intake 통과 후 항상 planner 전에 실행된다.

```python
g.add_edge(START, "intake_gate")
g.add_conditional_edges("intake_gate", route_after_intake,
                        {"context_manager": "context_manager", "final_answer": "final_answer"})
g.add_edge("context_manager", "supervisor_planner")
```

이 배선은 적절하다. planner가 task를 정하기 전에 `ContextPacket`이 만들어져야 "방금 그 이력", "아까 조건에서 토크만 변경" 같은 후속 질문을 올바르게 해석할 수 있다.

단, intake에서 차단되면 context_manager를 거치지 않고 final_answer로 간다. 그래서 차단 답변은 이전 context를 거의 쓰지 않는다. 안전 차단에는 맞지만, "왜 차단됐는지 이전 조건과 연결해서 설명" 같은 UX는 제한된다.

## 20. 현재 구현의 장점

- `user_id + thread_id` 기준으로 장기 메모리를 격리한다.
- LangGraph checkpoint와 SQLite 장기 메모리를 모두 사용해 재개성과 장기 context를 분리한다.
- feature별 최신값을 임의 조합하지 않고, 단일 `DiagnosisContext`만 재사용한다.
- `CURRENT_ONLY`, `USE_ACTIVE`, `PATCH_ACTIVE`, `SELECT_HISTORY`, `REFER_ACTIVE_RESULT`로 멀티턴 feature 재사용 의미를 명시적으로 모델링한다.
- 이전 artifact 참조 판단(`ContextCarryoverDecision`)과 feature snapshot 재사용 판단(`ContextResolution`)을 분리했다.
- `context_manager`가 새 턴 시작 시 이전 runtime artifact를 초기화하여 stale artifact가 현재 결과처럼 남는 일을 줄인다.
- evidence 검색에서 최근 대화 전체를 retrieval query에 넣지 않아 임베딩 검색 오염을 줄인다.
- SQL agent는 context를 받더라도 SELECT-only, allowed table, LIMIT, EXPLAIN 검증을 통과해야 실행한다.
- memory writer는 prompt injection 입력이나 blocked 입력에서 diagnosis context를 저장하지 않는다.

## 21. 아쉬운 점과 개선 제안

### 21.1 멀티턴 판단 LLM이 두 번 갈라져 있다

`_llm_context_carryover()`와 `_llm_context_resolution()`이 각각 LLM을 호출한다. 이후 `SupervisorPlanner`도 다시 멀티턴 의도를 해석한다. 판단 책임은 나뉘어 있지만, 실제로는 세 LLM이 같은 후속 질문을 각자 해석한다.

문제 가능성:

- carryover는 `uses_previous_sql=true`인데 planner가 `needs_sql=false`로 볼 수 있다.
- context resolution은 `REFER_ACTIVE_RESULT`인데 planner가 prediction을 실행할 수 있다.
- planner가 previous artifact를 "참고용"이 아니라 조건으로 사용할 수 있다.

개선:

- `ContextDecision` 같은 단일 상위 결정 모델을 두고, carryover/resolution/planner payload를 한 번의 판단 결과에서 파생한다.
- 최소한 planner payload에 `context_resolution.mode`와 `reason`을 넣어 planner가 feature 재사용 결정을 명시적으로 알게 한다.

### 21.2 previous summary가 carryover false여도 worker에 들어간다

`ContextPacket`에는 항상 `previous_*_summary`가 들어가고, SQL context summary도 carryover false인 경우 "참고용" label로 넣는다.

문제 가능성:

- LLM이 "참고용" 이전 SQL 결과를 현재 질의 조건처럼 사용할 수 있다.
- 사용자가 새 질문을 했는데 이전 evidence summary가 답변 방향을 오염시킬 수 있다.

개선:

- `carry.uses_previous_*`가 false면 worker prompt에는 previous summary를 넣지 않고 planner payload의 `available_previous_*` 정도에만 제한한다.
- 정말 참고가 필요하면 `non_binding_context` 필드로 분리하고 system prompt에서 "조건으로 사용 금지"를 더 강하게 검증한다.

### 21.3 최근 대화 요약이 원문 concat이다

`_summarize_recent_turns()`는 `chars=None`이면 원문을 자르지 않는다. 사용자 턴은 최대 50개까지 전부 들어간다.

문제 가능성:

- 긴 사용자 입력이 반복되면 context/planner/sql 프롬프트가 커진다.
- 오래된 사용자 발화가 최신 의도보다 과하게 반영될 수 있다.

개선:

- 최근 N턴 원문 + 오래된 구간 rolling summary 구조로 바꾼다.
- 사용자 턴에도 per-turn char cap을 둔다.
- `recent_turns_summary`를 문자열 하나가 아니라 구조화 배열로 유지해 role/time/source를 잃지 않게 한다.

### 21.4 stale feature 정책이 미구현이다

`MachineValue.is_stale` 필드는 있지만 항상 false다. `CONTEXT_RULES`에는 stale 표시가 있으나 실제 구현은 없다.

문제 가능성:

- 오래된 active context를 재사용해도 final answer에서 충분히 경고하지 못한다.
- 센서값의 시간 민감도가 높은 제조 진단에서 stale 여부는 중요하다.

개선:

- `DiagnosisContext`에 feature별 timestamp 또는 context age를 저장한다.
- `normalize_context()`에서 `created_at` 기준 stale 여부를 계산한다.
- prediction limitations에 "N분/시간 전 context 재사용"을 넣는다.

### 21.5 active context lifecycle이 단순하다

새 OK/PARTIAL prediction이 저장되면 무조건 active context가 된다.

문제 가능성:

- 사용자가 잠깐 다른 설비/다른 조건을 물어본 뒤 active context가 바뀐다.
- thread 안에서 여러 설비/시나리오를 다루면 "아까 조건"이 모호해진다.

개선:

- `DiagnosisContext`에 `label`, `equipment_id`, `scenario_id`, `source_question`을 추가한다.
- active context를 하나만 두지 말고 named context 또는 recent context selection UX를 제공한다.
- context switch가 감지되면 active를 바꾸기 전에 reason/warning을 남긴다.

### 21.6 장기 메모리와 checkpoint가 이중 source of truth다

최근 대화는 `ConversationStore.turns`와 LangGraph `messages` 양쪽에 저장된다. `context_manager`는 둘을 합치지만, `intake_gate`는 checkpoint messages만 본다.

문제 가능성:

- 한쪽 저장이 실패하면 노드마다 보는 맥락이 달라진다.
- 중복 제거 기준이 `(role, content)`라서 같은 문장을 다른 시점에 반복한 경우 하나로 합쳐질 수 있다.

개선:

- 최근 대화 source of truth를 하나로 정한다. 예: 장기 저장소를 기준으로 하고 checkpoint는 resume용 runtime state만 담당.
- dedup key에 timestamp 또는 turn id를 포함한다.
- intake도 `ConversationStore`를 함께 보거나, context_manager 전용 사전 context loader를 둔다.

### 21.7 `REFER_ACTIVE_RESULT`의 의미가 worker별로 약하다

`REFER_ACTIVE_RESULT`에서는 `resolved_features={}`가 된다. 이 모드는 "재진단하지 않고 이전 결과만 참조"라는 의미다.

문제 가능성:

- planner가 prediction task를 만들면 prediction은 입력 부족/skip으로 갈 가능성이 있다.
- final answer나 planner가 `context_resolution.mode`를 직접 강하게 활용하지 않는다.

개선:

- planner payload에 `context_resolution` 전체를 포함한다.
- `REFER_ACTIVE_RESULT`이면 prediction task를 기본적으로 만들지 않도록 planner normalization에 rule을 추가한다.
- final answer에서 이 모드일 때 "이번 턴은 새 진단이 아니라 이전 결과 참조"라는 표현을 일관되게 넣는다.

### 21.8 SQL 기준일이 고정 상수다

`SQL_REFERENCE_DATE = os.environ.get("MANUFACTURING_REFERENCE_DATE", "2026-06-21")`로 되어 있다.

문제 가능성:

- 시간이 지나면 "최근 30일"이 실제 현재일 기준이 아니라 데모 기준일 기준으로 해석된다.
- 데이터셋 기준일인지 서비스 현재일인지 코드만 보고 명확하지 않다.

개선:

- 변수명을 `SQL_DATASET_REFERENCE_DATE`처럼 바꾸고 문서화한다.
- API/runtime config에서 기준일을 명시적으로 주입한다.
- 실제 운영 데이터라면 현재 날짜 기준으로 계산하고, 데모 데이터라면 UI/답변에 "데이터셋 기준일"을 표시한다.

### 21.9 memory summary 압축 정책이 불균형하다

evidence/sql compact 함수는 구조적으로 일부 제한이 있지만, 문자열 길이 cap은 없다.

```python
# 길이 캡 없이 원문 그대로 보존한다(...)
return " | ".join(lines)
```

문제 가능성:

- summaries가 계속 커지면 prompt 비용이 커진다.
- sample rows에 민감하거나 불필요한 필드가 남을 수 있다.

개선:

- kind별 max chars를 둔다.
- SQL summary는 rows sample보다 집계/핵심 field만 저장한다.
- long summary는 별도 table에 raw artifact로 저장하고 prompt에는 short summary만 넣는다.

### 21.10 테스트가 PlanOps 중심이고 context 테스트가 부족하다

현재 regression test는 PlanOps/replanner 중심이다. 멀티턴 context는 LLM 호출이 많아 테스트가 어렵지만, deterministic 부분은 충분히 테스트 가능하다.

추가하면 좋은 테스트:

- `select_context()`가 thread_id별로 turns/summaries/context를 격리하는지
- `_dedup_turns()`가 store/checkpoint 중복을 제거하는지
- `resolve_context()` fallback이 LLM 실패 시 `CURRENT_ONLY`로 닫히는지
- `PATCH_ACTIVE`에서 current_values가 patch_values로 보정되는지
- `normalize_context()`가 `source=current/active_context/history_context`를 올바르게 붙이는지
- `memory_writer_node()`가 injection/blocked 입력에서 diagnosis context를 저장하지 않는지

## 22. 우선순위 높은 개선안

가장 먼저 손대면 좋은 순서는 다음이다.

1. context deterministic 테스트 추가
2. planner payload에 `context_resolution.mode/reason/base_context_id` 추가
3. worker prompt에서 carryover false인 previous summaries 제외
4. `recent_turns_summary` 길이 cap 또는 rolling summary 도입
5. `DiagnosisContext`에 source question/context age 추가
6. stale feature 표시 구현
7. checkpoint와 장기 memory의 source of truth 정리

## 23. 요약

현재 멀티턴 구조의 핵심은 `DiagnosisContext` snapshot과 `ContextResolution`이다. 단순히 "이전 대화에서 빠진 feature를 채우는" 방식이 아니라, 사용자가 명시적으로 이전 조건을 참조할 때만 active/history context 하나를 선택하거나 patch한다. 이 방향은 제조 진단 도메인에서 꽤 안전한 선택이다.

다만 현재 구현은 LLM 기반 판단 지점이 여러 개이고, previous summary가 넓게 주입되며, 최근 대화 요약과 stale 정책이 아직 거칠다. 따라서 다음 단계는 새로운 기능 추가보다 context 결정의 테스트 가능성, prompt 입력량 제어, previous artifact 주입 조건 강화, active context lifecycle 명확화에 집중하는 것이 좋다.
