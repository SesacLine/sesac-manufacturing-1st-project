# 컨텍스트/멀티턴 서브시스템 재설계 — 인터페이스 스펙 (ground truth)

> 이 문서는 병렬 구현 에이전트가 따라야 하는 **계약**이다. 여기 적힌 시그니처/필드/동작을 임의로 바꾸지 말 것.
> 범위: 컨텍스트 로직만 (비용/지연·정확도·단순화). **동시성(WAL/lock/api offload)은 이번 범위 제외(별도 작업).**
> 결정 사항: ① LLM 단일 `ContextDecision` 1콜로 통합 ② 기존 dev sqlite는 폐기 후 재생성 ③ 동시성 보류.

---

## 0. 핵심 설계 원칙 — "내부는 통합, 외부는 유지"

- **내부**: carryover + resolution 판단을 **단일 LLM 1콜(`ContextDecision`)** 로 통합하고, 자명한 턴은 **LLM 호출을 건너뛴다(short-circuit)**.
- **외부**: `ContextPacket` 의 공개 필드 모양은 **그대로 유지**한다. 즉 `context_resolution`(ContextResolution), `context_carryover`(ContextCarryoverDecision) 필드를 **`ContextDecision`에서 파생**해 계속 채운다. → 소비처(prediction_agent / planner / evidence_agent / final_answer / memory_writer)의 변경을 최소화한다.

이 원칙 덕분에 소비처는 "필드 읽는 위치"가 거의 안 바뀌고, 바뀌는 건 (a) LLM 호출 횟수, (b) stale 채움, (c) 토큰 캡, (d) 두 채널 요약 일원화뿐이다.

### 깨면 안 되는 계약(문서화된 가드레일)
1. feature **auto-merge 금지** — 단일 `DiagnosisContext` 하나만 base로 재사용.
2. `ContextMode` 5종만 사용: `CURRENT_ONLY | USE_ACTIVE | PATCH_ACTIVE | SELECT_HISTORY | REFER_ACTIVE_RESULT`.
3. `thread_id`=조회 범위, `user_id`=namespace, `run_id`=trace only. `RunnableConfig.configurable`에 도메인 데이터 금지.
4. checkpointer ↔ Store 역할 분리. `CHECKPOINT_SAFE_TYPES` allowlist에 신규 모델 등록 필수.
5. SQL은 `failure_history` 단일·SELECT-only·식별자 금지. RAG 검색 쿼리에 대화 원문 미주입.
6. `context_manager`는 task planning 안 함(그건 SupervisorPlanner).

---

## 1. 새 모델: `ContextDecision` (contracts/context.py)

carryover와 resolution을 합친 **단일 LLM 출력 + 코드 파생** 모델.

```python
class ContextDecision(BaseModel):
    """단일 LLM 1콜 결과 + 코드 후처리. carryover(이전 artifact 참조) + resolution(feature snapshot 재사용)을 통합."""
    # --- LLM이 채우는 부분 ---
    is_followup: bool = False
    referenced_artifacts: list[Literal["prediction", "sql", "evidence"]] = Field(default_factory=list)
    uses_previous_prediction: bool = False
    uses_previous_evidence: bool = False
    uses_previous_sql: bool = False
    inferred_time_range: Optional[dict] = None
    mode: ContextMode = "CURRENT_ONLY"
    base_context_id: Optional[str] = None
    patch_values: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    # --- 코드가 채우는 부분(검증/파생) ---
    current_values: dict[str, Any] = Field(default_factory=dict)
    resolved_features: dict[str, Any] = Field(default_factory=dict)
    changed_features: list[str] = Field(default_factory=list)
    reused_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_skipped: bool = False   # short-circuit으로 LLM을 건너뛰었으면 True

    # 파생 뷰 — 소비처 호환용
    def to_resolution(self) -> "ContextResolution": ...
    def to_carryover(self) -> "ContextCarryoverDecision": ...
```

- `ContextResolution`, `ContextCarryoverDecision` 는 **삭제하지 않는다.** `ContextDecision.to_resolution()/to_carryover()` 가 이들을 생성한다. `ContextPacket.context_resolution`/`context_carryover` 는 이 파생값으로 채운다.
- `DiagnosisContext` 에 stale 계산용 helper 추가:
```python
class DiagnosisContext(BaseModel):
    ...  # 기존 필드 유지
    def age_seconds(self, now_iso: str) -> Optional[float]: ...   # created_at 대비 경과초
```
- **build.py**: `CHECKPOINT_SAFE_TYPES` 튜플에 `ContextDecision` 추가.

---

## 2. ContextEngine (context/)

### 2.1 파일 역할 재편
| 파일 | 역할(변경 후) |
|------|----------------|
| `context/policy.py` | 유지. `STALE_THRESHOLD_SECONDS`(기본 `3600`), 토큰버짓 상수 추가. |
| `context/selector.py` | 유지(후보 수집). 자동 보완 금지 정책 유지. |
| `context/engine.py` (신규 또는 packer 흡수) | **단일 LLM 1콜 `decide_context()`** + short-circuit + 코드 검증. 기존 `_llm_context_carryover`+`resolve_context` 를 대체. |
| `context/normalizer.py` | `is_stale` **실제 계산**(age > 임계). |
| `context/packer.py` | `pack_contexts()` 유지하되 `ContextDecision` 입력. `_summarize_recent_turns` 에 **토큰/char 캡 + rolling**. **`build_context_summary()` 공개**(planner·evidence 공용). |
| `context/manager.py` | orchestration만: selector → engine.decide_context → normalize → pack. |

### 2.2 핵심 함수 시그니처
```python
# context/engine.py
def decide_context(user_message: str, selected: dict) -> ContextDecision:
    """selected(=select_context 결과 + 이전 요약)로부터 단일 ContextDecision 생성.
    short-circuit: active_context 없고 recent_contexts 없고 recent_turns 비고
                   이전 요약 전부 없으면 → LLM 호출 없이 CURRENT_ONLY(llm_skipped=True).
    아니면 LLM 1콜(CONTEXT_DECISION_SYS) → 코드가 mode 강등/patch 화이트리스트/resolved_features 계산."""

# context/packer.py
def build_context_summary(packet: ContextPacket, *, for_sql: bool = False,
                          prediction_result: Optional[Any] = None) -> str:
    """planner payload와 evidence_agent SQL context가 공유하는 단일 컨텍스트 요약 빌더.
    for_sql=True면 failure_history 가드 문장 + reference_date 포함."""
```

### 2.3 short-circuit 규칙 (목표 #1)
LLM `decide_context` 호출을 **건너뛰는** 조건(모두 만족):
- `selected["active_context"]` is None
- `selected["recent_contexts"]` 비어 있음
- `previous_prediction_summary` / `previous_evidence_summary` / `previous_sql_summary` 전부 falsy

> **주의(검증 후 수정)**: `recent_turns`(채팅 원문)는 **조건에서 제외**한다. context_manager 진입 시점엔 거의 항상 recent_turns ≥ 1이라 이를 넣으면 short-circuit이 사실상 발화하지 않는다. 그리고 설계 계약상 feature 값은 저장된 `DiagnosisContext`에서만 재사용하고 채팅 원문에서 자동 병합하지 않으므로, 재사용 가능한 context/artifact가 없으면 채팅 턴이 있어도 결과는 항상 CURRENT_ONLY다 → 안전하게 단락 가능. (e2e 확인: 첫 진단 턴 context-LLM 0콜, 후속 PATCH_ACTIVE 턴 1콜.)

→ `ContextDecision(mode="CURRENT_ONLY", current_values=..., resolved_features=current_values, changed_features=keys, llm_skipped=True, reason="trivial turn; no prior context")`.

### 2.4 통합 LLM 프롬프트 출력 스키마 (`CONTEXT_DECISION_SYS`)
```json
{"is_followup": bool, "referenced_artifacts": ["prediction|sql|evidence"],
 "uses_previous_prediction": bool, "uses_previous_evidence": bool, "uses_previous_sql": bool,
 "inferred_time_range": null|object,
 "mode": "CURRENT_ONLY|USE_ACTIVE|PATCH_ACTIVE|SELECT_HISTORY|REFER_ACTIVE_RESULT",
 "base_context_id": null|string, "patch_values": object, "reason": string}
```
코드 후처리(기존 `resolve_context` 검증 로직 그대로 이식):
- mode가 `USE_ACTIVE/PATCH_ACTIVE` 인데 base 없으면 → `CURRENT_ONLY` 강등 + warning.
- `SELECT_HISTORY` 인데 base_context_id가 후보에 없으면 → `CURRENT_ONLY` 강등 + warning.
- `patch_values` 는 **현재 턴 current_values 키만 허용**(`_filter_patch_values`).
- mode별 `resolved_features/changed/reused` 계산은 기존 `resolve_context`(packer.py:217-243)와 **동일 규칙**.
- carryover 정합성: `is_followup=False` 면 `uses_previous_* / referenced_artifacts` 강제 비움(기존 packer.py:99-105).
- 파싱 실패 → `CURRENT_ONLY` fallback + warning(기존과 동일).

---

## 3. memory/store.py (스키마 — 동시성 제외)

### 3.1 변경
- `add_summary(user_id, kind, content, thread_id=None, turn_id=None)` — `summaries` 테이블에 `turn_id TEXT` 컬럼 추가. 저장 시 turn_id 기록.
- `latest_summary(...)` 유지(기존 소비처 호환). **신규** `summary_by_turn(user_id, kind, turn_id, thread_id=None)` 추가(특정 과거 artifact 조회용 — "최신 1건만" 한계 완화).
- `get_active_context` / `get_recent_contexts` 반환은 그대로. `DiagnosisContext.age_seconds` 로 stale 판정은 normalizer에서.
- `add_summary` 저장 시 **content char cap**(기본 4000자) 적용.
- **동시성 보류**: WAL/busy_timeout/lock은 이번 범위 아님. 단, 스키마 변경으로 기존 sqlite는 **삭제 후 재생성**(아래 6).

### 3.2 마이그레이션
- 기존 dev DB는 폐기 → `_ensure_column` 으로 `summaries.turn_id` 보강 + 신규는 자동 생성. 폐기 전제라 `_drop_if_legacy` 패턴 재사용 가능.

---

## 4. 소비처 변경(최소) — 6곳

| 파일 | 변경 |
|------|------|
| `gates/intake_gate.py` | 변경 거의 없음(checkpoint 요약은 `_summarize_recent_turns` 캡 적용만 자동 반영). |
| `graph/planner.py` | `_supervisor_planner_payload` 의 수동 요약 조립을 **`build_context_summary(packet)`** 로 대체(중복 제거). carryover 필드는 `packet.context_carryover`(파생) 그대로 사용. |
| `agents/evidence_agent.py` | `_build_sql_context_summary` 를 **`build_context_summary(packet, for_sql=True, prediction_result=...)`** 로 대체 → A/B 채널 일원화. `_sanitize_time_range` 는 유지. |
| `agents/prediction_agent.py` | context meta 3중 fallback 체이닝 제거 → `packet.context_resolution` **단일 출처**만 사용(`ctx.selected_context` 중복 읽기 삭제). 동작 동일. |
| `nodes/final_answer_node.py` | 변경 없음(`packet.selected_machine_values` 그대로). |
| `nodes/memory_writer_node.py` | `add_summary(..., turn_id=request_id)` 로 turn_id 전달. 나머지 동일. |

---

## 5. 토큰 버짓 (목표 #1·#2) — packer
- `RECENT_TURN_WINDOW = 50` 유지(상한). `ASSISTANT_TURN_LIMIT = 4` 유지.
- 신규 `RECENT_SUMMARY_CHAR_BUDGET = 2000`: `_summarize_recent_turns` 결과가 이를 넘으면 **오래된 user 턴부터 잘라** 최신 우선 유지(말미 보존). 각 턴 본문도 `PER_TURN_CHAR_CAP = 300` 캡.
- 단일 `build_context_summary` 가 이 캡을 거친 요약만 노출.

---

## 6. 테스트 + DB 재생성 (목표: 정확도 회귀 방지)
- `tests/test_context_engine.py` (신규, pytest, 네트워크 없는 deterministic): `decide_context` 의 LLM 부분은 **monkeypatch로 가짜 응답 주입**.
  - short-circuit: prior 없음 → `llm_skipped=True`, CURRENT_ONLY.
  - PATCH_ACTIVE: base 있음 + patch_values 화이트리스트.
  - USE_ACTIVE base 없음 → CURRENT_ONLY 강등 + warning.
  - stale: 오래된 created_at → `is_stale=True`.
  - 파싱 실패 → CURRENT_ONLY fallback.
- DB 재생성: `agent_data/longterm_memory.sqlite`, `agent_data/checkpoints.sqlite`(및 -wal/-shm) 삭제. chroma는 건드리지 않음.
- 스모크: `import manufacturing_agent.runtime` 후 `build_graph` 컴파일 + 1턴 dry 실행(LLM 키 있으면).

---

## 7. 구현 순서(의존성)
1. (#2) contracts: `ContextDecision` + 파생메서드 + DiagnosisContext.age + build.py allowlist.
2. (#3) store: summaries.turn_id + summary_by_turn + char cap.  ← #2와 병렬 가능
3. (#4) engine/packer/normalizer/manager: 단일 1콜 + short-circuit + stale + build_context_summary + 토큰캡.
4. (#5) 소비처 6곳 — 서로 disjoint 파일이라 **병렬**.
5. (#6) 테스트 + DB 재생성 + 스모크.
