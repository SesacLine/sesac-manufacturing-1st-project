# Artifact 저장 & Final Answer 생성 흐름

> 대상: `manufacturing_agent/` (LangGraph 기반 Gate-driven Plan-and-Execute 멀티 에이전트)
> 한 줄 요약: **각 agent는 결과를 Pydantic artifact로 만들어 "공유 State의 전용 키"에 써넣고, gate가 검증 리포트를 누적하며, `final_answer_node`가 그 artifact들을 facts sheet로 종합해 LLM 해설 + 코드 보장 첨부로 최종 답변을 만든다.**

---

## 1. 큰 그림

```
START
  └─ intake_gate ───(PASS)──> context_manager ─> supervisor_planner ─> orchestrator_dispatcher
        │(차단)                                                                 │
        └────────────────────────> final_answer                                 │  (Plan-and-Execute 루프)
                                                          ┌──────────────────────┘
                                                          ▼
                          ┌───────────── orchestrator_dispatcher ──────────────┐
                          │            (다음 실행 task 선택)                     │
        ┌─────────────────┼──────────────────┬─────────────────┬───────────────┘
        ▼                 ▼                  ▼                 ▼
  prediction_agent   evidence_agent     sql_agent      supervisor_replanner   final_answer
        │                 │                  │                 │                   │
        ▼                 ▼                  ▼                 │                   ▼
  prediction_gate    evidence_gate      sql_gate              │            output_safety_gate
        │                 │                  │                 │                   │
        └────────────┬────┴──────────────────┴─────────────────┘                  ▼
                     ▼ (gate report 반영 후 다시 dispatcher)                   memory_writer ─> END
```

그래프 정의는 `graph/build.py:28` `build_graph()`에 있다. 핵심은 **모든 노드가 `ManufacturingState`라는 단일 공유 dict를 읽고, 부분 dict를 return해서 상태를 갱신**한다는 점이다 (LangGraph state reducer 패턴).

---

## 2. 공유 State = artifact 저장소

artifact는 별도 파일/DB가 아니라 **`ManufacturingState`의 특정 키**에 저장된다.
정의: `contracts/state.py:7` (`ManufacturingState(MessagesState, total=False)`)

| State 키 | 타입 | 누가 씀 | 누적/덮어쓰기 |
|---|---|---|---|
| `prediction_result` | `PredictionResult` | `prediction_agent` | 덮어쓰기 (단일 값) |
| `evidence_bundle` | `EvidenceArtifact` | `evidence_agent` | 덮어쓰기 |
| `sql_result` | `SQLHistoryArtifact` | `sql_agent` | 덮어쓰기 |
| `gate_reports` | `list[dict]` | 모든 gate | **누적 (append)** |
| `execution_plan` | `ExecutionPlan` | planner / dispatcher / replanner | 덮어쓰기 |
| `context_packet` | `ContextPacket` | `context_manager` | 덮어쓰기 |
| `agent_contexts` | `dict` | `context_manager` | 덮어쓰기 |
| `final_answer` | `FinalAnswer` | `final_answer_node` / `output_safety_gate` | 덮어쓰기 |
| `retry_counts` | `dict` | `_wrap_retry` 래퍼 | 누적 |

> **저장 메커니즘**: LangGraph 노드가 `return {"prediction_result": result}` 하면, 리듀서가 그 키만 상태에 병합한다. 단일 값 키는 새 값으로 교체되고, `gate_reports` 같은 리스트는 노드가 직접 `기존 + [새 항목]`을 만들어 누적한다(전용 reducer가 아니라 수동 append 방식).

artifact 스키마 정의는 모두 `contracts/context.py`에 있다.

---

## 3. 각 Agent의 artifact 생성 방식

각 worker 노드는 `graph/build.py:18` `_wrap_retry()`로 감싸져 실행마다 `retry_counts[key] += 1`이 추가된다(무한 루프 방지 + 관측).

### 3.1 prediction_agent → `prediction_result : PredictionResult`
- 파일: `agents/prediction_agent.py:23`
- 입력: `agent_contexts["prediction_agent"]`의 선택된 feature + 이번 턴 구조화 입력(`input_features`, 데이터 입력란이 context 해석보다 우선)
- 처리: `services/prediction_service.run_prediction(feats)` — **규칙 기반(rule-based) 부분 위험 진단** (ML 예측이 아님)
- 상태 매핑: `out["full"]→OK`, `out["risks"]→PARTIAL`, `out["missing"]→NEEDS_INPUT`, else `SKIPPED`
- artifact 핵심 필드: `status`, `risk_flags`(고장유형/레벨/score/규칙/계산식/영향변수/권장점검), `failure_types`, `cause_features`, `safety_hints`, `confidence`, `limitations`, `summary`, 멀티턴용 `context_mode`/`reused_features`
- 반환: `return {"prediction_result": result}` (`prediction_agent.py:106`)

### 3.2 evidence_agent → `evidence_bundle : EvidenceArtifact`
- 파일: `agents/evidence_agent.py:52`
- 처리: RAG 검색(`services/rag_service.rag_search`) → 검색 프로파일 선택(`_pick_profile`, 진단결과 있으면 `prediction_plus_rag`) → citation-aware 문서 구성 → LLM이 **citation 강제([C1] 형식) 근거 요약** 작성
- 상태 분기:
  - 문서 없음 → `EMPTY` (`evidence_agent.py:105`)
  - 관련성 낮음 → `LOW_RELEVANCE` (`:128`)
  - 요약 LLM 실패 → `FAIL` (`:171`, 노드가 죽지 않도록 계약상 artifact로 닫음)
  - 정상 → `OK` (`:192`)
- gate 피드백(`agent_feedback`)이 있으면 `profile="fallback_broad"`, k=8로 보완 재검색
- 반환: `return {"evidence_bundle": bundle}`

### 3.3 sql_agent → `sql_result : SQLHistoryArtifact`
- 파일: `agents/evidence_agent.py:668` (같은 파일에 함께 정의)
- 처리: **PydanticAI Text-to-SQL** Agent가 `failure_history` 단일 테이블에 대한 SELECT를 생성 → 다중 검증 후 readonly 실행
  - 검증: `validate_sql_query`(SELECT-only / 금지 키워드 / allowed_tables / LIMIT / max_rows) + `explain_sql_query`(EXPLAIN QUERY PLAN으로 실제 스키마 존재 검증)
  - 실행: `execute_readonly_sql` (`PRAGMA query_only = ON`, parameterized)
- 복합 질의는 여러 `SQLQueryResult`를 모아 `build_sql_history_artifact_from_results`로 집계(`evidence_agent.py:568`) — 하나라도 OK면 전체 OK, 일부 실패는 `limitations`에 기록
- 상태: `OK / EMPTY / INVALID_REQUEST / BLOCKED / FAIL`
- 시간표현 환각 보정: `_sanitize_time_range`가 reference_date(`2026-06-21`) 기준으로 "최근 N일"을 재계산
- 반환: `return {"sql_result": artifact, "sql_intent_decision": sql_intent}`

> 참고: artifact의 `summary`는 디버그/메모리용 짧은 상태 요약일 뿐이고, **사용자에게 보여줄 해석은 final_answer가 `rows`를 직접 읽어서 만든다**(`_summarize_sql_rows` 주석 참조).

---

## 4. Gate: artifact 검증 → GateReport 누적 → plan 전이

각 worker 직후 gate가 실행된다(`build.py:54-59`). gate는 **artifact를 절대 수정하지 않고**, `GateReport`를 만들어 `gate_reports`에 append만 한다.

- `prediction_gate` (`gates/quality_gates.py:20`): `OK/PARTIAL→PASS`, `NEEDS_INPUT→NEEDS_USER_INPUT`, `SKIPPED→PASS_WITH_WARNINGS`, 없음→`RETRYABLE_FAIL`
- `evidence_gate` (`:48`): success_criteria(min_docs, require_citation, evidence_required) 기준 평가. 필수인데 부족하면 `RETRYABLE_FAIL` → retry 후에도 부족하면 `PLAN_REPAIR_REQUIRED`
- `sql_gate` (`:90`): artifact의 SQL을 **다시 검증**(이중 안전망) + 상태 매핑. 정책 차단은 rerun 예산이 남으면 `PLAN_REPAIR_REQUIRED`, 소진되면 `BLOCK`

GateReport status → task 상태 전이는 `graph/plan_ops.py`의 `PlanOps.apply_gate_report`(`:79`)가 단일 출처로 담당한다:
- `RETRYABLE_FAIL` + retry 예산 남음 → task `PENDING`(재시도), 소진 → `FAIL`
- `PLAN_REPAIR_REQUIRED` → dispatcher가 `supervisor_replanner`로 위임 (task params를 패치 후 rerun)
- `PASS/PASS_WITH_WARNINGS/NEEDS_USER_INPUT/BLOCK` → 해당 종료 상태로

`gate.feedback`은 `route_hint`와 함께 `agent_feedback`으로 전달되어, 재실행되는 agent의 프롬프트/검색 전략을 바꾼다(`dispatcher.py:15` `_agent_feedback_from`).

---

## 5. Orchestration 루프

`orchestrator_dispatcher` (`graph/dispatcher.py:27`)가 매 사이클:
1. 최신 gate report를 plan에 반영(`apply_gate_report`) — 단 이미 소비한 replan report는 제외
2. 끊긴 RUNNING task를 PENDING으로 복구(`reset_orphan_running`, 안전장치)
3. gate가 plan repair 요청 시 → `supervisor_replanner`로 라우팅
4. `PlanOps.next_runnable`(`plan_ops.py:112`)로 **의존성(`depends_on`)이 모두 종결된 첫 PENDING task** 선택 → 해당 worker로
5. 실행 가능한 task가 없으면 → `action="FINALIZE"`, `final_answer`로

즉 prediction/evidence/sql은 `ExecutionPlan`에 task로 정의돼 있을 때만, 의존성 순서대로 실행되고 각자의 키에 artifact를 채운다. plan은 `supervisor_planner_node`가 사용자 의도(LLM 분류)에 따라 최초 생성한다.

---

## 6. Final Answer 생성 (`nodes/final_answer_node.py:679`)

설계 철학(파일 상단 주석): **코드가 모든 사실을 결정하고, LLM은 그 안에서 해설만 쓴다.** 숫자 hallucination을 구조적으로 차단하는 하이브리드 방식.

```
final_answer_node
 ├─ 0) intake 차단이면 차단 메시지 그대로 반환하고 종료
 ├─ 1) build_answer_context(state)  ← 3개 artifact + context_packet을 facts sheet로 결정적 조립
 │       · answer_mode 결정: COMBINED / SQL_ONLY / PREDICTION_ONLY / EVIDENCE_ONLY / HISTORY_WITH_EVIDENCE / ...
 │       · prediction_summary / history_summary / evidence_summary / safety_summary
 │       · diagnosis_block(정확 수치), checklist_block, citations
 ├─ 2) _allowed_numbers(ctx)        ← facts sheet에 등장한 모든 숫자 토큰 집합 추출
 ├─ 3) _synthesize_answer()         ← LLM(tier="final")이 facts sheet 범위 안에서 본문 작성
 │       · _final_answer_quality_feedback: 금지섹션/score노출/raw용어/citation누락/heading 검사
 │       · _number_guard: 단위 붙은 숫자가 allowed_numbers에 없으면 hallucination 플래그
 │       · 문제 있으면 수정 지시 붙여 1회 보수 재생성
 ├─ 4) 폴백: 본문이 비거나 숫자 hallucination이 남으면 _fallback_final_answer(결정적 답변)로 대체
 ├─ 5) 결정적 후처리(순서 중요):
 │       _remove_false_missing_input_section → _ensure_missing_input_visible
 │       → _ensure_diagnosis_block(고장종류별 근거표/체크리스트 정확수치 보장 첨부)
 │       → _ensure_safety_trailer(안전 책임 문구)
 │       → _verdict_banner(맨 앞 "종합 판단" 한 줄) + body
 │       → _localize_answer_terms + _clean_final_answer_format(raw 스키마용어→한국어, heading 제거)
 │       → _ensure_citations_visible([출처] 블록 부착)
 └─ 6) FinalAnswer(answer, citations, warnings, missing_inputs) 생성 + final_answer task를 PASS로 마킹
        return {"final_answer": fa, "execution_plan": ...}
```

핵심 포인트:
- **표·정확 수치·체크리스트·출처는 LLM이 못 짓는다.** 코드가 `risk_flags`에서 직접 렌더링해 보장 첨부한다(`_render_diagnosis_block`, `_render_checklist`).
- **answer_mode**에 따라 섹션 구성이 달라진다. 예: `SQL_ONLY`는 "현재 판단/지금 점검할 일/문서 근거" 섹션 금지.
- LLM이 없거나 검증 실패해도 **결정적 폴백**으로 항상 답이 나온다(안전성).

### 6.1 facts sheet 조립 상세 (`build_answer_context:487`)
artifact별 요약 함수: `_prediction_summary_for_answer`(측정값을 실단위로 풀어씀, 내부 score는 노출 안 함), `_history_summary_for_answer`(query_type별로 rows 집계 — 건수/다운타임/유형/대표사례/대표조치/재발방지), `_evidence_summary_for_answer`, `_safety_summary_for_answer`(intake 판정 + safety_hints + 현장 책임자 판단 문구).

---

## 7. Output Safety Gate (최종 안전망)

`final_answer → output_safety_gate` (`gates/quality_gates.py:214`):
- 결정적 정규식(`_contains_unsafe_execution_instruction`)으로 "점검 없이 재가동 지시 / 안전장치 해제 / 경보 무시 운전" 같은 위험 실행 지시를 먼저 차단 (주변에 부정/경고어 있으면 통과 — 오차단 방지)
- 그 후 LLM 안전 판정(`_llm_output_safety`), LLM이 통과시켜도 결정적 백스톱을 다시 한 번 적용
- 차단 시 `final_answer`를 안전 메시지로 **덮어쓰고** GateReport 기록

---

## 8. 영속화 (멀티턴/세션 복원)

두 갈래로 저장된다:

1. **LangGraph 체크포인트** (`graph/build.py:78` `make_sqlite_saver`)
   - `SqliteSaver` + `JsonPlusSerializer`로 **State 전체(모든 artifact 포함)**를 thread_id 단위로 직렬화
   - 직렬화 허용 타입은 `CHECKPOINT_SAFE_TYPES`(`build.py:66`)에 명시 (Pydantic 모델 allowlist)
   - 덕분에 노드 실패 후 재개(`runtime.py:69` `_invoke_from_checkpoint`)와 멀티턴 context 재사용이 가능

2. **memory_writer_node** (`nodes/memory_writer_node.py:57`, 그래프 마지막 노드)
   - 대화 turn 저장(user/assistant), prediction/evidence/sql **요약**을 `conversation_store`에 저장(다음 턴 carryover용)
   - 진단이 OK/PARTIAL이고 입력값이 있으면 `DiagnosisContext` snapshot 저장(이후 "아까 그 진단에서 토크만 바꾸면?" 같은 후속질문 지원)
   - 실행 이력(`gate_reports`, `retry_counts`)을 `run_store`에 저장
   - 최종적으로 `messages`에 AIMessage 추가

---

## 9. 한 턴 데이터 흐름 예시 (combined_analysis)

```
사용자: "이 입력값 위험 진단하고 비슷한 과거 고장도 찾아줘" + 수치 입력
 1. intake_gate         → 서비스 허용 PASS
 2. context_manager     → context_packet, agent_contexts 구성
 3. supervisor_planner  → ExecutionPlan{ pred → sql(depends pred) → evidence → final_answer }
 4. dispatcher → prediction_agent → prediction_result(PARTIAL) → prediction_gate(PASS)
 5. dispatcher → sql_agent        → sql_result(OK)            → sql_gate(PASS)
 6. dispatcher → evidence_agent   → evidence_bundle(OK)       → evidence_gate(PASS)
 7. dispatcher → (남은 task 없음) FINALIZE
 8. final_answer        → 3 artifact를 facts sheet로 종합 → LLM 해설 + 코드 수치/표/출처 보장 → final_answer
 9. output_safety_gate  → 위험표현 검사 PASS
10. memory_writer       → 대화/요약/DiagnosisContext/run 저장 → END
```

---

## 10. 아쉬운 점 / 개선 여지

코드를 읽으며 눈에 띈 부분 (우선순위 대략 높은 순):

1. **`gate_reports`가 무한 누적 + 수동 append**
   `state.get("gate_reports", []) + [report.model_dump()]` 패턴(`quality_gates.py:37` 등)을 모든 gate가 반복한다. LangGraph의 `Annotated[list, add]` reducer를 쓰면 더 안전하고 코드 중복이 준다. 또한 긴 멀티턴에서 report가 계속 쌓여 체크포인트가 비대해진다(`_last_report`는 매번 전체를 역순 순회). **턴 경계에서 정리하거나 상한**을 두는 게 좋다 — 현재는 `intake_gate`만 턴 시작에서 리셋한다(`intake_gate.py:151`).

2. **artifact가 단일 값이라 멀티턴에서 이전 결과가 덮인다**
   `prediction_result`/`evidence_bundle`/`sql_result`는 키 하나라 다음 턴에 덮어쓰여진다. 이전 결과는 `context_packet.previous_*_summary`(요약 문자열)로만 남아 정밀도 손실이 있다. 진단 snapshot은 `DiagnosisContext`로 따로 저장하지만 evidence/sql은 요약만 남는다.

3. **evidence_agent와 sql_agent가 한 파일(700+줄)에 공존**
   `agents/evidence_agent.py`에 SQL Text-to-SQL 전체 로직, 스키마 가이드, few-shot, 검증, gate 헬퍼까지 들어 있다. `prediction_agent.py`만 분리돼 있어 일관성이 없다. SQL 관련은 `agents/sql_agent.py`로 분리하면 가독성이 크게 는다.

4. **provider 혼재**: 본문 LLM은 `call_llm`(설정상 Claude 계열로 보이는 tier 추상화)인데, **SQL Agent만 PydanticAI + OpenAI(`gpt-4.1-mini`)에 하드 의존**(`evidence_agent.py:212`, `OPENAI_API_KEY` 필수). 키가 없으면 SQL task가 통째로 FAIL로 닫힌다. 단일 provider 추상화로 통일하거나, 최소한 폴백 경로가 있으면 견고해진다.

5. **`_number_guard`의 한계(설계상 트레이드오프)**
   단위 없는 숫자·한 자리 숫자는 오탐 방지를 위해 검사하지 않는다(`final_answer_node.py:556`). 즉 "약 3배", "50%p" 같은 단위 없는 환각 수치는 통과할 수 있다. 또 facts sheet에 우연히 같은 숫자가 있으면(예: chunk index 5 vs 5건) 무관한 숫자를 허용해버릴 수 있다 — allowed 집합이 출처 구분 없는 토큰 집합이라서다.

6. **재시도/리플랜 예산이 곳곳에 흩어진 매직넘버**
   `TaskSpec.max_retries=2`, `max_reruns=2`, RAG `k=20/16/8`, `recursion_limit=50` 등이 상수로 박혀 있다. 설정(config)으로 모으면 튜닝과 환경별 조정이 쉬워진다.

7. **관측성(`run_trace`) 미완**
   State에 `run_trace: Optional[RunTrace]` 필드가 선언돼 있지만(`state.py:41`) 실제로 채우는 코드가 없다. 디버깅은 `gate_reports`와 `_print_turn_result`에 의존한다. 이벤트 타임라인을 `run_trace`에 적재하면 추적성이 좋아진다.

8. **테스트/CI 부재로 보이는 점**
   plan 상태머신(`PlanOps`), 숫자 가드, SQL 검증처럼 순수 함수가 많아 단위 테스트하기 좋은 구조인데 `tests/`가 보이지 않는다. 회귀 방지를 위해 골든 테스트(특히 final_answer facts sheet 조립과 number_guard)를 추가하면 안전하다.

9. **노트북/패키지 이중 관리**
   `manufacturing_agent_v6.ipynb`와 `manufacturing_agent/` 패키지가 같은 코드를 담는 듯하다(각 파일이 `print("...정의 완료")`로 끝남 — 노트북 셀 흔적). 동기화가 어긋나면 디버깅이 혼란스러울 수 있어, 패키지를 단일 출처로 하고 노트북은 import만 하도록 정리하는 편이 낫다.
