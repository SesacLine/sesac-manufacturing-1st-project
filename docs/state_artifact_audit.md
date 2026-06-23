# State / Artifact 필드 감사 — 과잉·중복·죽은 값

> ✅ **Tier 1 제거 완료** (유닛 8개 + 그래프 컴파일 + 단일/결합 e2e 회귀 통과).
> 제거됨: 죽은 모델 `RunTrace`/`ConversationTurn`/`ContextState`, `state["run_trace"]`,
> `EvidenceArtifact` legacy 10필드(`is_retry` 유지), `GateReport` 입력가드 5필드,
> `PredictionResult`의 `available_features`/`evidence_hints`/`full_prediction_available`/`partial_risks`,
> `ContextPacket.current_values`/`previous_prediction_result`, `RouteDecision.stop`,
> `MachineValue.unit`, `SQLIntentDecision.requires_clarification`, 및 모든 set 사이트. (체크포인트 allowlist도 슬림화)
> Tier 2/3는 관측 정책 결정 보류 — 미적용.

> 방법: `manufacturing_agent/**/*.py` + `api/**/*.py`에서 각 필드의 write/read 위치 추적(서브에이전트 4 병렬 + 직접 grep 교차검증). `.venv`·`*.ipynb` 제외.
> "관측 전용" = 런타임 그래프 노드는 안 읽고 `tests/`·`scripts/`에서만 읽음 → 운영 코드 기준 미사용이나 디버그 의도일 수 있음.
> 이 문서는 **조사 결과**이며 코드는 수정하지 않았다.

---

## Tier 1 — 완전 DEAD (어디서도 안 읽힘, 제거 안전) ✅ grep 검증됨

| 대상 | 근거 |
|------|------|
| **`RunTrace` 모델 + `state["run_trace"]`** | `RunTrace(` 인스턴스화 0건. state 세팅·읽기 0. `build.py:72` allowlist 등록만. memory_writer는 RunTrace 대신 plain dict 저장 |
| **`ConversationTurn` 모델** | `ConversationTurn(` 0건. turn은 전부 plain dict로 흐름 |
| **`ContextState` 모델** | `ContextState(` 0건. active_context_id/recent_contexts는 DB·dict로만 처리 |
| **`RouteDecision.stop`** | set·read 모두 0 (항상 기본 False) |
| **`MachineValue.unit`** | normalizer가 unit 미전달 → 항상 None, read 0 |
| **`PredictionResult.full_prediction_available`** | 주석 'legacy', set만(prediction_agent:112), read 0 |
| **`PredictionResult.partial_risks`** | 주석 'legacy', `risk_flags`와 중복, read 0 |
| **`EvidenceArtifact` legacy 블록 (10개)** | `user_query, mode, search_query, tags, doc_whitelist, failure_types, failure_ko, is_prediction_based, supervisor_intent, feedback` — 전부 read 0. (`is_retry`만 살아있음 → 유지) |
| **`GateReport` 입력가드 5필드** | `block, block_reason, layer, message, flags` — intake_gate에서 set만, read 0. status·`input_decision`과 전부 중복 |
| **`ContextPacket.current_values`** | packer:136 set, packet 경유 read 0. (resolution.current_values만 normalizer:41에서 읽힘) |
| **`ContextPacket.previous_prediction_result`** | 전체 객체 set만(manager:43, packer:139), read 0. `previous_prediction_summary`(요약)만 소비 |
| **`SQLIntentDecision.requires_clarification`** | 항상 False 하드코딩, read 0 |

**죽은 모델 3개**: `RunTrace`, `ConversationTurn`, `ContextState` — 클래스 통째로 미사용.

---

## Tier 2 — state 비대화 (런타임 미사용, 관측/테스트만 읽음)

> 제거하면 `tests/test_regression.py`·`scripts/run_manufacturing_scenarios.py`의 결과 덤프가 깨지므로 **동반 정리 필요**.

| 대상 | 근거 |
|------|------|
| `state["input_flags"]` (`InputFlags` 사본) | state["input_flags"] read 0. `is_injection`만 intake_gate 로컬에서 사용, state 사본은 잉여. (IntakeDecision·InputDecision은 둘 다 소비처 있어 유지) |
| `state["intent"]` | `execution_plan.intent`와 중복. state["intent"] 직접 read 0 (소비자는 `plan.intent`를 읽음) |
| `state["orchestrator_decision"]` (`OrchestratorDecision` 준-데드) | 라우팅은 `route`(RouteDecision)+`active_task_id`로 함. orchestrator_decision 런타임 read 0. `next_node`=route.next_node 중복, `active_task_id`=state 최상위 중복 |
| `state["supervisor_planner_decision"]` / `["supervisor_replanner_decision"]` / `["sql_intent_decision"]` | 셋 다 노드가 set만, 런타임 read 0 (관측 전용). `SQLIntentDecision` 모델 자체가 준-데드 |
| `ExecutionPlan`: `created_by`(항상 "llm"), `reason_summary`, `confidence`, `replan_count`, `replan_history` | 기록만, 런타임 read 0 |
| `TaskSpec`: `reason`, `feedback_history`, `plan_revision`, `invalidated_by` | 누적/기록만, 런타임 read 0 |
| `SupervisorReplannerDecision.invalidate_task_ids` | PATCH_AND_RERUN 시 코드가 항상 `final_1` 강제 추가(replanner:96) → 입력값 사실상 무의미 |

---

## Tier 3 — 통합 후 잔재 / WRITE_ONLY (디버그·약한 사용)

| 대상 | 분류 |
|------|------|
| `ContextResolution.patch_values`, `.reason` | 파생만 되고 소비 0 (디버그 추정) |
| `ContextCarryoverDecision.reason_summary` | carryover 경유 read 0 |
| `ContextDecision.llm_skipped` | 런타임 read 0, **테스트만 읽음**(short-circuit 검증). 관측 신호로 유지 권장 |
| `PredictionResult.available_features` | set만, read 0 |
| `PredictionResult.evidence_hints` (+ `EvidenceHint` 모델, `FailureRisk.evidence_query_terms`) | RAG는 `failure_types`/`cause_features`로 동작 → evidence_hints read 0. EvidenceHint 빌드 경로 통째로 죽음 |
| `PredictionResult.base_context_id` | 런타임 read 0 (changed/reused_features만 final_answer에서 읽힘) |
| `PredictionResult.used_stale_features` | 런타임 read 0, scenarios 테스트만 (최근 stale 구현으로 값은 채워짐) |
| `AgentContextPacket.prior_results`의 잉여 키 | evidence_agent는 `is_followup`/`evidence_summary`/`sql_summary`만 읽음 → 나머지 키 WRITE_ONLY |
| `DiagnosisContext.user_id`/`thread_id`(모델 필드), `is_safe_to_reuse`(항상 True) | DB 컬럼과 중복 / 분기 무의미 |
| `GateReport.diagnostics`, `ContextPacket.context_warnings` | 런타임 read 0 (UI/trace 추정 — py 범위 밖 소비 가능성 있어 단정 보류) |

---

## 중복 아님 (확인 — 건드리지 말 것)

- **`TaskSpec.rerun_count/max_reruns` vs `retry_count/max_retries`**: 의미가 다름. retry=동일 params 재실행(plan_ops), rerun=replanner가 params 보정 후 재실행. 둘 다 별도 분기에서 읽힘.
- **`IntakeDecision` vs `InputDecision`**: 어댑터 관계(InputDecision은 IntakeDecision에서 파생)지만 **둘 다 다른 소비처에서 읽힘**. `InputFlags`만 잉여.
- **`OutputSafetyDecision`**: output_safety_gate 안에서 생성·즉시 소비. 데드 아님.
- **`SQLHistoryArtifact` top-level `sql/rows/query_type`**: `results[primary]`의 복제지만, final_answer가 results≥2면 results 경로·1개 이하면 top-level 경로를 쓰는 **이중 폴백**이라 둘 다 소비됨. 제거하려면 final_answer:327-340 + memory_writer:23-25를 `results[0]` 기준으로 통일하는 리팩터 선행 필요.

---

## 정리 우선순위 제안

1. **Tier 1 즉시 제거** — 리스크 0. 특히 죽은 모델 3개(`RunTrace`/`ConversationTurn`/`ContextState`) + `EvidenceArtifact` legacy 10필드 + `GateReport` 입력가드 5필드 + `PredictionResult` legacy 2필드. 코드량 대폭 감소, 체크포인트 직렬화 allowlist도 슬림화.
2. **Tier 2는 "관측을 유지할지" 정책 결정 후** — 유지하려면 state가 아니라 별도 trace/log 채널로 빼서 state 비대화를 막는 게 낫다. 제거하려면 tests/scripts 동반 수정.
3. **Tier 3는 Tier 1과 함께 묶어** 통합-후-잔재 청소(특히 `ContextPacket.current_values`/`previous_prediction_result`, `evidence_hints`/`EvidenceHint`).
