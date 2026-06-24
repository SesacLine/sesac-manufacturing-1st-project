# 골든 데이터셋 (evals/golden)

`docs/evaluation_framework.md`의 형식을 따른 **평가용 정답 데이터** 스켈레톤. 각 파일은 JSONL(한 줄 = 한 케이스).
실제 시스템(failure_history DB, 문서 코퍼스, detail/aggregate, 28 시나리오)에 맞춘 **샘플**이며, 확장해서 쓴다.

> 라벨 품질 주의: `rag_retrieval`의 relevant_doc_ids와 `final_answer`의 gold_answer는 **사람이 검수/보강**해야 한다(여기 값은 초안). 명확/적대/경계 케이스를 섞고, 일부는 held-out으로 분리.

## 파일 / 평가 대상 / 1순위 지표
| 파일 | 컴포넌트 | 지표 |
|---|---|---|
| `routing.jsonl` | supervisor_planner | EM(needs 집합) + per-label F1 |
| `intake.jsonl` | intake_gate | 위험 Recall + False-Block률 |
| `multiturn.jsonl` | context_manager | mode 정확도 + uses_* F1 |
| `text_to_sql.jsonl` | sql_agent | Execution Accuracy(invariant) + 안전 거절률 |
| `rag_retrieval.jsonl` | evidence(검색) | Recall@k + MRR |
| `final_answer.jsonl` | final_answer | LLM-judge 루브릭 + 숫자환각률 + (BERTScore) |

## 공통 필드
- `id`: 고유 식별자. `split`: `clear`(명확) / `adversarial`(경계) / `holdout`(튜닝 미사용).
- `msg`/`query`: 입력. `input_features`: 구조화 수치(선택).
- `expect`/`gold`/`rubric`: 정답·판정 기준.

## 참고 상수
- reference_date = `2026-06-21`. failure_type ∈ {TWF, HDF, OSF, PWF, SAFETY_INTERLOCK}.
- features: type(L/M/H), air_temperature, process_temperature, rotational_speed, torque, tool_wear.
- 문서 코퍼스(**haas PDF 전용**, html 미고려): Mechanical Service Manual / haascnc.com Mill Spindle / haascnc.com Mill Chatter PDF.
  - osha/kosha 안전문서·Vector Drive는 코퍼스에서 제거됨 → `rag_retrieval` golden에서 제외. `rag_retrieval_eval.py`는 검색 결과의 .html source도 제외한다.
