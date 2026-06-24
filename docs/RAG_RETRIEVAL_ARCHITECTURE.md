# RAG Retrieval Architecture & Manufacturing Taxonomy Design

## 목적

현재 RAG는 사용자 질문 또는 Prediction Agent의 결과를 그대로 Vector Search에 전달하여 검색을 수행한다.

하지만 AI4I 데이터셋의 변수명과 실제 Haas Maintenance Manual에서 사용하는 용어가 다르기 때문에 단순한 벡터 검색만으로는 검색 품질이 떨어질 수 있다.

예를 들어

| AI4I 변수             | Haas 문서 표현                                                    |
| ------------------- | ------------------------------------------------------------- |
| Torque              | Tool Load, Cutting Force, Heavy Cutting                       |
| Tool Wear           | Tool Wear, Toolholder, Tool Life, Chatter                     |
| Process Temperature | Spindle Temperature, Overheating, Lubrication, Thermal Growth |
| Rotational Speed    | RPM, Spindle Speed, Feedrate, Chip Load                       |
| Air Temperature     | Ambient Temperature, Thermal Growth                           |

따라서

```
AI4I 변수
        ↓
Manufacturing Taxonomy
        ↓
Haas 문서 표현
        ↓
Vector Search
```

의 중간 계층이 필요하다.

이를 위해 `manufacturing_taxonomy.py`를 추가한다.

---

# 현재 사용 문서

모든 RAG 문서는

```
documents/
└── haas/
```

하위에 저장되어 있다.

현재 사용하는 핵심 문서는 다음 세 개이다.

---

## 1. Mechanical Service Manual

역할

* 기계 전체 Troubleshooting
* General Maintenance
* Accuracy
* Thermal Growth
* Machine Level
* Ballscrew
* Axis
* Coolant
* Spindle
* Surface Finish
* Vibration

가장 범용적인 문서이며 기본(Fallback) 검색 대상으로 사용한다.

---

## 2. Mill Spindle Troubleshooting Guide

역할

* Spindle
* Bearing
* Drawbar
* Lubrication
* Overheating
* Encoder
* Belt
* Tool Load
* Spindle Temperature

스핀들 관련 문제를 담당한다.

---

## 3. Mill Chatter Troubleshooting Guide

역할

* Chatter
* Tool Wear
* Tool Holder
* Tool Life
* Surface Finish
* RPM
* Feedrate
* Chip Load
* Cutting Force
* Toolpath

절삭 조건 및 Tool Wear 관련 문제를 담당한다.

---

# Manufacturing Taxonomy의 역할

taxonomy는 단순 Alias Dictionary가 아니다.

다음 여섯 가지 기능을 수행한다.

```
1. Feature Alias

2. Failure Alias

3. Symptom Alias

4. Document Routing

5. Query Expansion

6. Query Fan-out
```

즉

```
Prediction Result

↓

Manufacturing Taxonomy

↓

Retriever
```

사이에서만 동작한다.

LLM은 taxonomy를 직접 사용하지 않는다.

---

# Feature Routing

AI4I 변수별 우선 검색 문서

| Feature             | 우선 검색                                            |
| ------------------- | ------------------------------------------------ |
| Tool Wear           | Mill Chatter → Mill Spindle → Mechanical Service |
| Torque              | Mill Chatter → Mechanical Service → Mill Spindle |
| Rotational Speed    | Mill Chatter → Mill Spindle → Mechanical Service |
| Process Temperature | Mill Spindle → Mechanical Service                |
| Air Temperature     | Mechanical Service → Mill Spindle                |

---

# Failure Routing

| Failure | 우선 검색                                            |
| ------- | ------------------------------------------------ |
| TWF     | Mill Chatter → Mechanical Service                |
| HDF     | Mill Spindle → Mechanical Service                |
| OSF     | Mill Chatter → Mill Spindle → Mechanical Service |
| PWF     | Mill Spindle → Mechanical Service                |
| RNF     | Mechanical Service → Mill Spindle → Mill Chatter |

---

# Symptom Routing

사용자가 자연어로 입력하는 증상도 taxonomy에서 관리한다.

예시

| 사용자 표현  | 검색 표현                              |
| ------- | ---------------------------------- |
| 채터      | chatter, vibration, surface finish |
| 과열      | spindle temperature, overheating   |
| 소음      | bearing, spindle noise             |
| 진동      | vibration, chatter                 |
| 표면 거칠다  | poor finish, chatter               |
| 공구 안 빠짐 | tool sticking, spindle taper       |
| 냉각수     | coolant, coolant system            |

---

# Query Expansion

taxonomy는 AI4I 용어를 Haas 문서 표현으로 확장한다.

예시

```
Torque

↓

Tool Load

Cutting Force

Heavy Cutting

Feedrate

Depth of Cut

Width of Cut
```

---

```
Process Temperature

↓

Spindle Temperature

Overheating

Lubrication

Thermal Growth

Bearing Heat
```

---

```
Tool Wear

↓

Tool Wear

Toolholder

Tool Life

Surface Finish

Chatter
```

---

# Query Fan-out

현재

```
question

↓

vector search
```

변경

```
question

↓

Prediction Result

↓

taxonomy

↓

search terms 생성

↓

fan-out query 생성

↓

vector search

↓

merge

↓

deduplicate

↓

ranking
```

예시

사용자 질문

```
토크가 높고 공구 마모가 심한데 어떻게 점검해야 하나?
```

Prediction Result

```
failure

OSF

features

- torque
- tool_wear
```

taxonomy 결과

```
Tool Load

Cutting Force

Heavy Cutting

Tool Wear

Toolholder

Chatter
```

최종 Query

```
원문 질문

OSF troubleshooting

Tool Load

Heavy Cutting

Tool Wear

Chatter
```

---

# Retriever 변경 방향

taxonomy는 Retriever에서만 사용한다.

기존

```python
vector_search(question)
```

변경

```python
queries = build_rag_queries(...)

for query in queries:
    vector_search(query)

↓

merge

↓

deduplicate

↓

rerank
```

---

# Document Routing

taxonomy는 검색 대상 문서를 먼저 결정한다.

예시

```
Tool Wear

↓

Mill Chatter
```

---

```
HDF

↓

Mill Spindle

↓

Mechanical Service
```

---

```
Accuracy

↓

Mechanical Service
```

---

```
Thermal Growth

↓

Mechanical Service

↓

Mill Spindle
```

---

# 기존 코드에서 수정할 위치

## 1. services/rag_service.py

수정

* Query Builder
* adaptive_retrieve()
* rag_search()

taxonomy를 사용하도록 변경

사용 함수

```
route_documents()

build_search_terms()

build_rag_queries()
```

---

## 2. Evidence Agent

Prediction 결과를 그대로 Retriever에 전달하지 않는다.

taxonomy를 통해 Query를 생성한 뒤 Retriever를 호출한다.

---

## 3. Vector Search

priority_docs를 활용하여 우선 검색한다.

필요 시 다른 문서까지 확장 검색한다.

---

# 수정하지 않는 부분

다음 구조는 그대로 유지한다.

* Supervisor
* ManufacturingState
* Context Manager
* PredictionResult
* EvidenceBundle
* Agent Interface
* LangGraph Workflow

즉 Agent 구조는 변경하지 않는다.

taxonomy는 Retrieval Layer에서만 동작한다.

---

# Claude Code 구현 요청

다음 사항을 반영하여 코드를 수정한다.

1. `manufacturing_taxonomy.py`를 프로젝트에 추가한다.

2. `rag_service.py`에서 taxonomy를 호출하도록 수정한다.

3. Query Builder를 taxonomy 기반 Query Expansion 방식으로 변경한다.

4. Feature, Failure, Symptom을 이용한 Query Fan-out을 구현한다.

5. `priority_docs`를 활용하여 문서 우선 검색을 수행한다.

6. 여러 Query 결과를 Merge → Deduplicate → Re-rank 하도록 Retriever를 수정한다.

7. 기존 Agent 구조와 State는 변경하지 않는다.

8. 새로운 제조사 문서가 추가되더라도 taxonomy만 수정하면 동작하도록 확장성을 유지한다.

---

# 최종 목표

최종 Retrieval 구조는 다음과 같다.

```
User Question
        │
        ▼
Supervisor
        │
        ▼
Prediction Agent
        │
        ▼
Prediction Result
        │
        ▼
Manufacturing Taxonomy
        ├── Feature Alias
        ├── Failure Alias
        ├── Symptom Alias
        ├── Document Routing
        ├── Query Expansion
        └── Query Fan-out
        │
        ▼
Retriever
        │
        ├── Priority Document Search
        ├── Vector Search
        ├── Merge
        ├── Deduplicate
        └── Re-rank
        │
        ▼
Evidence Agent
        │
        ▼
Final Answer
```

이 구조를 유지하면 향후 Siemens, FANUC, Mazak 등 다른 제조사 문서를 추가하더라도 Retrieval 구조를 변경하지 않고 taxonomy만 확장하여 대응할 수 있다.

---

# 구현 노트 (Retrieval Layer 실제 동작)

> taxonomy 연동 + NO_EVIDENCE fallback 구현 후 실제 동작 기준 메모.

## 코드 위치 / 단일 소스

- 모든 RAG retrieval 로직은 `manufacturing_agent/services/rag_service.py` 한 곳에 있다.
- API 서비스 경로(`api/` → `manufacturing_agent.runtime`)와 노트북(`manufacturing_agent_v6.ipynb`),
  시나리오 러너(`scripts/run_manufacturing_scenarios*.py`)가 **모두 이 패키지를 import**한다.
  (노트북은 더 이상 RAG 코드를 inline으로 복제하지 않는다.)
- taxonomy(`manufacturing_agent/services/manufacturing_taxonomy.py`)는 LLM prompt가 아니라
  rule-based retrieval helper로만 사용된다.

## NO_EVIDENCE fallback

- `rag_search`는 top_k 결과가 없거나 모든 score가 `MIN_EVIDENCE_SCORE` 미만이면
  `status="NO_EVIDENCE"`로 documents/citations를 비워 반환한다.
  → Evidence Agent가 LLM 요약(추측)을 호출하지 않고, 사용자에게 담당자 확인 안내를 노출한다.
- `MIN_EVIDENCE_SCORE` 기본값 **0.45** (env `MIN_EVIDENCE_SCORE`로 조정).
  - on-topic Haas 질의 score ≈ 0.54+, off-topic(영문 Haas 코퍼스) ≈ 0.42 이하 → 0.45로 분리됨.
- 담당자 연락처는 하드코딩하지 않고 env에서 읽는다:
  `SUPPORT_CONTACT_NAME`, `SUPPORT_CONTACT_EMAIL`, `SUPPORT_CONTACT_PHONE` (`config.support_contact_text()`).
- 디버그 로그: env `RAG_DEBUG=true` → 입력 feature/failure, priority_docs, fan-out queries,
  검색 chunk(id/source/page/score), fallback 발생 여부를 stderr로 출력.
  (`rag_search` 반환값의 `debug` 키에도 동일 페이로드가 들어간다.)

## 코퍼스 범위

- 코퍼스는 **haas PDF 전용**이다(Mechanical Service Manual / Mill Spindle / Mill Chatter).
  osha/kosha 안전문서 및 `safety_procedure_rag` 프로파일·`safety_guidance` intent는 제거되었다.
- NO_EVIDENCE 임계값 `MIN_EVIDENCE_SCORE`(기본 0.45)는 off-topic 질의를 차단한다.
