"""RAG 전용 시나리오 러너.

기존 전체 시나리오 러너(`run_manufacturing_scenarios.py` / `run_manufacturing_scenarios_v2.py`)를
**수정하지 않고** Runtime Loader / Scenario / Turn / run_scenario / summarize_result 구조를 재사용해
RAG(문서 근거) 검색 품질만 검증하는 별도 러너다.

검증 포인트
- 단순 RAG: evidence task만 생성(prediction/sql 생성 시 실패)
- 예측 기반 RAG: single(단일 턴 prediction+evidence) / multiturn(1턴 예측 -> 2턴 후속 RAG) 분리
- 모든 RAG 답변: citation 또는 "문서 근거" 섹션 필수, 빈 답변/일반론 실패
- NO_EVIDENCE fallback: 근거 없음/담당자 확인 안내 검증
- taxonomy fan-out / priority_docs / 검색 source·page·score를 trace/debug로 출력

실행:
    python scripts/run_rag_scenarios.py
    python scripts/run_rag_scenarios.py --group cause
    python scripts/run_rag_scenarios.py --group prediction_rag_single
    python scripts/run_rag_scenarios.py --group prediction_rag_multiturn
    python scripts/run_rag_scenarios.py --scenario RAG_CAUSE_01
    python scripts/run_rag_scenarios.py --full-answer --json
    python scripts/run_rag_scenarios.py --dump-dir traces/rag
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 기존 러너 모듈을 라이브러리로 재사용 (scripts/ 를 import 경로에 추가)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manufacturing_scenarios as base  # noqa: E402
import run_manufacturing_scenarios_v2 as v2  # noqa: E402  (보정된 DEFINITION_CELLS 로더 재사용)
from run_manufacturing_scenarios import (  # noqa: E402
    Turn,
    Scenario,
    FEATURES_HIGH_RISK,
    _require,
    _task_types,
    _answer,
)

CheckResult = list[str]


# ---------------------------------------------------------------------------
# state 직접 읽기 헬퍼 (artifacts dict가 비어있는 런타임 대비)
# ---------------------------------------------------------------------------
def _pred_status(r: dict[str, Any]) -> str | None:
    return getattr(r.get("prediction_result"), "status", None)


def _ev(r: dict[str, Any]):
    return r.get("evidence_bundle")


def _ev_status(r: dict[str, Any]) -> str | None:
    return getattr(r.get("evidence_bundle"), "status", None)


# 답변에 citation 또는 문서 근거 섹션이 있는지
_EVIDENCE_SECTION_MARKERS = ("[출처]", "[C1]", "문서 근거")
# NO_EVIDENCE / 근거 없음 정직 표현
_NO_EVIDENCE_PHRASES = (
    "근거 없음", "근거가 부족", "근거 부족", "찾지 못", "담당자",
    "확인이 필요", "확인할 필요", "단정하기 어렵", "제공되지 않", "확인되지 않",
)


def _has_evidence_section(answer: str) -> bool:
    return any(m in (answer or "") for m in _EVIDENCE_SECTION_MARKERS)


# ---------------------------------------------------------------------------
# RAG trace (요구사항 6/8): fan-out / priority_docs / source·page·score
# ---------------------------------------------------------------------------
def _rag_trace(result: dict[str, Any]) -> dict[str, Any]:
    ev = _ev(result)
    if not ev:
        return {}
    cits = getattr(ev, "citations", None) or []
    return {
        "status": getattr(ev, "status", None),
        "profile": getattr(ev, "retrieval_profile", None),
        "mode": getattr(ev, "mode", None),
        "search_query": getattr(ev, "search_query", None),
        "expansion_tags": (getattr(ev, "tags", None) or [])[:12],   # taxonomy가 만든 Haas 확장어
        "priority_docs": getattr(ev, "doc_whitelist", None),         # route_documents 결과
        "failure_types": getattr(ev, "failure_types", None),
        "retrieved": [
            {
                "source": c.get("source"),
                "page": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "score": c.get("score"),
            }
            for c in cits
        ],
    }


def _print_rag_trace(results: list[dict[str, Any]]) -> None:
    tr = _rag_trace(results[-1]) if results else {}
    if not tr:
        print("    rag_trace: (evidence_bundle 없음)")
        return
    print(f"    rag_trace: status={tr['status']} profile={tr['profile']} mode={tr['mode']}")
    print(f"      search_query: {tr['search_query']}")
    print(f"      expansion_tags: {tr['expansion_tags']}")
    print(f"      priority_docs: {tr['priority_docs']} | failure_types: {tr['failure_types']}")
    if tr["retrieved"]:
        print("      retrieved (source/chunk/score):")
        for d in tr["retrieved"]:
            print(f"        - {d['source']} (chunk={d['chunk_index']}, page={d['page']}) score={d['score']}")
    else:
        print("      retrieved: (없음)")


def _score_list(result: dict[str, Any]) -> list[float]:
    """검색된 문서 유사도 score 목록(citation score = 1 - cosine_distance)."""
    ev = _ev(result)
    cits = (getattr(ev, "citations", None) or []) if ev else []
    return [c["score"] for c in cits if c.get("score") is not None]


def report_rag_result(scenario: Scenario, ok: bool, failures: CheckResult,
                      results: list[dict[str, Any]], summary: dict[str, Any],
                      elapsed: float, *, full_answer: bool = False, trace: bool = False) -> None:
    """py 러너 / 노트북 공용 결과 출력 — 소요 시간 + 유사도 점수 + rag_trace."""
    print(f"  {'PASS ✅' if ok else 'FAIL ❌'} ({elapsed:.2f}s)")
    if summary.get("tasks"):
        print(f"  tasks={summary['tasks']}")
    scores = _score_list(results[-1]) if results else []
    if scores:
        print(f"  유사도 score: top={max(scores):.3f} 평균={sum(scores) / len(scores):.3f} "
              f"전체={[round(s, 3) for s in scores]}")
    else:
        print("  유사도 score: (검색 문서 없음 / NO_EVIDENCE)")
    if not ok:
        for failure in failures:
            print(f"  - {failure}")
        print("  질의:")
        for i, turn in enumerate(scenario.turns, 1):
            print(f"    {i}. {turn.message}{' (+입력 피처)' if turn.input_features else ''}")
    if not ok or trace:
        _print_rag_trace(results)
    if summary.get("answer_preview"):
        print(f"  answer={summary['answer_preview']}")
    if full_answer and summary.get("answer"):
        print("  --- 최종 답변 ---")
        for line in summary["answer"].splitlines():
            print(f"    {line}")


def run_rag_inline(g: dict[str, Any], scenario: Scenario, run_id: str,
                   results_acc: dict[str, bool] | None = None,
                   timings_acc: dict[str, float] | None = None,
                   *, full_answer: bool = True, trace: bool = True) -> dict[str, Any]:
    """노트북/스크립트 공용: 시나리오 1개 실행 + 소요시간 측정 + 결과 출력.

    notebook 셀에서 `summary = run_rag_inline(g, sc, run_id, RESULTS, TIMINGS)` 형태로 호출한다.
    """
    print(f"[{scenario.sid}] {scenario.description}")
    t0 = time.perf_counter()
    ok, failures, results = base.run_scenario(g, scenario, run_id)
    elapsed = time.perf_counter() - t0
    summary = base.summarize_result(
        scenario, ok, failures, results,
        include_full_answer=full_answer, include_trace=trace,
    )
    summary["elapsed_sec"] = round(elapsed, 2)
    summary["rag_trace"] = _rag_trace(results[-1]) if results else {}
    summary["scores"] = _score_list(results[-1]) if results else []
    if results_acc is not None:
        results_acc[scenario.sid] = ok
    if timings_acc is not None:
        timings_acc[scenario.sid] = elapsed
    report_rag_result(scenario, ok, failures, results, summary, elapsed,
                      full_answer=full_answer, trace=trace)
    return summary


# ---------------------------------------------------------------------------
# 체크 함수
# ---------------------------------------------------------------------------
def _check_rag_only(results: list[dict[str, Any]], g: dict[str, Any]) -> CheckResult:
    """단순 RAG: evidence task만 생성. prediction/sql 생성 시 실패. citation/문서 근거 필수."""
    r = results[-1]
    failures: CheckResult = []
    tasks = _task_types(r)
    _require("evidence" in tasks, f"단순 RAG인데 evidence task 없음: {tasks}", failures)
    _require("prediction" not in tasks, f"단순 RAG인데 prediction task가 생성됨: {tasks}", failures)
    _require("sql" not in tasks, f"단순 RAG인데 sql task가 생성됨: {tasks}", failures)
    answer = _answer(r)
    _require(bool(answer.strip()), "답변이 비어 있음", failures)
    _require(_has_evidence_section(answer),
             "citation/문서 근거 섹션 없음(문서 근거 없이 일반론 답변 의심)", failures)
    return failures


def _check_prediction_rag_single(results: list[dict[str, Any]], g: dict[str, Any]) -> CheckResult:
    """예측 기반 RAG(단일 턴): prediction + evidence 생성, sql 없음, citation/문서 근거 필수."""
    r = results[-1]
    failures: CheckResult = []
    tasks = _task_types(r)
    _require("prediction" in tasks, f"prediction task 없음: {tasks}", failures)
    _require("evidence" in tasks, f"evidence task 없음: {tasks}", failures)
    _require("sql" not in tasks, f"예측 기반 RAG인데 sql task가 생성됨: {tasks}", failures)
    answer = _answer(r)
    _require(bool(answer.strip()), "답변이 비어 있음", failures)
    _require(_has_evidence_section(answer), "citation/문서 근거 섹션 없음", failures)
    return failures


def _check_prediction_rag_multiturn(results: list[dict[str, Any]], g: dict[str, Any]) -> CheckResult:
    """예측 기반 RAG(멀티턴): 1턴 prediction OK/PARTIAL -> 2턴 evidence + 이전 prediction 참조."""
    failures: CheckResult = []
    if len(results) < 2:
        _require(False, f"멀티턴 결과가 2턴이 아님: turns={len(results)}", failures)
        return failures
    first, second = results[0], results[-1]
    _require(_pred_status(first) in {"OK", "PARTIAL"},
             f"1턴 prediction status가 OK/PARTIAL이 아님: {_pred_status(first)}", failures)
    tasks = _task_types(second)
    _require("evidence" in tasks, f"2턴 evidence task 없음: {tasks}", failures)
    packet = second.get("context_packet")
    carry = getattr(packet, "context_carryover", None) if packet else None
    refs_prev = bool(packet and (
        getattr(packet, "previous_prediction_summary", None)
        or getattr(packet, "previous_prediction_result", None)
        or (carry and getattr(carry, "uses_previous_prediction", False))
    ))
    _require(refs_prev, "2턴이 이전 prediction_result를 참조하지 않음(context_packet에 이전 진단 없음)", failures)
    answer = _answer(second)
    _require(bool(answer.strip()), "답변이 비어 있음", failures)
    _require(_has_evidence_section(answer), "citation/문서 근거 섹션 없음", failures)
    return failures


def _check_no_evidence_fallback(results: list[dict[str, Any]], g: dict[str, Any]) -> CheckResult:
    """코퍼스에 없는 주제: evidence task 생성 + 근거 없음/담당자 확인 안내, 없는 내용 단정 금지."""
    r = results[-1]
    failures: CheckResult = []
    _require("evidence" in _task_types(r), f"evidence task 없음: {_task_types(r)}", failures)
    answer = _answer(r)
    _require(bool(answer.strip()), "답변이 비어 있음", failures)
    _require(
        any(p in answer for p in _NO_EVIDENCE_PHRASES),
        f"근거 없음/근거 부족/담당자 확인 안내 표현이 없음(없는 내용을 단정했을 가능성): {answer[:180]}",
        failures,
    )
    return failures


# ---------------------------------------------------------------------------
# 시나리오 목록
# ---------------------------------------------------------------------------
_DIAGNOSE = "입력한 데이터로 고장 위험을 진단해줘."


def scenarios() -> list[Scenario]:
    cause = [
        ("RAG_CAUSE_01", "스핀들이 과열되는 원인은 뭐야?"),
        ("RAG_CAUSE_02", "밀링 머신 진동이 심한 이유가 뭐야?"),
        ("RAG_CAUSE_03", "절삭면 품질이 갑자기 나빠졌는데 왜 그럴까?"),
        ("RAG_CAUSE_04", "공구가 빨리 마모되는 원인은?"),
        ("RAG_CAUSE_05", "토크가 높아지면 어떤 문제가 생겨?"),
        ("RAG_CAUSE_06", "회전수가 너무 낮으면 어떤 영향이 있어?"),
        ("RAG_CAUSE_07", "공정 온도가 높으면 어떤 문제가 발생해?"),
        ("RAG_CAUSE_08", "냉각이 제대로 안 되면 어떤 고장이 발생할 수 있어?"),
    ]
    inspection = [
        ("RAG_INSPECTION_01", "스핀들 점검 절차 알려줘."),
        ("RAG_INSPECTION_02", "공구 마모는 어떻게 확인해?"),
        ("RAG_INSPECTION_03", "벨트 이상 여부는 어떻게 점검해?"),
        ("RAG_INSPECTION_04", "냉각 시스템 점검 순서 알려줘."),
        ("RAG_INSPECTION_05", "베어링 이상은 어떻게 확인하지?"),
        ("RAG_INSPECTION_06", "진동이 심할 때 가장 먼저 확인할 부분은?"),
        ("RAG_INSPECTION_07", "모터 이상 여부를 확인하는 방법 알려줘."),
    ]
    maintenance = [
        ("RAG_MAINTENANCE_01", "공구를 언제 교체해야 해?"),
        ("RAG_MAINTENANCE_02", "스핀들 정비 절차 알려줘."),
        ("RAG_MAINTENANCE_03", "냉각장치 청소는 어떻게 해?"),
        ("RAG_MAINTENANCE_04", "토크가 높을 때 어떤 조치를 해야 해?"),
        ("RAG_MAINTENANCE_05", "과열이 발생했을 때 대응 절차 알려줘."),
        ("RAG_MAINTENANCE_06", "작업 재개 전에 확인해야 하는 사항은?"),
    ]
    preventive = [
        ("RAG_PREVENTIVE_01", "공구 수명을 늘리는 방법 알려줘."),
        ("RAG_PREVENTIVE_02", "예방 점검 주기는 어떻게 잡아?"),
        ("RAG_PREVENTIVE_03", "과부하를 방지하려면 어떻게 해야 해?"),
        ("RAG_PREVENTIVE_04", "스핀들 수명을 늘리는 관리 방법 알려줘."),
    ]

    items: list[Scenario] = []
    for group, rows in [("cause", cause), ("inspection", inspection),
                        ("maintenance", maintenance), ("preventive", preventive)]:
        for sid, q in rows:
            items.append(Scenario(sid, q, [Turn(q)], _check_rag_only, tags=[group, "rag"]))

    # prediction_rag_single: 단일 턴(설비 입력 + 예측/근거)
    pred_single = [
        ("RAG_PRED_SINGLE_01", "입력한 데이터로 고장 위험을 진단하고 관련 정비 기준도 알려줘."),
        ("RAG_PRED_SINGLE_02", "입력한 데이터로 HDF 위험을 진단하고 지금 가장 먼저 해야 할 점검을 알려줘."),
        ("RAG_PRED_SINGLE_03", "입력한 데이터로 HDF 위험을 진단하고 작업을 계속해도 되는지 문서 근거와 함께 알려줘."),
        ("RAG_PRED_SINGLE_04", "입력한 데이터로 HDF 위험을 진단하고 어떤 부품부터 확인해야 하는지 알려줘."),
        ("RAG_PRED_SINGLE_05", "입력한 데이터로 HDF 위험을 진단하고 점검 순서를 알려줘."),
        ("RAG_PRED_SINGLE_06", "입력한 데이터로 HDF 위험을 진단하고 작업을 중단해야 하는 상황인지 알려줘."),
    ]
    for sid, q in pred_single:
        items.append(Scenario(sid, q, [Turn(q, FEATURES_HIGH_RISK)],
                              _check_prediction_rag_single, tags=["prediction_rag_single", "rag"]))

    # prediction_rag_multiturn: 1턴 예측 -> 2턴 후속 RAG
    pred_multi = [
        ("RAG_PRED_MULTI_01", "이런 상황에서 관련 정비 기준 있어?"),
        ("RAG_PRED_MULTI_02", "지금 가장 먼저 해야 할 점검은?"),
        ("RAG_PRED_MULTI_03", "작업 계속해도 돼?"),
        ("RAG_PRED_MULTI_04", "어떤 부품부터 확인해야 해?"),
        ("RAG_PRED_MULTI_05", "점검 순서를 알려줘."),
        ("RAG_PRED_MULTI_06", "작업을 중단해야 하는 상황이야?"),
        ("RAG_PRED_MULTI_07", "토크가 높으면 왜 OSF가 발생하지?"),
        ("RAG_PRED_MULTI_08", "토크를 낮추면 위험이 줄어들까?"),
        ("RAG_PRED_MULTI_09", "회전수를 올리면 어떻게 돼?"),
        ("RAG_PRED_MULTI_10", "공구를 교체하면 괜찮을까?"),
    ]
    for sid, q in pred_multi:
        items.append(Scenario(
            sid, q,
            [Turn(_DIAGNOSE, FEATURES_HIGH_RISK), Turn(q)],
            _check_prediction_rag_multiturn, tags=["prediction_rag_multiturn", "rag"],
        ))

    # empty: 코퍼스에 없는 주제 -> NO_EVIDENCE fallback
    empty = [
        ("RAG_EMPTY_01", "용접 로봇 토치 케이블 교체 주기에 대한 정비 문서 근거를 찾아줘."),
        ("RAG_EMPTY_02", "반도체 노광 장비 렌즈 세정 주기에 대한 Haas 문서 근거를 찾아줘."),
    ]
    for sid, q in empty:
        items.append(Scenario(sid, q, [Turn(q)], _check_no_evidence_fallback, tags=["empty", "rag"]))

    return items


GROUPS = ["cause", "inspection", "maintenance", "preventive",
          "prediction_rag_single", "prediction_rag_multiturn", "empty"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 전용 시나리오 러너 (문서 근거 검색 품질 검증).")
    parser.add_argument("--scenario", action="append", help="특정 시나리오 id만 실행(반복 지정 가능).")
    parser.add_argument("--group", action="append", help=f"그룹만 실행: {', '.join(GROUPS)}")
    parser.add_argument("--json", action="store_true", help="전체 JSON 요약 출력.")
    parser.add_argument("--full-answer", action="store_true", help="각 시나리오 최종 답변 전체 출력/포함.")
    parser.add_argument("--trace", action="store_true", help="rag_trace(fan-out/priority/source/score)와 내부 state를 출력/포함.")
    parser.add_argument("--dump-dir", help="시나리오별 상세 JSON trace(+rag_trace)를 이 디렉터리에 저장.")
    args = parser.parse_args()

    # trace/dump 시 retrieval layer 디버그 로그(stderr)도 켠다. (런타임 로드 전에 설정해야 config가 읽음)
    if args.trace or args.dump_dir:
        os.environ.setdefault("RAG_DEBUG", "true")

    selected = scenarios()
    if args.group:
        wanted = set(args.group)
        unknown = wanted - set(GROUPS)
        if unknown:
            print(f"Unknown groups: {sorted(unknown)} (사용 가능: {GROUPS})", file=sys.stderr)
            return 2
        selected = [s for s in selected if wanted & set(s.tags)]
    if args.scenario:
        want = set(args.scenario)
        selected = [s for s in selected if s.sid in want]
        missing = want - {s.sid for s in selected}
        if missing:
            print(f"Unknown scenario ids: {sorted(missing)}", file=sys.stderr)
            return 2
    if not selected:
        print("선택된 시나리오가 없습니다.", file=sys.stderr)
        return 2

    g = v2._load_runtime()  # 보정된 DEFINITION_CELLS 로더 재사용
    run_id = str(int(time.time()))
    summaries: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for scenario in selected:
        print(f"\n[{scenario.sid}] {scenario.description}", flush=True)
        t0 = time.perf_counter()
        ok, failures, results = base.run_scenario(g, scenario, run_id)
        elapsed = time.perf_counter() - t0
        timings[scenario.sid] = elapsed
        summary = base.summarize_result(
            scenario, ok, failures, results,
            include_full_answer=args.full_answer or bool(dump_dir),
            include_trace=args.trace or bool(dump_dir),
        )
        # 소요 시간 / 유사도 score / RAG trace를 요약에 추가
        summary["elapsed_sec"] = round(elapsed, 2)
        summary["scores"] = _score_list(results[-1]) if results else []
        summary["rag_trace"] = _rag_trace(results[-1]) if results else {}
        if dump_dir:
            trace_file = dump_dir / f"{scenario.sid}.json"
            trace_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["trace_file"] = str(trace_file)
        summaries.append(summary)

        # 소요 시간 + 유사도 점수 + (실패/trace 시) rag_trace 출력
        report_rag_result(scenario, ok, failures, results, summary, elapsed,
                          full_answer=args.full_answer, trace=args.trace)
        if summary.get("trace_file"):
            print(f"  trace_file={summary['trace_file']}")

    passed = sum(1 for s in summaries if s["ok"])
    total_time = sum(timings.values())
    print(f"\nRAG scenario result: {passed}/{len(summaries)} passed | 총 {total_time:.2f}s")
    # 그룹별 요약
    by_group: dict[str, list[bool]] = {}
    for s, sc in zip(summaries, selected):
        grp = next((t for t in sc.tags if t in GROUPS), "?")
        by_group.setdefault(grp, []).append(s["ok"])
    print("그룹별:")
    for grp in GROUPS:
        if grp in by_group:
            oks = by_group[grp]
            print(f"  {grp}: {sum(oks)}/{len(oks)}")

    if dump_dir:
        index_file = dump_dir / "index.json"
        index_file.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Trace index: {index_file}")
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0 if passed == len(summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
