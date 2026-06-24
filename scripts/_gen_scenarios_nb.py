"""run_manufacturing_scenarios_v2.py 를 '시나리오=셀' 노트북으로 생성한다(인라인 버전).

각 시나리오 셀에 Scenario(...) / Turn(...) 정의를 그대로 펼쳐서 넣는다.
검증 함수 본문(수백 줄)은 중복하지 않고 v2 모듈을 참조한다(`v2._check_...`).
- 시나리오 1개당 (markdown 설명 셀 + code 정의·실행 셀) 1쌍
- 결과: manufacturing_scenarios_v2.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_manufacturing_scenarios as base  # noqa: E402
import run_manufacturing_scenarios_v2 as v2  # noqa: E402

OUT = ROOT / "scripts" / "manufacturing_scenarios_v2.ipynb"

GROUP_LABELS = {
    "A": "A. 안전 · 차단",
    "B": "B. 위험 진단 (조합 매트릭스)",
    "C": "C. 과거 이력 조회",
    "D": "D. 지식 검색 (RAG)",
    "R": "R. 구조 · 안전 회귀",
}

# sid -> 셀에 그대로 렌더링할 check 표현식.
# 검증 로직은 v2 모듈을 재사용한다(본문 중복 없음). 팩토리 check는 호출식 그대로 둔다.
CHECK_EXPR = {
    "S1_prompt_injection": 'v2._checks_intake_block("injection")',
    "S2_out_of_scope": 'v2._checks_intake_block("out_of_scope")',
    "S3_dangerous_execution": 'v2._checks_intake_block("dangerous_request")',
    "S4_diagnosis": "v2._check_diagnosis_only",
    "S4-1_diagnosis_evidence": "v2._check_diag_plus_evidence",
    "S4-2_diagnosis_history": "v2._check_diag_plus_history",
    "S4-3_diagnosis_history_evidence": "v2._check_combined",
    "S5_multiturn_rediagnose": "v2._check_multiturn_stale",
    "S5-1_multiturn_evidence": "v2._check_multiturn_diag_plus(need_evidence=True, need_sql=False)",
    "S5-2_multiturn_history": "v2._check_multiturn_diag_plus(need_evidence=False, need_sql=True)",
    "S5-3_multiturn_history_evidence": "v2._check_multiturn_diag_plus(need_evidence=True, need_sql=True)",
    "S6_missing_features": "v2._check_missing_features",
    "S7_failure_history": "v2._check_failure_history_actions",
    "S8_similar_incidents": "v2._check_similar_incidents",
    "S9_failure_type_filter": "v2._check_failure_type_filter",
    "S10_empty_history": "v2._check_failure_history_actions",
    "S11_maintenance_guide": "v2._check_evidence_only",
    "S12_missing_guide": "v2._check_evidence_empty",
    "R2_injection_in_doc": 'v2._checks_intake_block("injection")',
    "R3_output_safety_direct": "v2._check_output_safety_direct",
    "R4_multiturn_sql_followup": "v2._check_multiturn_sql_followup",
    "R5_multiturn_evidence_followup": "v2._check_multiturn_evidence_followup",
    "R6_structural_boundaries": "v2._check_structural_boundaries",
    "R7_text_to_sql_rag_quality": "v2._check_text_to_sql_and_rag_quality",
    "R8_plan_and_execute_replan": "v2._check_plan_and_execute_replan",
    "R9_broad_lookup_no_contamination": "v2._check_broad_problem_lookup_feature_context",
    "R10_sqlite_checkpoint_resume": "v2._check_sqlite_checkpoint_resume",
}

_CID = [0]


def _next_id() -> str:
    _CID[0] += 1
    return f"cell-{_CID[0]:03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


BOOTSTRAP = '''\
# === 부트스트랩 (제일 먼저 1회 실행) =========================================
# v2 러너를 라이브러리로 재사용한다. 노트북 런타임(g)을 1회 로드해 모든 셀이 공유한다.
import os, sys, time, uuid
from pathlib import Path

# 이 노트북은 scripts/ 에 있다. 어디서 열든 작업 디렉터리를 저장소 루트로 고정해야
# 메인 노트북의 상대경로(agent_data/...: DB·chroma)가 깨지지 않는다.
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
print("작업 디렉터리:", Path.cwd())

import run_manufacturing_scenarios as base
import run_manufacturing_scenarios_v2 as v2
# 각 시나리오 셀이 인라인으로 사용하는 심볼들
from run_manufacturing_scenarios import Turn, Scenario, FEATURES_HIGH_RISK

g = v2._load_runtime()                      # manufacturing_agent_v6.ipynb 런타임 구성(LLM 포함)
#run_id = str(int(time.time()))
RESULTS: dict[str, bool] = {}               # 셀별 PASS/FAIL 누적
TIMINGS: dict[str, float] = {}              # 셀별 소요 시간(초) 누적

def run_scenario_inline(sc, full_answer: bool = True):
    """셀에서 인라인 정의한 Scenario 1개를 실행하고 결과를 보기 좋게 출력한다."""
    print(f"[{sc.sid}] {sc.description}")
    _t0 = time.perf_counter()
    #ok, failures, results = base.run_scenario(g, sc, run_id)
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    ok, failures, results = base.run_scenario(g, sc, run_id)
    elapsed = time.perf_counter() - _t0
    summary = base.summarize_result(
        sc, ok, failures, results,
        include_full_answer=full_answer, include_trace=False,
    )
    RESULTS[sc.sid] = ok
    TIMINGS[sc.sid] = elapsed
    print(f"PASS ✅ ({elapsed:.2f}s)" if ok else f"FAIL ❌ ({elapsed:.2f}s)")
    for f in failures:
        print("  -", f)
    if summary.get("tasks"):
        print("  tasks =", summary["tasks"])
    if summary.get("gates"):
        print("  gates =", summary["gates"])
    if summary.get("sql_results"):
        compact = [(r["query_type"], r.get("status"), r["rows"]) for r in summary["sql_results"]]
        print("  sql   =", compact)
    if full_answer and summary.get("answer"):
        print("  --- 최종 답변 ---")
        for line in summary["answer"].splitlines():
            print("   ", line)
    return summary

print("부트스트랩 완료.")
'''


def render_turns(sc) -> str:
    if not sc.turns:
        return "[]"
    parts = []
    for t in sc.turns:
        if t.input_features is not None:
            parts.append(f"        Turn({t.message!r}, FEATURES_HIGH_RISK),")
        else:
            parts.append(f"        Turn({t.message!r}),")
    return "[\n" + "\n".join(parts) + "\n    ]"


def render_scenario_cell(sc) -> str:
    lines = ["sc = Scenario("]
    lines.append(f"    sid={sc.sid!r},")
    lines.append(f"    description={sc.description!r},")
    lines.append(f"    turns={render_turns(sc)},")
    lines.append(f"    check={CHECK_EXPR[sc.sid]},")
    if sc.mode != "graph":
        lines.append(f"    mode={sc.mode!r},")
    lines.append(f"    tags={sc.tags!r},")
    lines.append(")")
    lines.append("summary = run_scenario_inline(sc)")
    return "\n".join(lines)


def main() -> int:
    cells: list[dict] = []
    all_scenarios = v2.scenarios()
    missing = {sc.sid for sc in all_scenarios} - set(CHECK_EXPR)
    if missing:
        print("경고: CHECK_EXPR 누락 sid:", sorted(missing))
        return 1
    cells.append(md(
        "# Manufacturing Agent v2 — 시나리오별 실행 노트북 (인라인 정의)\n\n"
        "`scripts/run_manufacturing_scenarios_v2.py` 의 A~D 사용자 시나리오 + R 회귀 트랙을 "
        "**시나리오 1개당 셀 1개**로 분리한 실행형 노트북이다. 각 셀에는 `Scenario(...)` 정의가 "
        "그대로 펼쳐져 있어, 어떤 대화 턴과 어떤 검증을 쓰는지 셀만 보고 알 수 있다.\n\n"
        "사용법:\n"
        "1. 아래 **부트스트랩 셀**을 먼저 1회 실행한다 (런타임 `g` 로드, LLM 포함).\n"
        "2. 원하는 시나리오 셀을 실행한다. 셀 안의 `Scenario(...)` 를 직접 고쳐 실험해도 된다.\n"
        "3. 맨 아래 **요약 셀**로 지금까지 실행한 시나리오의 PASS/FAIL 집계를 본다.\n\n"
        "> 검증 함수 본문은 길어서 `run_manufacturing_scenarios_v2.py` 의 것을 재사용한다(`v2._check_...`).\n"
        "> 이 노트북은 `scripts/_gen_scenarios_nb.py` 로 자동 생성된다."
    ))
    cells.append(code(BOOTSTRAP))

    last_group = None
    for sc in all_scenarios:
        group = next((t for t in sc.tags if t in GROUP_LABELS), None)
        if group != last_group:
            cells.append(md(f"## {GROUP_LABELS.get(group, group or '기타')}"))
            last_group = group

        lines = [f"### `{sc.sid}` — {sc.description}", ""]
        if sc.turns:
            lines.append("**대화 턴:**")
            for i, turn in enumerate(sc.turns, 1):
                feat = " _(+ 입력 피처 FEATURES_HIGH_RISK)_" if turn.input_features else ""
                lines.append(f"{i}. {turn.message}{feat}")
        else:
            lines.append(f"_노드 직접 검증 시나리오 (mode={sc.mode}, 대화 턴 없음)_")
        lines.append("")
        lines.append(f"태그: {', '.join(sc.tags)}")
        cells.append(md("\n".join(lines)))

        cells.append(code(render_scenario_cell(sc)))

    cells.append(md("## 실행 요약"))
    cells.append(code(
        "# 지금까지 이 세션에서 실행한 시나리오의 PASS/FAIL + 소요 시간 집계\n"
        "if not RESULTS:\n"
        "    print(\"아직 실행한 시나리오가 없습니다.\")\n"
        "else:\n"
        "    passed = sum(1 for ok in RESULTS.values() if ok)\n"
        "    total = sum(TIMINGS.values())\n"
        "    print(f\"{passed}/{len(RESULTS)} passed | 총 {total:.2f}s\")\n"
        "    for sid, ok in RESULTS.items():\n"
        "        print(f\"  {'✅' if ok else '❌'} {sid:<34} {TIMINGS.get(sid, 0):6.2f}s\")\n"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    print("생성 완료:", OUT)
    print("셀 수:", len(cells), "| 시나리오 수:", len(all_scenarios))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
