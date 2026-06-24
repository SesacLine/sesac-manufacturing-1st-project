"""run_rag_scenarios.py 를 '시나리오=셀' 노트북으로 생성한다(인라인 버전).

manufacturing_scenarios_v2.ipynb 와 동일한 방식:
- 시나리오 1개당 (markdown 설명 셀 + code 정의·실행 셀) 1쌍
- 각 셀에 Scenario(...) 정의를 펼쳐 넣고, check는 run_rag_scenarios(R) 함수를 참조
- 실행 시 소요 시간 + 유사도 score + rag_trace(fan-out/priority/source/score)를 출력

결과: scripts/manufacturing_rag_scenarios.ipynb
재생성: uv run python scripts/_gen_rag_scenarios_nb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_rag_scenarios as R  # noqa: E402

OUT = ROOT / "scripts" / "manufacturing_rag_scenarios.ipynb"

GROUP_LABELS = {
    "cause": "원인 (cause) — 단순 RAG",
    "inspection": "점검 (inspection) — 단순 RAG",
    "maintenance": "정비 (maintenance) — 단순 RAG",
    "preventive": "예방 (preventive) — 단순 RAG",
    "prediction_rag_single": "예측 기반 RAG — 단일 턴 (prediction + evidence)",
    "prediction_rag_multiturn": "예측 기반 RAG — 멀티턴 (1턴 예측 → 2턴 후속 RAG)",
    "empty": "근거 없음 (empty) — NO_EVIDENCE fallback",
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
# RAG 전용 러너(run_rag_scenarios)를 라이브러리로 재사용한다.
# 노트북 런타임(g)을 1회 로드해 모든 시나리오 셀이 공유한다.
import os, sys, time, uuid
from pathlib import Path

# 이 노트북은 scripts/ 에 있다. 어디서 열든 작업 디렉터리를 저장소 루트로 고정해야
# 메인 노트북의 상대경로(agent_data/...: SQLite DB 등)가 깨지지 않는다.
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
print("작업 디렉터리:", Path.cwd())

# (선택) retrieval layer 디버그 로그를 stderr로 보고 싶으면 주석 해제
# os.environ["RAG_DEBUG"] = "true"

import run_manufacturing_scenarios_v2 as v2     # 보정된 DEFINITION_CELLS 로더 재사용
import run_rag_scenarios as R                    # RAG 전용 check / 출력 헬퍼
from run_manufacturing_scenarios import Turn, Scenario, FEATURES_HIGH_RISK

g = v2._load_runtime()                           # manufacturing_agent_v6.ipynb 런타임(LLM 포함)
RESULTS: dict[str, bool] = {}                    # 셀별 PASS/FAIL 누적
TIMINGS: dict[str, float] = {}                   # 셀별 소요 시간(초) 누적


def run_rag(sc, full_answer: bool = True, trace: bool = True):
    """시나리오 1개 실행 — 소요 시간 + 유사도 score + rag_trace 출력."""
    rid = f"rag-{int(time.time())}-{uuid.uuid4().hex[:8]}"   # 매 실행 고유 thread (체크포인트 충돌 방지)
    return R.run_rag_inline(g, sc, rid, RESULTS, TIMINGS, full_answer=full_answer, trace=trace)


print("부트스트랩 완료. RAG 시나리오 수:", len(R.scenarios()))
'''


def render_turns(sc) -> str:
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
    lines.append(f"    check=R.{sc.check.__name__},")
    lines.append(f"    tags={sc.tags!r},")
    lines.append(")")
    lines.append("summary = run_rag(sc)")
    return "\n".join(lines)


def main() -> int:
    all_scenarios = R.scenarios()
    cells: list[dict] = []

    cells.append(md(
        "# Manufacturing Agent — RAG 전용 시나리오 노트북\n\n"
        "`scripts/run_rag_scenarios.py` 의 RAG(문서 근거) 시나리오를 "
        "**시나리오 1개당 셀 1개**로 분리한 실행형 노트북이다. 각 셀에는 `Scenario(...)` 정의가 "
        "펼쳐져 있고, 실행하면 **소요 시간 + 유사도 score + rag_trace(fan-out/priority/source/score)** 를 출력한다.\n\n"
        "사용법:\n"
        "1. 아래 **부트스트랩 셀**을 먼저 1회 실행한다 (런타임 `g` 로드, LLM 포함).\n"
        "2. 원하는 시나리오 셀을 실행한다. 셀 안의 `Scenario(...)` 를 직접 고쳐 실험해도 된다.\n"
        "3. 맨 아래 **요약 셀**로 PASS/FAIL + 소요 시간을 집계한다.\n\n"
        "검증 기준(check)은 `run_rag_scenarios.py` 의 것을 재사용한다(`R._check_*`).\n"
        "이 노트북은 `scripts/_gen_rag_scenarios_nb.py` 로 자동 생성된다."
    ))
    cells.append(code(BOOTSTRAP))

    last_group = None
    for sc in all_scenarios:
        group = next((t for t in sc.tags if t in GROUP_LABELS), None)
        if group != last_group:
            cells.append(md(f"## {GROUP_LABELS.get(group, group or '기타')}"))
            last_group = group

        lines = [f"### `{sc.sid}` — {sc.description}", "", "**대화 턴:**"]
        for i, turn in enumerate(sc.turns, 1):
            feat = " _(+ 입력 피처 FEATURES_HIGH_RISK)_" if turn.input_features else ""
            lines.append(f"{i}. {turn.message}{feat}")
        lines.append("")
        lines.append(f"check: `{sc.check.__name__}` | 태그: {', '.join(sc.tags)}")
        cells.append(md("\n".join(lines)))

        cells.append(code(render_scenario_cell(sc)))

    cells.append(md("## 실행 요약"))
    cells.append(code(
        "# 지금까지 이 세션에서 실행한 RAG 시나리오의 PASS/FAIL + 소요 시간 집계\n"
        "if not RESULTS:\n"
        "    print(\"아직 실행한 시나리오가 없습니다.\")\n"
        "else:\n"
        "    passed = sum(1 for ok in RESULTS.values() if ok)\n"
        "    total = sum(TIMINGS.values())\n"
        "    print(f\"{passed}/{len(RESULTS)} passed | 총 {total:.2f}s\")\n"
        "    for sid, ok in RESULTS.items():\n"
        "        print(f\"  {'✅' if ok else '❌'} {sid:<28} {TIMINGS.get(sid, 0):6.2f}s\")\n"
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
