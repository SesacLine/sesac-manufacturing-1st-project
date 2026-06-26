"""Calibration eval — judge 신뢰도 스모크 테스트.

사전 작성된 답변(의도적 결함 포함)을 실제 judge에게 채점시키고,
사람이 정한 reference 점수와 비교한다. judge가 '심은 결함'을 실제로
잡아내는지(=judge를 믿어도 되는지) 검증하는 것이 목적이다.

실행 (jupyter_v4/ 에서):
    .venv\\Scripts\\python.exe evals/scripts/run_calibration_eval.py
    .venv\\Scripts\\python.exe evals/scripts/run_calibration_eval.py --out personal_space/calibration_result.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]   # jupyter_v4/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))       # test_llm_judge 임포트용

DEFAULT_CASES = ROOT / "evals" / "calibration" / "calibration_cases.jsonl"

AXES = ["groundedness", "completeness", "narrative_clarity", "no_overreach"]
ABBR = {"groundedness": "G", "completeness": "C",
        "narrative_clarity": "N", "no_overreach": "O"}
DEFECT_CATCH_MAX = 3   # 결함 축은 judge가 ≤3을 줘야 '잡았다'고 본다


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def judge_case(case: dict) -> dict[str, Any]:
    """실제 judge(test_llm_judge의 프롬프트/모델)로 사전작성 답변을 채점."""
    from test_llm_judge import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE, SECTION_SPEC
    from manufacturing_agent.config import call_llm

    mode = case["answer_mode"]
    spec = SECTION_SPEC.get(mode, "answer_mode에 맞는 섹션만 작성")
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=case.get("question", ""),
        answer_mode=mode,
        section_spec=spec,
        facts=case["facts_sheet"],
        candidate=case["candidate_answer"],
    )
    raw = call_llm(JUDGE_SYSTEM, user_prompt, tier="default")
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"_parse_error": raw[:300]}


def as_int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) else None


def evaluate(cases: list[dict]) -> list[dict]:
    records = []
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        print(f"[{i}/{len(cases)}] {cid} ...", end=" ", flush=True)
        try:
            scores = judge_case(case)
        except Exception as e:
            print(f"JUDGE ERROR: {e}")
            records.append({"case": case, "scores": {"_error": str(e)},
                            "axis_rows": [], "defect_caught": None})
            continue

        ref = case["reference_scores"]
        rows = []
        for axis, ref_v in ref.items():
            jv = as_int(scores.get(axis))
            diff = abs(jv - ref_v) if jv is not None else None
            rows.append({"axis": axis, "ref": ref_v, "judge": jv,
                         "diff": diff,
                         "exact": diff == 0,
                         "within1": diff is not None and diff <= 1})

        defect = case.get("primary_defect", "none")
        defect_caught = None
        if defect != "none":
            jv = as_int(scores.get(defect))
            defect_caught = (jv is not None and jv <= DEFECT_CATCH_MAX)

        # 오탐: clean 케이스인데 judge가 어떤 축이든 ≤3을 줬는가
        false_alarm = False
        if defect == "none":
            false_alarm = any(r["judge"] is not None and r["judge"] <= DEFECT_CATCH_MAX
                              for r in rows)

        print(f"verdict={scores.get('verdict','?')}  "
              + " ".join(f"{ABBR[r['axis']]}={r['judge']}(ref{r['ref']})" for r in rows))
        records.append({"case": case, "scores": scores, "axis_rows": rows,
                        "defect": defect, "defect_caught": defect_caught,
                        "false_alarm": false_alarm})
    return records


def summarize(records: list[dict]) -> dict:
    all_rows = [r for rec in records for r in rec["axis_rows"]
                if r["judge"] is not None]
    n = len(all_rows)
    exact = sum(1 for r in all_rows if r["exact"])
    within1 = sum(1 for r in all_rows if r["within1"])
    mae = (sum(r["diff"] for r in all_rows) / n) if n else None

    defect_recs = [rec for rec in records if rec.get("defect", "none") != "none"]
    defect_caught = sum(1 for rec in defect_recs if rec["defect_caught"])
    clean_recs = [rec for rec in records if rec.get("defect", "none") == "none"]
    false_alarms = sum(1 for rec in clean_recs if rec.get("false_alarm"))

    return {
        "n_axis": n,
        "exact": exact, "exact_pct": (exact / n * 100) if n else 0,
        "within1": within1, "within1_pct": (within1 / n * 100) if n else 0,
        "mae": mae,
        "n_defect": len(defect_recs), "defect_caught": defect_caught,
        "n_clean": len(clean_recs), "false_alarms": false_alarms,
    }


def render_md(records: list[dict], s: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    trustworthy = (s["within1_pct"] >= 70 and s["n_defect"] > 0
                   and s["defect_caught"] == s["n_defect"]
                   and s["false_alarms"] == 0)
    L = [
        "# Calibration 평가 결과 (judge 신뢰도 검증)",
        "",
        f"**실행:** {now}  ",
        f"**케이스:** {len(records)}건  ",
        "**방법:** 사전 작성된 답변(결함 포함)을 실제 judge가 채점 → 사람 reference 점수와 비교  ",
        "",
        "## 종합 판정",
        "",
        f"| 지표 | 값 | 기준 | 판정 |",
        "|---|---|---|---|",
        f"| 결함 적발률 | {s['defect_caught']}/{s['n_defect']} | = 전부 | "
        f"{'✅' if s['n_defect'] and s['defect_caught']==s['n_defect'] else '❌'} |",
        f"| 오탐(clean을 결함으로) | {s['false_alarms']}/{s['n_clean']} | = 0 | "
        f"{'✅' if s['false_alarms']==0 else '❌'} |",
        f"| ±1 일치율 | {s['within1_pct']:.0f}% ({s['within1']}/{s['n_axis']}) | ≥70% | "
        f"{'✅' if s['within1_pct']>=70 else '❌'} |",
        f"| 정확 일치율 | {s['exact_pct']:.0f}% ({s['exact']}/{s['n_axis']}) | 참고 | — |",
        f"| 평균 절대 오차(MAE) | {s['mae']:.2f} | 낮을수록 | — |",
        "",
        f"**결론: judge {'신뢰 가능 ✅' if trustworthy else '주의 — 아래 실패 항목 확인 ⚠️'}**",
        "",
        "> 결함 적발률은 judge가 일부러 심은 결함(환각·과도한 주장·섹션 누락 등)을 "
        "해당 축에서 ≤3점으로 깎았는지를 본다. 하나라도 못 잡으면 그 축의 judge는 신뢰 불가.",
        "",
        "## 케이스별 결과",
        "",
    ]
    for rec in records:
        case = rec["case"]
        sc = rec["scores"]
        cid = case["id"]
        mode = case["answer_mode"]
        defect = case.get("primary_defect", "none")
        verdict = sc.get("verdict", "?")
        if "_error" in sc:
            L += [f"### `{cid}` — ⚠️ JUDGE ERROR", "", f"```\n{sc['_error']}\n```", ""]
            continue
        if "_parse_error" in sc:
            L += [f"### `{cid}` — ⚠️ 파싱 실패", "", f"```\n{sc['_parse_error']}\n```", ""]
            continue

        if defect == "none":
            tag = "🟢 clean" + ("  ❗오탐" if rec.get("false_alarm") else "  ✅오탐없음")
        else:
            tag = (f"🎯 결함={ABBR.get(defect, defect)} "
                   + ("✅ 적발" if rec["defect_caught"] else "❌ 놓침"))
        L += [f"### `{cid}` ({mode}) — {tag}  · judge verdict={verdict}", "",
              "| 축 | reference | judge | 차이 |", "|---|---|---|---|"]
        for r in rec["axis_rows"]:
            mark = "" if r["within1"] else " ⚠️"
            L.append(f"| {ABBR[r['axis']]} {r['axis']} | {r['ref']} | "
                     f"{r['judge']} | {r['diff']}{mark} |")
        reasons = sc.get("reasons", {})
        if reasons:
            L.append("")
            for k, v in reasons.items():
                if k in ABBR:
                    L.append(f"- **{ABBR[k]}**: {v}")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Calibration judge 신뢰도 검증")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("=== Calibration Eval (judge 신뢰도) ===")
    print(f"Cases: {args.cases}")
    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} cases.\n")

    records = evaluate(cases)
    s = summarize(records)

    print("\n=== 요약 ===")
    print(f"결함 적발률: {s['defect_caught']}/{s['n_defect']}")
    print(f"오탐:        {s['false_alarms']}/{s['n_clean']}")
    print(f"±1 일치율:   {s['within1_pct']:.0f}% ({s['within1']}/{s['n_axis']})")
    print(f"정확 일치율: {s['exact_pct']:.0f}%   MAE={s['mae']:.2f}")

    if args.out:
        md = render_md(records, s)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
