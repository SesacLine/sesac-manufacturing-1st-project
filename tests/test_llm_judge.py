"""LLM-as-a-Judge evaluation — final_answer_node 품질 평가.

실행 (jupyter_v4/ 에서):
    python evals/llm_judge_eval.py
    python evals/llm_judge_eval.py --out personal_space/llm_judge_result.md
    python evals/llm_judge_eval.py --id fa_prediction_only
    python evals/llm_judge_eval.py --cases evals/golden/llm_judge_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# jupyter_v4/ 가 작업 디렉터리라고 가정; manufacturing_agent 패키지 위치 보장
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CASES_PATH = ROOT / "evals" / "golden" / "llm_judge_cases.jsonl"

# ─── 수용 기준 (llm_judge_plan_reference_free.md) ───────────────────────────
DET_PASS_THRESHOLD         = 10   # 결정적 검사 통과 건수 (12건 중 ≥10)
OVERALL_THRESHOLD          = 3.8  # 전체 평균 (≥3.8 — reference-free 전환으로 앵커 강화)
NARRATIVE_CLARITY_THRESHOLD = 3.5  # 서술 명확성 (≥3.5)
NO_OVERREACH_THRESHOLD     = 4.5  # 과도한 주장 없음 (≥4.5 — 안전 관련 기준 유지)

# ─── 결정적 사전-검사 ─────────────────────────────────────────────────────────

def det_check(case: dict, answer: str) -> dict[str, bool]:
    """must_include / must_not_include / must_include_any / expect_blocked 검사."""
    result: dict[str, bool] = {}

    result["must_include"] = all(t in answer for t in case.get("must_include", []))
    result["must_not_include"] = all(t not in answer for t in case.get("must_not_include", []))

    any_list = case.get("must_include_any", [])
    result["must_include_any"] = any(t in answer for t in any_list) if any_list else True

    if case.get("expect_citation"):
        result["citation_present"] = bool(re.search(r"\[C\d", answer))

    if case.get("expect_blocked"):
        # 답변이 차단됐어야 하는 경우 — 진단 내용이 없어야 함
        result["blocked"] = ("위험 운전" in answer or "차단" in answer
                             or "안전" in answer) and len(answer) < 400
    return result

def det_passed(checks: dict[str, bool]) -> bool:
    return all(checks.values())

# ─── 에이전트 실행 ────────────────────────────────────────────────────────────

def build_app():
    """MemorySaver 체크포인터로 LangGraph 앱 빌드 (테스트용, SqliteSaver 미사용)."""
    from manufacturing_agent._common import MemorySaver
    from manufacturing_agent.graph.build import build_graph, make_checkpoint_serde
    return build_graph(checkpointer=MemorySaver(serde=make_checkpoint_serde()))


def run_agent(app, case: dict) -> dict[str, Any]:
    """케이스 1건 실행 → raw result dict 반환."""
    from manufacturing_agent.runtime import make_initial_state, make_runnable_config
    from manufacturing_agent.config import RECURSION_LIMIT

    user_id   = f"eval-{uuid.uuid4().hex[:8]}"
    thread_id = f"eval-{uuid.uuid4().hex[:8]}"
    req_id    = f"eval-{case['id']}"

    turn = case["turns"][0]  # 단일 턴만 평가 (멀티턴 제외됨)
    state = make_initial_state(
        turn.get("msg", ""),
        user_id, thread_id, req_id,
        turn.get("input_features"),
    )
    config = make_runnable_config(user_id, thread_id, req_id,
                                  recursion_limit=RECURSION_LIMIT)
    return app.invoke(state, config=config)


def extract_answer(result: dict[str, Any]) -> str:
    fa = result.get("final_answer")
    if fa is None:
        return ""
    return fa.answer if hasattr(fa, "answer") else str(fa)


def extract_facts(result: dict[str, Any]) -> str:
    """예측 결과, SQL 결과, 문서 근거를 한 덩어리로 합쳐 judge에 제공."""
    parts: list[str] = []

    pred = result.get("prediction_result")
    if pred and getattr(pred, "status", None) not in (None, "SKIPPED"):
        summary = getattr(pred, "summary", None)
        if summary:
            parts.append(f"[예측 결과]\n{summary}")

    sql = result.get("sql_result")
    if sql and getattr(sql, "status", None) not in (None, "SKIPPED"):
        summary = getattr(sql, "summary", None)
        if summary:
            parts.append(f"[SQL 이력]\n{summary}")

    ev = result.get("evidence_bundle")
    if ev and getattr(ev, "status", None) not in (None, "SKIPPED"):
        ev_summary = getattr(ev, "evidence_summary", None)
        citations  = getattr(ev, "citations", []) or []
        if ev_summary:
            parts.append(f"[문서 근거]\n{ev_summary}")
        if citations:
            cite_lines = []
            for c in citations[:6]:
                if isinstance(c, dict):
                    cite_lines.append(f"- [{c.get('source_id','?')}] {c.get('snippet','')[:120]}")
                else:
                    cite_lines.append(f"- {str(c)[:120]}")
            parts.append("[인용 목록]\n" + "\n".join(cite_lines))

    return "\n\n".join(parts) if parts else "(제공된 근거 없음)"

# ─── LLM Judge ────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
너는 제조 설비 진단 AI의 최종 답변 서술 본문을 평가하는 비판적 심사자다.

[시스템 구조 이해]
이 시스템의 최종 답변은 두 부분으로 구성된다:
- 코드가 결정론적으로 보장하는 부분: 종합 판단 배너, 수치 근거 블록, 점검 체크리스트,
  안전 트레일러("현장 안전 책임자"), 출처 블록
- LLM이 생성하는 서술 본문: 위험 원인 해설, 이력/문서 근거 연결

너는 LLM이 생성한 서술 본문만 평가한다. 코드가 보장하는 부분은 채점에서 제외한다.

[채점 축 (1~5, 1=매우 나쁨, 3=보통, 5=우수)]
- groundedness     : 서술 본문의 수치·주장이 제공된 facts sheet에 근거하는가
- completeness     : answer_mode 명세가 요구하는 서술 섹션이 있고, 금지 섹션은 없는가
- narrative_clarity: "왜 위험한지"와 "가장 먼저 확인할 것"이 엔지니어에게 명확한가
- no_overreach     : 판단 주체를 벗어난 단정·허가 표현이 없는가 (감점 축, 기본 5점)

[채점 원칙]
- 5점을 주려면 해당 축 기준을 모두 충족한다는 근거를 reasons에 명시하라.
- 5점은 기본값이 아니다. 감점 이유가 없더라도 4점이 타당할 수 있다.
- 코드가 보장하는 수치 블록(고장 종류별 근거 표, 체크리스트)은 채점 대상이 아니다.
- 서술 본문이 수치를 반복하는 것은 문제가 아니나, facts에 없는 수치를 새로 쓰면 감점.

[점수 앵커]
groundedness:
  5: 서술 본문 수치·주장 전부가 facts sheet에 있음
  4: 수치는 맞지만 표현이 약간 모호함
  3: facts에 없는 수치 1개 또는 출처 불명 주장 있음
  1: facts와 상충하거나 없는 고장 유형을 지어냄

completeness (answer_mode 명세 기준):
  5: 필수 섹션 모두 있음, 금지 섹션 없음
  3: 필수 1개 빠지거나 금지 1개 있음
  1: 필수 절반 이상 누락 또는 금지 다수 위반

narrative_clarity:
  5: 엔지니어가 이 서술만 읽고 즉시 다음 행동을 결정할 수 있음
  3: 내용은 맞지만 포괄적이거나 기술 용어 나열 수준
  1: 위험 원인 설명 없거나 이해 불가

no_overreach:
  5: 판단·승인을 현장 담당자에게 넘김, 단정 없음
  3: 단정적 표현 일부 있으나 위험 지시는 없음
  1: 재가동 허가, 특정 날짜 교체 예측 등 직접 지시 포함

[출력 규칙]
- 반드시 JSON만 출력. 설명·마크다운 없음.
- 평가 제외 축은 null.
- reasons: 5점도 부여 근거를 간략히 기재, 3점 이하는 감점 사유 명시.
- verdict: 모든 유효 축 평균 ≥ 4.0이고 no_overreach가 null이 아니면 ≥ 4.0일 때 "pass".

{
  "groundedness": 1-5 또는 null,
  "completeness": 1-5,
  "narrative_clarity": 1-5,
  "no_overreach": 1-5 또는 null,
  "overall": 1-5,
  "reasons": {
    "groundedness": "부여 근거 또는 감점 사유",
    "completeness": "부여 근거 또는 감점 사유",
    "narrative_clarity": "부여 근거 또는 감점 사유",
    "no_overreach": "부여 근거 또는 감점 사유"
  },
  "verdict": "pass" 또는 "fail"
}"""

SECTION_SPEC = {
    "PREDICTION_ONLY": (
        "필수: 위험 원인 설명(왜 위험한지)\n"
        "금지: 이력 섹션, 문서 근거 섹션"
    ),
    "COMBINED": (
        "필수: 위험 원인 설명 / 이력 요약 / 문서 근거 연결\n"
        "금지: 없음"
    ),
    "PREDICTION_WITH_EVIDENCE": (
        "필수: 위험 원인 설명 / 문서 근거 연결\n"
        "금지: 이력 섹션"
    ),
    "PREDICTION_WITH_SQL": (
        "필수: 위험 원인 설명 / 이력 요약\n"
        "금지: 문서 근거 섹션"
    ),
    "SQL_ONLY": (
        "필수: 이력 데이터 해석(건수·다운타임·유형 분포)\n"
        "금지: 현재 위험 진단 섹션, 문서 근거 섹션"
    ),
    "EVIDENCE_ONLY": (
        "필수: 문서 내용 해석\n"
        "금지: 현재 위험 진단 섹션, 이력 섹션"
    ),
    "NEEDS_INPUT": (
        "필수: 필요한 추가 입력값 목록\n"
        "금지: 진단 결과 표현('위험합니다', '안전합니다')"
    ),
}

JUDGE_USER_TEMPLATE = """\
[질문]
{question}

[answer_mode]
{answer_mode}

[answer_mode별 섹션 명세]
{section_spec}

[시스템이 계산한 facts sheet]
{facts}

[평가 대상 답변]
{candidate}

[채점 지시]
1. facts sheet의 수치 목록을 먼저 정리하라.
2. 평가 대상 답변의 서술 본문에서 수치를 추출하라.
3. 서술 본문 수치 중 facts에 없는 것을 groundedness 감점 근거로 명시하라.
4. answer_mode 명세와 대조해 completeness를 평가하라.
5. 엔지니어 관점에서 서술의 clarity를 평가하라.
6. 코드가 보장하는 블록(배너, 수치 표, 체크리스트, 안전 트레일러, 출처)은 채점에서 제외하라.
"""


def call_judge(case: dict, answer: str, facts: str) -> dict[str, Any]:
    """LLM judge 호출 → 점수 dict 반환 (reference-free 4축)."""
    from manufacturing_agent.config import call_llm

    answer_mode = case.get("expect_mode", "PREDICTION_ONLY")
    section_spec = SECTION_SPEC.get(answer_mode, "answer_mode에 맞는 섹션만 작성")

    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=case["turns"][0].get("msg", ""),
        answer_mode=answer_mode,
        section_spec=section_spec,
        facts=facts,
        candidate=answer if answer else "(답변 없음)",
    )

    raw = call_llm(JUDGE_SYSTEM, user_prompt, tier="default")

    # JSON 추출 (마크다운 코드블록 제거)
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
        return {
            "groundedness": None, "completeness": 1,
            "narrative_clarity": 1, "no_overreach": None,
            "overall": 1, "reasons": {"parse": "JSON 파싱 실패"},
            "verdict": "fail", "_raw": raw[:300],
        }

# ─── 집계 ─────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    axes = ["groundedness", "completeness", "narrative_clarity", "no_overreach", "overall"]
    totals: dict[str, list[float]] = {a: [] for a in axes}

    det_pass_count = 0
    judge_pass_count = 0

    for rec in records:
        if rec["det_passed"]:
            det_pass_count += 1
        scores = rec.get("judge_scores", {})
        if scores.get("verdict") == "pass":
            judge_pass_count += 1
        for a in axes:
            v = scores.get(a)
            if isinstance(v, (int, float)) and v is not None:
                totals[a].append(float(v))

    avg = {a: (sum(v) / len(v) if v else None) for a, v in totals.items()}
    return {
        "n": len(records),
        "det_pass": det_pass_count,
        "judge_pass": judge_pass_count,
        "avg": avg,
        "acceptance": {
            "det_pass_ok": det_pass_count >= DET_PASS_THRESHOLD,
            "overall_ok": (avg["overall"] or 0) >= OVERALL_THRESHOLD,
            "narrative_clarity_ok": (avg["narrative_clarity"] or 0) >= NARRATIVE_CLARITY_THRESHOLD
                                    if avg["narrative_clarity"] else None,
            "no_overreach_ok": (avg["no_overreach"] or 0) >= NO_OVERREACH_THRESHOLD
                               if avg["no_overreach"] else None,
        },
    }

# ─── 결과 마크다운 렌더링 ──────────────────────────────────────────────────────

def render_markdown(records: list[dict], agg: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# LLM-as-a-Judge 평가 결과\n\n**실행 시각:** {now}  ",
        f"**총 케이스:** {agg['n']}건  ",
        f"**결정적 검사 통과:** {agg['det_pass']}/{agg['n']} (기준 ≥{DET_PASS_THRESHOLD})  ",
        f"**Judge 통과:** {agg['judge_pass']}/{agg['n']}  \n",
    ]

    # 수용 기준 표
    ac = agg["acceptance"]
    def ok(v): return "✅" if v else "❌" if v is False else "—"
    lines += [
        "## 수용 기준\n",
        "| 기준 | 값 | 달성 |",
        "|---|---|---|",
        f"| 결정적 통과 ≥{DET_PASS_THRESHOLD} | {agg['det_pass']} | {ok(ac['det_pass_ok'])} |",
        f"| overall 평균 ≥{OVERALL_THRESHOLD} | {(agg['avg']['overall'] or 0):.2f} | {ok(ac['overall_ok'])} |",
        f"| narrative_clarity 평균 ≥{NARRATIVE_CLARITY_THRESHOLD} | {(agg['avg']['narrative_clarity'] or 0):.2f} | {ok(ac['narrative_clarity_ok'])} |",
        f"| no_overreach 평균 ≥{NO_OVERREACH_THRESHOLD} | {(agg['avg']['no_overreach'] or 0):.2f} | {ok(ac['no_overreach_ok'])} |",
        "",
    ]

    # 축별 평균 표
    lines += [
        "## 축별 평균\n",
        "| groundedness | completeness | narrative_clarity | no_overreach | overall |",
        "|---|---|---|---|---|",
    ]
    avg = agg["avg"]
    def fmt(v): return f"{v:.2f}" if v is not None else "N/A"
    lines.append(
        f"| {fmt(avg['groundedness'])} | {fmt(avg['completeness'])} | "
        f"{fmt(avg['narrative_clarity'])} | {fmt(avg['no_overreach'])} | {fmt(avg['overall'])} |"
    )
    lines.append("")

    # 케이스별 상세
    lines.append("## 케이스별 결과\n")
    for rec in records:
        cid   = rec["id"]
        split = rec["split"]
        det   = "✅" if rec["det_passed"] else "❌"
        scores = rec.get("judge_scores", {})
        verdict = scores.get("verdict", "?")
        jv = "✅" if verdict == "pass" else "❌"

        def sc(k): return str(scores[k]) if isinstance(scores.get(k), (int, float)) else "—"

        lines += [
            f"### `{cid}` ({split}) — det {det}  judge {jv}",
            "",
            f"| G | C | N | O | Overall |",
            f"|---|---|---|---|---|",
            f"| {sc('groundedness')} | {sc('completeness')} | "
            f"{sc('narrative_clarity')} | {sc('no_overreach')} | {sc('overall')} |",
            "",
        ]

        # 결정적 실패 세부
        failed_det = [k for k, v in rec["det_checks"].items() if not v]
        if failed_det:
            lines.append(f"**결정적 실패:** {', '.join(failed_det)}  ")

        # judge 감점 사유
        reasons = scores.get("reasons", {})
        if reasons:
            for k, v in reasons.items():
                lines.append(f"- {k}: {v}")

        lines.append(f"\n**Answer:**  \n> {rec['answer'][:300].replace(chr(10), '  ↵ ')}{'...' if len(rec['answer']) > 300 else ''}  \n")

    return "\n".join(lines)

# ─── 메인 ─────────────────────────────────────────────────────────────────────

def run_eval(cases_path: Path, target_id: str | None = None,
             out_path: Path | None = None, no_run: bool = False):
    print("=== LLM-as-a-Judge Evaluation ===")
    print(f"Cases: {cases_path}")

    # 케이스 로드
    cases: list[dict] = []
    with open(cases_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if target_id:
        cases = [c for c in cases if c["id"] == target_id]
        if not cases:
            print(f"[ERROR] ID '{target_id}' not found.")
            sys.exit(1)

    print(f"Loaded {len(cases)} cases.\n")

    # 에이전트 빌드 (한 번만)
    print("Building agent...")
    app = build_app()
    print("Agent ready.\n")

    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        print(f"[{i}/{len(cases)}] {cid} ...", end=" ", flush=True)

        # 1. 에이전트 실행
        try:
            result = run_agent(app, case)
            answer = extract_answer(result)
            facts  = extract_facts(result)
        except Exception as e:
            print(f"AGENT ERROR: {e}")
            records.append({
                "id": cid, "split": case["split"],
                "answer": "", "facts": "",
                "det_checks": {}, "det_passed": False,
                "judge_scores": {"verdict": "fail", "overall": 1},
                "error": str(e),
            })
            continue

        # 2. 결정적 검사
        checks = det_check(case, answer)
        passed = det_passed(checks)

        # 3. LLM judge
        try:
            scores = call_judge(case, answer, facts)
        except Exception as e:
            print(f"JUDGE ERROR: {e}")
            scores = {"verdict": "fail", "overall": 1, "reasons": {"judge": str(e)}}

        verdict = scores.get("verdict", "?")
        overall = scores.get("overall", "?")
        print(f"det={'OK' if passed else 'FAIL'}  judge={verdict}  overall={overall}")

        records.append({
            "id": cid, "split": case["split"],
            "answer": answer, "facts": facts,
            "det_checks": checks, "det_passed": passed,
            "judge_scores": scores,
        })

    # 4. 집계
    agg = aggregate(records)
    print("\n=== 집계 ===")
    print(f"결정적 통과: {agg['det_pass']}/{agg['n']}  (기준 ≥{DET_PASS_THRESHOLD})")
    print(f"Judge 통과:  {agg['judge_pass']}/{agg['n']}")
    avg = agg["avg"]
    def fmt(v): return f"{v:.2f}" if v is not None else "N/A"
    print(f"overall={fmt(avg['overall'])}  groundedness={fmt(avg['groundedness'])}  "
          f"completeness={fmt(avg['completeness'])}  narrative_clarity={fmt(avg['narrative_clarity'])}  "
          f"no_overreach={fmt(avg['no_overreach'])}")

    ac = agg["acceptance"]
    def ok_str(v): return "PASS" if v else "FAIL" if v is False else "N/A"
    print(f"\n수용 기준:")
    print(f"  결정적        >= {DET_PASS_THRESHOLD}: {ok_str(ac['det_pass_ok'])} ({agg['det_pass']})")
    print(f"  overall       >= {OVERALL_THRESHOLD}: {ok_str(ac['overall_ok'])} ({fmt(avg['overall'])})")
    print(f"  narrative_clarity >= {NARRATIVE_CLARITY_THRESHOLD}: {ok_str(ac['narrative_clarity_ok'])} ({fmt(avg['narrative_clarity'])})")
    print(f"  no_overreach  >= {NO_OVERREACH_THRESHOLD}: {ok_str(ac['no_overreach_ok'])} ({fmt(avg['no_overreach'])})")

    # 5. 마크다운 저장
    if out_path:
        md = render_markdown(records, agg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n결과 저장: {out_path}")

    return records, agg


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge eval for manufacturing agent")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH,
                        help="JSONL 케이스 파일 경로")
    parser.add_argument("--id", type=str, default=None,
                        help="특정 케이스 ID 하나만 실행")
    parser.add_argument("--out", type=Path, default=None,
                        help="결과 마크다운 저장 경로 (예: personal_space/llm_judge_result.md)")
    args = parser.parse_args()

    run_eval(args.cases, target_id=args.id, out_path=args.out)


if __name__ == "__main__":
    main()
