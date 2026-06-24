"""진짜 멀티턴(종단) 평가 — golden/multiturn_seq.jsonl.

각 케이스는 turns 리스트. 같은 user_id/thread_id로 실제 그래프를 순차 invoke해서
턴1의 실제 상태(checkpoint + conversation_store에 저장된 DiagnosisContext/요약)가
턴2로 흐르는 경로 전체를 검증한다. (decide_context 단위 테스트는 multiturn.jsonl 담당.)

turn.expect 지원: context_mode / needs_prediction|sql|evidence / answer_must_include|answer_must_not_include.

실행: PYTHONUTF8=1 PYTHONPATH=. python evals/multiturn_seq_eval.py
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import manufacturing_agent.runtime as rt

GOLD = os.path.join(os.path.dirname(__file__), "golden", "multiturn_seq.jsonl")


def _mode(res):
    pk = res.get("context_packet")
    cr = getattr(pk, "context_resolution", None) if pk else None
    if cr is not None:
        return cr.mode
    pr = res.get("prediction_result")
    return getattr(pr, "context_mode", None)


def _task_types(res):
    plan = res.get("execution_plan")
    return {t.task_type for t in (getattr(plan, "tasks", []) or [])}


def _check_turn(res, exp):
    fails = []
    if "context_mode" in exp:
        got = _mode(res)
        if got != exp["context_mode"]:
            fails.append(f"mode {got}!={exp['context_mode']}")
    types = _task_types(res)
    for key, tt in [("needs_prediction", "prediction"), ("needs_sql", "sql"), ("needs_evidence", "evidence")]:
        if key in exp:
            got = tt in types
            if got != exp[key]:
                fails.append(f"{key} {got}!={exp[key]}")
    ans = res.get("final_answer").answer if res.get("final_answer") else ""
    for t in exp.get("answer_must_include", []):
        if t not in ans:
            fails.append(f"missing '{t}'")
    for t in exp.get("answer_must_not_include", []):
        if t in ans:
            fails.append(f"leak '{t}'")
    return fails


def run_case(c):
    uid, tid = "seqv-u-" + c["id"], "seqv-t-" + c["id"]   # 케이스 내 모든 턴이 같은 thread → 실제 carryover
    out = []
    for i, turn in enumerate(c["turns"], 1):
        turn_id = f"{c['id']}-t{i}"
        res = rt.app.invoke(
            rt.make_initial_state(turn["msg"], uid, tid, turn_id, turn.get("input_features")),
            config=rt.make_runnable_config(uid, tid, turn_id))
        exp = turn.get("expect")
        if not exp:
            out.append((i, "setup", []))
        else:
            fails = _check_turn(res, exp)
            out.append((i, "PASS" if not fails else "FAIL", fails))
    return out


def main():
    cases = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    total = passed = 0
    case_pass = 0
    print("=== 시퀀스(종단) 멀티턴 평가 ===")
    for c in cases:
        rs = run_case(c)
        case_ok = True
        for (i, st, fails) in rs:
            if st == "setup":
                print(f"  · {c['id']} t{i}: setup")
                continue
            total += 1; passed += int(st == "PASS")
            case_ok &= (st == "PASS")
            mark = "✓" if st == "PASS" else "✗"
            print(f"  {mark} {c['id']} t{i}: {st}" + (f"  {fails}" if fails else ""))
        case_pass += int(case_ok)
    print(f"\n어서션 통과: {passed}/{total} | 시퀀스 통과: {case_pass}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
