"""통합 골든 러너 — evals/golden/*.jsonl 로 컴포넌트별 종합 점수.
대상: routing(EM/F1) · intake(Recall/False-Block) · multiturn(mode/carryover) · text_to_sql(invariant/안전) · rag(Recall@k/MRR).
(final_answer는 비용 큰 LLM-judge라 evals/final_answer_judge.py로 별도 실행.)

실행: PYTHONUTF8=1 PYTHONPATH=. python evals/run_golden.py
"""
from __future__ import annotations
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manufacturing_agent.contracts.context import (ContextPacket, ContextCarryoverDecision, DiagnosisContext,
    TaskSpec, ExecutionPlan, AgentContextPacket)
from manufacturing_agent.graph.planner import _llm_supervisor_planner_decision
from manufacturing_agent.gates.intake_gate import intake_gate
from manufacturing_agent.context.engine import decide_context
from manufacturing_agent.context.policy import extract_machine_values
from manufacturing_agent.agents.sql_agent import sql_agent, SQLSuccess, SQLGeneratedQuery
from manufacturing_agent.services.rag_service import rag_search
from langchain_core.messages import HumanMessage, AIMessage

GD = os.path.join(os.path.dirname(__file__), "golden")
def load(fn): return [json.loads(l) for l in open(os.path.join(GD, fn), encoding="utf-8") if l.strip()]


# ---------- routing ----------
def eval_routing():
    cases = load("routing.jsonl")
    em = 0; lab = {k: {"tp":0,"fp":0,"fn":0,"tn":0} for k in ("needs_prediction","needs_sql","needs_evidence")}
    for c in cases:
        pk = None
        pr = c.get("prior") or {}
        if pr:
            pk = ContextPacket(current_question=c["msg"],
                previous_sql_summary="status=OK ..." if pr.get("previous_sql_summary") else None,
                previous_evidence_summary="status=OK ..." if pr.get("previous_evidence_summary") else None,
                previous_prediction_summary="위험 높음 ..." if pr.get("previous_prediction_summary") else None,
                context_carryover=ContextCarryoverDecision(is_followup=True,
                    uses_previous_sql=bool(pr.get("previous_sql_summary")),
                    uses_previous_evidence=bool(pr.get("previous_evidence_summary")),
                    uses_previous_prediction=bool(pr.get("previous_prediction_summary"))))
        d = _llm_supervisor_planner_decision({"user_message": c["msg"], "input_features": c.get("input_features"), "context_packet": pk})
        exp = c["expect"]; ok = True
        for k in ("needs_prediction","needs_sql","needs_evidence"):
            if k in exp:
                got = getattr(d, k); want = exp[k]
                if got != want: ok = False
                cell = ("tp" if got and want else "tn" if not got and not want else "fp" if got and not want else "fn")
                lab[k][cell] += 1
        if not ok:
            print(f"   ✗ routing {c['id']}: got P{int(d.needs_prediction)}S{int(d.needs_sql)}E{int(d.needs_evidence)} | exp={exp}")
        em += int(ok)
    def f1(s):
        p = s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"] else 0.0
        r = s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"] else 0.0
        return (2*p*r/(p+r)) if p+r else 0.0
    return {"n": len(cases), "EM": em/len(cases),
            "F1": {k: round(f1(s),2) for k,s in lab.items()}}


# ---------- intake ----------
def eval_intake():
    cases = load("intake.jsonl");
    block_tp=block_fn=allow_tn=allow_fp=0; scored=0; obs=[]
    for c in cases:
        exp = c["expect"].get("blocked")
        msgs = []
        pr = c.get("prior") or {}
        if pr.get("previous_prediction_summary"):
            msgs = [HumanMessage(content="이 수치로 진단해줘"), AIMessage(content="종합 판단: 위험 높음. 토크 64, 공구마모 215.")]
        out = intake_gate({"user_message": c["msg"], "messages": msgs, "input_features": c.get("input_features")})
        blocked = bool(getattr(out.get("input_decision"), "blocked", False))
        if exp == "either":
            obs.append((c["id"], blocked)); continue
        scored += 1
        if exp is True:
            block_tp += int(blocked); block_fn += int(not blocked)
            if not blocked: print(f"   ✗ intake {c['id']}: 차단 기대인데 통과")
        else:
            allow_tn += int(not blocked); allow_fp += int(blocked)
            if blocked: print(f"   ✗ intake {c['id']}: 통과 기대인데 차단(reason={getattr(out.get('input_decision'),'reason',None)})")
    danger_recall = block_tp/(block_tp+block_fn) if (block_tp+block_fn) else None
    false_block = allow_fp/(allow_fp+allow_tn) if (allow_fp+allow_tn) else None
    return {"n": len(cases), "scored": scored, "위험차단_Recall": round(danger_recall,2) if danger_recall is not None else None,
            "False_Block률": round(false_block,2) if false_block is not None else None,
            "정확도": round((block_tp+allow_tn)/scored,2) if scored else None, "boundary관찰": obs}


# ---------- multiturn ----------
def eval_multiturn():
    cases = load("multiturn.jsonl"); mode_ok=0; carry_ok=0; carry_n=0; resolved_ok=0; resolved_n=0; scored=0
    for c in cases:
        af = c.get("active_features"); recents = c.get("recent")
        active = DiagnosisContext(id="A", turn_id="t", user_id="u", thread_id="th", features=af or {},
                                  failure_types=[], prediction_summary="이전 진단", created_at="2026-06-23T10:00:00") if af else None
        rc = [DiagnosisContext(id=r["id"], turn_id="t", user_id="u", thread_id="th", features=r.get("features",{}),
                               failure_types=r.get("failure_types",[]), prediction_summary="prev", created_at="2026-06-23T10:00:00")
              for r in (recents or [])]
        if rc and active is None: active = rc[0]
        pr = c.get("prior") or {}
        selected = {"current_values": dict(extract_machine_values(c["msg"])), "active_context": active,
                    "recent_contexts": rc or ([active] if active else []), "recent_turns": [],
                    "previous_prediction_summary": "prev" if pr.get("previous_prediction_summary") else (active.prediction_summary if active else None),
                    "previous_sql_summary": "rows" if pr.get("previous_sql_summary") else None,
                    "previous_evidence_summary": "src" if pr.get("previous_evidence_summary") else None}
        d = decide_context(c["msg"], selected); exp = c["expect"]; case_ok = True; scored += 1
        if "mode" in exp:
            mode_ok += int(d.mode == exp["mode"]);  case_ok &= (d.mode == exp["mode"])
            if d.mode != exp["mode"]:
                print(f"   ✗ multiturn {c['id']}: mode {d.mode} != {exp['mode']}")
        for k in ("is_followup","uses_previous_sql","uses_previous_evidence","uses_previous_prediction"):
            if k in exp:
                carry_n += 1; carry_ok += int(getattr(d,k)==exp[k])
        if "resolved" in exp:
            resolved_n += 1
            rok = all(abs(d.resolved_features.get(kk,-9e9)-vv) < 1e-6 for kk,vv in exp["resolved"].items())
            if "must_not_reuse" in exp: rok = rok and all(kk not in d.resolved_features for kk in exp["must_not_reuse"])
            resolved_ok += int(rok)
        elif "must_not_reuse" in exp:
            resolved_n += 1; resolved_ok += int(all(kk not in d.resolved_features for kk in exp["must_not_reuse"]))
    return {"n": len(cases), "mode정확도": round(mode_ok/scored,2),
            "carryover정확도": round(carry_ok/carry_n,2) if carry_n else None,
            "resolved정확도": round(resolved_ok/resolved_n,2) if resolved_n else None}


# ---------- text_to_sql ----------
def _sql_state(msg, qts):
    task = TaskSpec(task_id="t", task_type="sql", params={"query_types": qts, "default_time_window_days": 30})
    return {"user_message": msg, "context_packet": ContextPacket(current_question=msg),
            "agent_contexts": {"sql_agent": AgentContextPacket(agent_name="sql_agent", current_question=msg, prior_results={})},
            "execution_plan": ExecutionPlan(intent="history_lookup", tasks=[task]), "active_task_id": "t"}

def eval_sql():
    cases = load("text_to_sql.jsonl"); real_ok=real_n=0; safe_ok=safe_n=0
    for c in cases:
        g = c["gold"]
        if c.get("fake_sql"):       # boundary 안전
            safe_n += 1
            runner = lambda *a, **k: SQLSuccess(queries=[SQLGeneratedQuery(query_type=c.get("fake_query_type","detail"),
                purpose="x", sql_query=c["fake_sql"], explanation="x")], reason_summary="x")
            out = sql_agent(_sql_state(c.get("msg","회귀"), ["detail"]), config={"configurable": {"text_to_sql_runner": runner}})
            art = out.get("sql_result"); st = {r.query_type: r.status for r in (getattr(art,"results",[]) or [])}
            status = (list(st.values())[0] if st else None) or getattr(art,"status",None)
            safe_ok += int(status in set(g.get("expect_status_in", ["BLOCKED","FAIL"])))
            continue
        real_n += 1
        out = sql_agent(_sql_state(c["msg"], ["detail","aggregate"]))
        art = out.get("sql_result"); results = getattr(art,"results",[]) or []
        ok = True
        if g.get("must_run"): ok &= (getattr(art,"status",None) in (g.get("expect_status_in") or ["OK","EMPTY"]))
        sqltext = " ".join((r.sql or "").lower() for r in results)
        if g.get("query_type"): ok &= any(r.query_type==g["query_type"] for r in results)
        if g.get("must_have_groupby"): ok &= ("group by" in sqltext)
        if g.get("failure_type_filter"): ok &= (g["failure_type_filter"].lower() in sqltext)
        for col in g.get("must_contain_cols", []): ok &= (col.lower() in sqltext)
        for col in g.get("must_not_contain_cols", []): ok &= (col.lower() not in sqltext)
        if not ok: print(f"   ✗ sql {c['id']}: status={getattr(art,'status',None)} qtypes={[r.query_type for r in results]}")
        real_ok += int(ok)
    return {"n": len(cases), "real_invariant통과": f"{real_ok}/{real_n}", "안전거절률": f"{safe_ok}/{safe_n}"}


# ---------- rag (rag_retrieval_eval 의 로직 재사용 — 단일 소스) ----------
from rag_retrieval_eval import ranked_sources, metrics, PASS_RECALL
def eval_rag():
    cases = load("rag_retrieval.jsonl"); rec=[]; mrr=[]
    for c in cases:
        rel = list(c.get("relevant_doc_ids") or [])
        if not rel: continue
        k = int(c.get("k", 16))
        ranked = ranked_sources(c["query"], k, c.get("profile", "troubleshooting_rag"))
        m = metrics(ranked, rel, k)
        rec.append(m["recall"]); mrr.append(m["mrr"])
        flag = "" if (m["recall"] or 0) >= PASS_RECALL else "  ✗"
        print(f"   · rag {c['id']:<22} R={m['recall']:.2f} MRR={m['mrr']:.2f}{flag}"
              + (f" missed={m['missed']}" if m["missed"] else ""))
    return {"n": len(cases), "scored": len(rec), "Recall@k": round(sum(rec)/len(rec),2) if rec else None,
            "MRR": round(sum(mrr)/len(mrr),2) if mrr else None}


def main():
    import time
    print("골든 종합 평가 실행 중...\n")
    res = {}
    for name, fn in [("routing", eval_routing), ("intake", eval_intake), ("multiturn", eval_multiturn),
                     ("text_to_sql", eval_sql), ("rag", eval_rag)]:
        t = time.perf_counter()
        try: res[name] = fn()
        except Exception as e:
            import traceback; res[name] = {"ERROR": f"{type(e).__name__}: {e}"}; traceback.print_exc()
        print(f"[{name}] ({time.perf_counter()-t:.0f}s) {json.dumps(res[name], ensure_ascii=False)}")
    print("\n=== 종합 대시보드 ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
