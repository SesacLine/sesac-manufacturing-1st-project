from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.graph.build import build_graph, make_sqlite_saver

# SQLite 체크포인터: 노트북 수명 동안 컨텍스트를 유지
sql_saver = make_sqlite_saver(CHECKPOINT_DB)
print("SQLite 체크포인터(SqliteSaver + explicit msgpack allowlist) 활성:", CHECKPOINT_DB)

# SQLite 체크포인터로 컴파일 (세션 간 복원 시연)
app = build_graph(checkpointer=sql_saver)
print("그래프 컴파일 완료")

# Demo execution helpers
import time

DEMO_USER_ID = "demo-user-001"
DEMO_RUN_ID = str(int(time.time()))


def make_runnable_config(user_id: str, thread_id: str, request_id: Optional[str] = None,
                         *, checkpoint_ns: str = "", recursion_limit: int = 50,
                         source: str = "notebook") -> RunnableConfig:
    configurable = {"thread_id": thread_id, "user_id": user_id}
    if checkpoint_ns:
        configurable["checkpoint_ns"] = checkpoint_ns
    config: RunnableConfig = {
        "configurable": configurable,
        "metadata": {"source": source},
        "tags": ["manufacturing-agent"],
        "recursion_limit": recursion_limit,
    }
    if request_id:
        config["metadata"]["run_id"] = request_id
    return config


def make_initial_state(user_message: str, user_id: str, thread_id: str, request_id: str,
                       input_features: Optional[dict] = None) -> ManufacturingState:
    effective_msg = user_message or ("입력된 설비 수치로 고장 위험을 진단해줘." if input_features else "")
    return {
        "request_id": request_id, "user_id": user_id, "thread_id": thread_id,
        "user_message": effective_msg, "input_features": input_features or None,
        "messages": [], "agent_contexts": {}, "gate_reports": [], "retry_counts": {},
        "execution_plan": None, "supervisor_planner_decision": None, "supervisor_replanner_decision": None, "sql_intent_decision": None,
        "orchestrator_decision": None, "active_task_id": None,
        "route": None, "intent": None, "agent_feedback": {}, "consumed_replan_report_index": None,
        "input_decision": None, "intake_decision": None,
    }


def _invoke_from_checkpoint(app_obj, config: RunnableConfig, *, max_resume_attempts: int = 1):
    last_exc = None
    for attempt in range(1, max_resume_attempts + 1):
        try:
            return app_obj.invoke(None, config=config)
        except Exception as exc:
            last_exc = exc
            snapshot = app_obj.get_state(config)
            print(f"checkpoint resume 실패 {attempt}/{max_resume_attempts}: {type(exc).__name__}: {exc}")
            print("남은 graph node:", tuple(snapshot.next or ()))
    raise last_exc


def _invoke_resumable(app_obj, state_in: ManufacturingState, config: RunnableConfig,
                      *, resume_on_error: bool = True, max_resume_attempts: int = 1):
    try:
        return app_obj.invoke(state_in, config=config)
    except Exception as exc:
        snapshot = app_obj.get_state(config)
        print(f"graph 실행 중 실패: {type(exc).__name__}: {exc}")
        print("마지막 checkpoint 이후 남은 graph node:", tuple(snapshot.next or ()))
        if not resume_on_error:
            raise
        if not snapshot.next:
            raise
        return _invoke_from_checkpoint(app_obj, config, max_resume_attempts=max_resume_attempts)


def _print_turn_result(user_message: str, input_features: Optional[dict], result: ManufacturingState,
                       *, debug: bool = False):
    print("=" * 70)
    print("👤 USER:", user_message or "(텍스트 없음 — 구조화 수치 입력)")
    if input_features:
        print("🔢 INPUT FEATURES:", input_features)
    print("-" * 70)
    fa = result.get("final_answer")
    print("🤖 ANSWER:\n" + (fa.answer if fa else "(없음)"))
    if not debug:
        return result
    if fa and fa.citations:
        print("\nCITATIONS:", [c.get("source_id", c) for c in fa.citations])
    if fa and fa.warnings:
        print("WARNINGS:", fa.warnings)
    plan = result.get("execution_plan")
    if plan:
        print("🧭 TASKS:", [(t.task_id, t.task_type, t.status, t.retry_count) for t in plan.tasks])
    pk = result.get("context_packet")
    if pk:
        resolution = pk.context_resolution
        print("\n🧠 ContextMode:", getattr(resolution, "mode", None))
        print("🧠 사용된 입력값:",
              {k: f"{v.value}({'cur' if v.is_current else v.source})"
               for k, v in pk.selected_machine_values.items()})
    print("🚪 GATES:", [(r["gate_name"], r["status"]) for r in result.get("gate_reports", [])])
    return result


def run_turn(user_message: str, user_id: str, thread_id: str, request_id: str,
             input_features: Optional[dict] = None, debug: bool = False,
             *, resume_on_error: bool = True, max_resume_attempts: int = 1,
             checkpoint_ns: str = ""):
    config = make_runnable_config(user_id, thread_id, request_id,
                                  checkpoint_ns=checkpoint_ns, recursion_limit=50)
    state_in = make_initial_state(user_message, user_id, thread_id, request_id, input_features)
    result = _invoke_resumable(app, state_in, config,
                               resume_on_error=resume_on_error,
                               max_resume_attempts=max_resume_attempts)
    return _print_turn_result(user_message, input_features, result, debug=debug)


def resume_turn(user_id: str, thread_id: str, request_id: str = "resume",
                *, debug: bool = False, max_resume_attempts: int = 1,
                checkpoint_ns: str = ""):
    config = make_runnable_config(user_id, thread_id, request_id,
                                  checkpoint_ns=checkpoint_ns, recursion_limit=50)
    result = _invoke_from_checkpoint(app, config, max_resume_attempts=max_resume_attempts)
    return _print_turn_result(result.get("user_message", ""), result.get("input_features"), result, debug=debug)
