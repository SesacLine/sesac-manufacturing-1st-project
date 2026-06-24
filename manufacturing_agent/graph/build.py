from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.agents.evidence_agent import evidence_agent
from manufacturing_agent.agents.sql_agent import sql_agent
from manufacturing_agent.agents.prediction_agent import prediction_agent
from manufacturing_agent.context.manager import context_manager
from manufacturing_agent.contracts.context import AgentContextPacket, ContextCarryoverDecision, ContextDecision, ContextPacket, ContextResolution, DiagnosisContext, EvidenceArtifact, ExecutionPlan, FinalAnswer, GateReport, InputDecision, InputFlags, IntakeDecision, MachineFeatureInput, MachineValue, OrchestratorDecision, OutputSafetyDecision, PredictionResult, RouteDecision, RunTrace, SQLHistoryArtifact, SQLIntentDecision, SQLQueryResult, SupervisorPlannerDecision, SupervisorReplannerDecision, TaskPatch, TaskSpec
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.gates.intake_gate import intake_gate
from manufacturing_agent.gates.quality_gates import evidence_gate, output_safety_gate, prediction_gate, sql_gate
from manufacturing_agent.graph.dispatcher import orchestrator_dispatcher, route_after_intake, route_after_orchestrator, route_after_output_safety
from manufacturing_agent.graph.planner import supervisor_planner_node
from manufacturing_agent.graph.replanner import supervisor_replanner_node
# 실험: LLM-자유 경량 최종 답변 노드로 교체(롤백하려면 아래 import를 final_answer_node로 되돌린다)
# from manufacturing_agent.nodes.final_answer_node import final_answer_node
from manufacturing_agent.nodes.final_answer_llm_node import final_answer_node
from manufacturing_agent.nodes.memory_writer_node import memory_writer_node

# run_trace 관측 설정. RunTrace(contracts.context 정의)를 노드 실행마다 events에 누적한다.
RUN_TRACE_MAX_EVENTS = int(os.environ.get("RUN_TRACE_MAX_EVENTS", "200"))

def _trace_node(name, fn):
    """모든 노드 실행을 run_trace에 1 event씩 누적한다(순차 그래프 전제).
    intake_gate(턴 첫 노드)에서 새 턴 trace를 시작하고, events는 상한으로 잘라 폭주를 막는다."""
    def _inner(state):
        out = fn(state) or {}
        prior = state.get("run_trace")
        prior_events = list(prior.events) if (prior and name != "intake_gate") else []
        event = {
            "node": name,
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "active_task_id": (out.get("active_task_id") if isinstance(out, dict) else None) or state.get("active_task_id"),
        }
        # dispatcher 등에서 라우팅 결정을 함께 남긴다.
        route = out.get("route") if isinstance(out, dict) else None
        if route is not None:
            event["next_node"] = getattr(route, "next_node", None)
        events = (prior_events + [event])[-RUN_TRACE_MAX_EVENTS:]
        out["run_trace"] = RunTrace(request_id=state.get("request_id", "") or "", events=events)
        return out
    return _inner

# 재시도 카운터(관측 + 무한루프 방지). worker 실행마다 +1.
def _wrap_retry(agent_fn, key):
    def _inner(state: ManufacturingState) -> dict:
        out = agent_fn(state)
        rc = dict(state.get("retry_counts", {}))
        rc[key] = rc.get(key, 0) + 1
        out["retry_counts"] = rc
        return out
    return _inner

# ---------- graph/graph.py (Gate-driven Plan-and-Execute) ----------
def build_graph(checkpointer=None):
    g = StateGraph(ManufacturingState)
    # 모든 노드를 _trace_node로 감싸 run_trace에 실행 event를 1개씩 누적한다.
    add = lambda name, fn: g.add_node(name, _trace_node(name, fn))
    add("intake_gate", intake_gate)
    add("context_manager", context_manager)
    add("supervisor_planner", supervisor_planner_node)
    add("orchestrator_dispatcher", orchestrator_dispatcher)
    add("supervisor_replanner", supervisor_replanner_node)
    add("prediction_agent", _wrap_retry(prediction_agent, "prediction"))
    add("prediction_gate", prediction_gate)
    add("evidence_agent", _wrap_retry(evidence_agent, "evidence"))
    add("evidence_gate", evidence_gate)
    add("sql_agent", _wrap_retry(sql_agent, "sql"))
    add("sql_gate", sql_gate)
    add("final_answer", final_answer_node)
    add("output_safety_gate", output_safety_gate)
    add("memory_writer", memory_writer_node)

    g.add_edge(START, "intake_gate")
    g.add_conditional_edges("intake_gate", route_after_intake,
                            {"context_manager": "context_manager", "final_answer": "final_answer"})
    g.add_edge("context_manager", "supervisor_planner")
    g.add_edge("supervisor_planner", "orchestrator_dispatcher")
    g.add_conditional_edges("orchestrator_dispatcher", route_after_orchestrator,
                            {"prediction_agent": "prediction_agent", "evidence_agent": "evidence_agent",
                             "sql_agent": "sql_agent", "supervisor_replanner": "supervisor_replanner",
                             "final_answer": "final_answer"})
    g.add_edge("prediction_agent", "prediction_gate")
    g.add_edge("prediction_gate", "orchestrator_dispatcher")
    g.add_edge("evidence_agent", "evidence_gate")
    g.add_edge("evidence_gate", "orchestrator_dispatcher")
    g.add_edge("sql_agent", "sql_gate")
    g.add_edge("sql_gate", "orchestrator_dispatcher")
    g.add_edge("supervisor_replanner", "orchestrator_dispatcher")
    g.add_edge("final_answer", "output_safety_gate")
    g.add_conditional_edges("output_safety_gate", route_after_output_safety, {"memory_writer": "memory_writer"})
    g.add_edge("memory_writer", END)
    return g.compile(checkpointer=checkpointer)

CHECKPOINT_SAFE_TYPES = (
    MachineValue, DiagnosisContext, ContextResolution,
    ContextCarryoverDecision, ContextDecision, SupervisorPlannerDecision, SQLIntentDecision,
    ContextPacket, AgentContextPacket, PredictionResult, EvidenceArtifact, SQLQueryResult,
    SQLHistoryArtifact, FinalAnswer, InputFlags, InputDecision, IntakeDecision,
    OutputSafetyDecision, MachineFeatureInput, TaskSpec, ExecutionPlan, TaskPatch,
    SupervisorReplannerDecision, OrchestratorDecision, RouteDecision, GateReport,
    RunTrace,
)

def make_checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_SAFE_TYPES)

def make_sqlite_saver(path: str = CHECKPOINT_DB) -> SqliteSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=make_checkpoint_serde())

