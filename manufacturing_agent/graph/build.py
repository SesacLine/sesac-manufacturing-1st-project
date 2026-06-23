from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.agents.evidence_agent import evidence_agent, sql_agent
from manufacturing_agent.agents.prediction_agent import prediction_agent
from manufacturing_agent.context.manager import context_manager
from manufacturing_agent.contracts.context import AgentContextPacket, ContextCarryoverDecision, ContextDecision, ContextPacket, ContextResolution, DiagnosisContext, EvidenceArtifact, ExecutionPlan, FinalAnswer, GateReport, InputDecision, InputFlags, IntakeDecision, MachineFeatureInput, MachineValue, OrchestratorDecision, OutputSafetyDecision, PredictionResult, RouteDecision, SQLHistoryArtifact, SQLIntentDecision, SQLQueryResult, SupervisorPlannerDecision, SupervisorReplannerDecision, TaskPatch, TaskSpec
from manufacturing_agent.contracts.state import ManufacturingState
from manufacturing_agent.gates.intake_gate import intake_gate
from manufacturing_agent.gates.quality_gates import evidence_gate, output_safety_gate, prediction_gate, sql_gate
from manufacturing_agent.graph.dispatcher import orchestrator_dispatcher, route_after_intake, route_after_orchestrator, route_after_output_safety
from manufacturing_agent.graph.planner import supervisor_planner_node
from manufacturing_agent.graph.replanner import supervisor_replanner_node
from manufacturing_agent.nodes.final_answer_node import final_answer_node
from manufacturing_agent.nodes.memory_writer_node import memory_writer_node

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
    g.add_node("intake_gate", intake_gate)
    g.add_node("context_manager", context_manager)
    g.add_node("supervisor_planner", supervisor_planner_node)
    g.add_node("orchestrator_dispatcher", orchestrator_dispatcher)
    g.add_node("supervisor_replanner", supervisor_replanner_node)
    g.add_node("prediction_agent", _wrap_retry(prediction_agent, "prediction"))
    g.add_node("prediction_gate", prediction_gate)
    g.add_node("evidence_agent", _wrap_retry(evidence_agent, "evidence"))
    g.add_node("evidence_gate", evidence_gate)
    g.add_node("sql_agent", _wrap_retry(sql_agent, "sql"))
    g.add_node("sql_gate", sql_gate)
    g.add_node("final_answer", final_answer_node)
    g.add_node("output_safety_gate", output_safety_gate)
    g.add_node("memory_writer", memory_writer_node)

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
)

def make_checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_SAFE_TYPES)

def make_sqlite_saver(path: str = CHECKPOINT_DB) -> SqliteSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=make_checkpoint_serde())

print("build_graph(Gate-driven Plan-and-Execute + targeted replan + output safety) 정의 완료")
