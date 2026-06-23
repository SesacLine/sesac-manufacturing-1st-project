from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403
from manufacturing_agent.contracts.context import ContextPacket, EvidenceArtifact, ExecutionPlan, FinalAnswer, InputDecision, InputFlags, IntakeDecision, MachineFeatureInput, OrchestratorDecision, PredictionResult, RouteDecision, SQLHistoryArtifact, SQLIntentDecision, SupervisorPlannerDecision, SupervisorReplannerDecision

# ---------- contracts/state.py ----------
class ManufacturingState(MessagesState, total=False):
    # (상속) messages: Annotated[list[BaseMessage], add_messages]
    request_id: str
    thread_id: str
    user_id: str
    user_message: str
    input_features: Optional[MachineFeatureInput]

    input_decision: Optional[InputDecision]
    input_flags: Optional[InputFlags]
    intake_decision: Optional[IntakeDecision]

    context_packet: Optional[ContextPacket]
    agent_contexts: dict

    execution_plan: Optional[ExecutionPlan]
    supervisor_planner_decision: Optional[SupervisorPlannerDecision]
    supervisor_replanner_decision: Optional[SupervisorReplannerDecision]
    sql_intent_decision: Optional[SQLIntentDecision]
    orchestrator_decision: Optional[OrchestratorDecision]
    active_task_id: Optional[str]
    route: Optional[RouteDecision]
    intent: Optional[str]
    agent_feedback: dict
    consumed_replan_report_index: Optional[int]

    prediction_result: Optional[PredictionResult]
    evidence_bundle: Optional[EvidenceArtifact]
    sql_result: Optional[SQLHistoryArtifact]

    gate_reports: list
    retry_counts: dict

    final_answer: Optional[FinalAnswer]

print("ManufacturingState(MessagesState 상속) 정의 완료")
