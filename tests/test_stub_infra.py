"""StubLLM 픽스처 자체 검증 — call_llm 교체와 스크립트 응답이 동작하는지 확인."""
from __future__ import annotations

from manufacturing_agent.gates import intake_gate as ig


def test_stub_llm_replaces_call_llm(stub_llm):
    stub = stub_llm(ig)
    stub.set_json({"service_allowed": True, "input_reason": "none", "safety_action": "ALLOW"})

    decision = ig._llm_intake("아무 제조 질문")

    assert decision.service_allowed is True
    assert decision.safety_action == "ALLOW"
    assert len(stub.calls) == 1
    assert stub.calls[0]["tier"] == "default"


def test_stub_llm_raw_passthrough(stub_llm):
    stub = stub_llm(ig)
    stub.set_raw("not-json-at-all")

    # _llm_intake는 파싱 실패 시 예외 없이 안전하게 닫는다.
    # "안전 종료"의 계약은 service_allowed가 아니라 safety_action="HUMAN_HANDOFF"이며
    # (service_allowed는 True로 둔다), 다운스트림 _decision_from_intake가 이를 차단으로 이어준다.
    decision = ig._llm_intake("아무 제조 질문")

    assert decision.service_allowed is True
    assert decision.safety_action == "HUMAN_HANDOFF"
    assert len(stub.calls) == 1
