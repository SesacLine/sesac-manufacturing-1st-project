import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config.py는 OPENAI_API_KEY가 비어 있으면 import 시점에 RuntimeError를 던진다.
# 키 없는 CI에서도 게이트/그래프 모듈을 import할 수 있도록 더미 키를 심는다.
# 실제 LLM 호출은 테스트에서 stub으로 교체되므로 네트워크로 나가지 않는다.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test-dummy-key"

import json

import pytest


class StubLLM:
    """call_llm 대체용. 스크립트된 응답 문자열을 반환하고 호출 인자를 기록한다.

    - set_json(payload): 다음 호출부터 payload의 JSON 문자열을 반환
    - set_raw(raw): 원시 문자열을 그대로 반환 (파싱 실패 케이스용)
    - calls: 받은 {system, user, tier} 기록 리스트
    """

    def __init__(self) -> None:
        self._response = "{}"
        self.calls: list[dict] = []

    def set_json(self, payload: dict) -> None:
        self._response = json.dumps(payload, ensure_ascii=False)

    def set_raw(self, raw: str) -> None:
        self._response = raw

    def __call__(self, system, user, *, tier="default") -> str:
        self.calls.append({"system": system, "user": user, "tier": tier})
        return self._response


@pytest.fixture
def stub_llm(monkeypatch):
    """주어진 모듈의 call_llm을 새 StubLLM으로 교체하고 그 인스턴스를 반환한다.

    사용: stub = stub_llm(intake_gate_module); stub.set_json({...})
    """

    def _install(module) -> StubLLM:
        stub = StubLLM()
        monkeypatch.setattr(module, "call_llm", stub)
        return stub

    return _install
