"""intake_gate의 도메인 용어 백스톱(_mentions_domain_term) 회귀 테스트.

LLM intake가 짧은 약어 질문("TWF가 뭐야?")을 out_of_scope/gibberish로 오판하는 것을
막기 위한 결정적 백스톱이다. 순수 함수(문자열 → bool)이므로 LLM 없이 검증한다.

실행:
    OPENAI_API_KEY=sk-test USE_OPENAI_EMBEDDINGS=false \
    PYTHONUTF8=1 PYTHONPATH=. uv run python -m pytest tests/test_intake_domain.py -v
"""
from __future__ import annotations

import pytest

from manufacturing_agent.gates.intake_gate import _mentions_domain_term


# --- 도메인 용어/고장 코드 → True(서비스 허용 보정 대상) ---
@pytest.mark.parametrize("msg", [
    "TWF가 뭐야?",          # 약어 + 한국어 조사
    "twf 알려줘",           # 소문자
    "OSF는 무슨 뜻이야?",
    "HDF/PWF 차이 설명해줘",
    "RNF 사례 있어?",
    "공구 마모 점검 방법",
    "과부하가 뭐야",
    "방열 불량 원인",
    "고장 유형별로 정리해줘",
    "설비 진단 부탁해",
])
def test_domain_terms_detected(msg):
    assert _mentions_domain_term(msg) is True


# --- 도메인과 무관 → False(오탐 방지) ---
@pytest.mark.parametrize("msg", [
    "오늘 날씨 어때?",
    "점심 뭐 먹지",
    "SOFTWARE 업데이트 해줘",   # 'OSF' 등이 영문 단어 내부에 있어도 매칭되면 안 됨
    "그냥 인사하려고",
    "",
])
def test_non_domain_not_flagged(msg):
    assert _mentions_domain_term(msg) is False
