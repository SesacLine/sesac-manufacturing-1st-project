"""gates/patterns.py 정규식 상수 검증 — config 비의존, 키 없이 import 가능."""
from __future__ import annotations

import re

from manufacturing_agent.gates import patterns


def test_forbidden_patterns_match_dangerous_request():
    assert any(re.search(p, "점검 없이 재가동해", re.I) for p in patterns.FORBIDDEN_PATTERNS)


def test_forbidden_patterns_ignore_safe_question():
    assert not any(re.search(p, "점검 절차가 궁금해요", re.I) for p in patterns.FORBIDDEN_PATTERNS)


def test_control_command_pattern_matches():
    assert re.search(patterns.CONTROL_COMMAND_PATTERN, "재가동 승인", re.I)


def test_output_forbidden_matches_instruction():
    ans = "점검 없이 재가동해도 됩니다"
    assert any(re.search(p, ans, re.I | re.S) for p in patterns.OUTPUT_FORBIDDEN_PATTERNS)


def test_safe_negation_detects_warning():
    assert patterns.SAFE_NEGATION.search("절대 하지 마세요")
