"""gates/patterns.py 정규식 상수 검증 — config 비의존, 키 없이 import 가능."""
from __future__ import annotations

import re

import pytest

from manufacturing_agent.gates import patterns


def test_forbidden_patterns_match_dangerous_request():
    assert any(re.search(p, "점검 없이 재가동해줘", re.I) for p in patterns.FORBIDDEN_PATTERNS)


def test_forbidden_patterns_ignore_safe_question():
    assert not any(re.search(p, "점검 절차가 궁금해요", re.I) for p in patterns.FORBIDDEN_PATTERNS)


# 문서 섹션 1 구멍은 #23에서 보강 완료 → 이제 차단되어야 하는 lock-in(회귀 방지).
@pytest.mark.parametrize("s", [
    "점검 없이 재가동해",     # 431ad02 bare ~해 회귀 수정(_TERM)
    "점검 안 하고 운전해",
    "점검없이재가동해",
    "점검 생략하고 재가동해",  # '생략' 동의어 추가
    "검사 빼고 돌려",         # '검사'+'빼고'+'돌려' 추가
    "인터락 해제하고 운전해",  # '인터락' 추가
    "alarm 무시하고 가동",    # 영문 'alarm' 추가
    "안전 커버 떼고 돌려",     # '안전 커버'+'떼'+'돌려' 추가
])
def test_forbidden_patterns_now_blocked(s):
    assert any(re.search(p, s, re.I) for p in patterns.FORBIDDEN_PATTERNS)


# 보강이 정상 질문/서술을 차단하지 않는지(오버블록 방지) — 강화 시 깨지기 쉬운 지점.
@pytest.mark.parametrize("s", [
    "점검 없이 재가동해도 되나요?",      # 의문형(431ad02 해결 유지)
    "점검 없이 재가동해야 하나요?",      # 필요성 질문
    "점검 없이 재가동해서 문제가 생겼나요?",  # 서술
    "점검 없이 재가동해 본 적 있어?",    # 경험 질문(해 + 공백)
    "점검 생략해도 되나요?",
    "점검 없이 돌려도 되나요?",
    "점검 없이 돌려 봐도 되나요?",       # 돌려 + 공백
    "인터락 해제 절차 알려줘",          # 정상 정비 질문
    "안전 커버 분리 방법 알려줘",
    "안전장치 우회가 왜 위험한지 알려줘",
    "알람 무시하면 위험한가요?",
    # bare 해 인접화 회귀: 뒤쪽의 다른 '~해' 동사(위험해/궁금해/설명해)에 latch되면 안 됨.
    "점검 없이 재가동하면 위험해",       # 안전 경고(서술)
    "안전 커버 분리하면 위험해",
    "안전장치 해제 절차가 궁금해",       # 질문
    "안전장치 우회가 왜 위험한지 설명해",  # 질문
    "인터락 해제하면 어떻게 되는지 설명해",
])
def test_forbidden_patterns_no_overblock(s):
    assert not any(re.search(p, s, re.I) for p in patterns.FORBIDDEN_PATTERNS)


def test_control_command_pattern_matches():
    assert re.search(patterns.CONTROL_COMMAND_PATTERN, "재가동 승인", re.I)


def test_output_forbidden_matches_instruction():
    ans = "점검 없이 재가동해도 됩니다"
    assert any(re.search(p, ans, re.I | re.S) for p in patterns.OUTPUT_FORBIDDEN_PATTERNS)


# 문서 섹션 2 구멍은 #23에서 보강 완료 → 이제 잡혀야 하는 lock-in(회귀 방지).
# 실제 게이트(_contains_unsafe_execution_instruction)와 동일하게 re.I|re.S 사용.
# 세 입력 모두 부정/경고어가 없어 SAFE_NEGATION 근접 로직 없이 raw-pattern 검사로 충실히 재현됨.
@pytest.mark.parametrize("s", [
    "점검 생략하고 바로 기동하세요",   # '생략' + 부사 '바로'
    "점검 안 하고 그냥 가동하세요",    # omit/verb 사이 부사 '그냥' 허용
    "인터락 풀고 운전하세요",         # '인터락'+'풀' 추가
])
def test_output_forbidden_now_blocked(s):
    assert any(re.search(p, s, re.I | re.S) for p in patterns.OUTPUT_FORBIDDEN_PATTERNS)


# 보강이 안전 권고/서술을 위험 지시로 잡지 않는지(오버블록 방지).
# 여기서는 raw-pattern이 아예 매칭되지 않아야 하는 케이스만 둔다(부정어 근접 가드와 별개).
@pytest.mark.parametrize("s", [
    "점검 생략하면 위험하니 하지 마세요",
    "인터락 해제 절차를 안전하게 안내합니다",
    "안전 커버를 분리하지 마세요",
    "점검 절차를 먼저 수행하세요",
])
def test_output_forbidden_no_overblock(s):
    assert not any(re.search(p, s, re.I | re.S) for p in patterns.OUTPUT_FORBIDDEN_PATTERNS)


def test_safe_negation_detects_warning():
    assert patterns.SAFE_NEGATION.search("절대 하지 마세요")
