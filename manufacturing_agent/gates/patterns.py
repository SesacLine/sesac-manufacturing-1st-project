"""게이트 정규식 패턴 모음 (데이터만 — 판정 로직은 각 gate에 둔다).

config에 의존하지 않으므로 OPENAI_API_KEY 없이 import 가능하다.
"""
from __future__ import annotations

import re

# intake_gate: 입력 단계에서 위험 실행 '요청'을 잡는 deterministic backstop
FORBIDDEN_PATTERNS = [
    r"점검\s*(없이|전에?|안\s*하고)\s*(재?가동|기동|운전)",
    r"안전\s*장치\S*\s*(우회|해제|끄|꺼|무시).*(돌려|가동|운전|진행|해)",
    r"(경고|알람|위험)\s*\S*\s*무시.*(가동|운전|계속|진행)",
    r"(재가동|기동|가동)\s*\S*\s*(강행|밀어붙|그냥\s*(해|진행))",
]

# intake_gate: control command 관측 플래그(is_control_command)용
CONTROL_COMMAND_PATTERN = r"가동|재가동|기동|운전|정지|승인|우회|해제|LOTO"

# output_safety_gate: 최종 답변에서 위험 실행 '지시'를 잡는 deterministic backstop
OUTPUT_FORBIDDEN_PATTERNS = [
    r"점검\s*(없이|전에?|안\s*하고)\s*(재?가동|기동|운전).{0,20}(해도\s*(됩니다|된다|돼)|하세요|하라|가능|승인|계속)",
    r"안전\s*장치\S*\s*(우회|해제|끄|꺼|무시).{0,30}(하세요|하라|해도|됩니다|가능|운전|계속|진행)",
    r"(경고|알람|위험)\s*\S*\s*무시.{0,30}(가동|운전|계속|진행|하세요|하라)",
]

# output_safety_gate: 매치 주변 부정/경고어 → 안전 권고로 보고 통과(오차단 방지)
SAFE_NEGATION = re.compile(r"피하|하지\s*마|마라|마세요|말아|금지|않|불가|위험|안\s*됩니다|안\s*돼|삼가|자제|주의")
