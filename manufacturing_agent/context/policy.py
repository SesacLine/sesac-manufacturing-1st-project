from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403

# ---------- context/context_policy.py ----------
STANDARD_FEATURES = ["type", "air_temperature", "process_temperature",
                     "rotational_speed", "torque", "tool_wear"]

# 재사용한 진단 snapshot이 이 시간보다 오래됐으면 stale로 표시한다(센서값 신선도).
STALE_THRESHOLD_SECONDS = int(os.environ.get("CONTEXT_STALE_THRESHOLD_SECONDS", "3600"))

# 최근 대화 요약 토큰 폭주 방지 캡(목표: 비용/지연).
RECENT_SUMMARY_CHAR_BUDGET = int(os.environ.get("CONTEXT_RECENT_SUMMARY_CHAR_BUDGET", "2000"))
PER_TURN_CHAR_CAP = int(os.environ.get("CONTEXT_PER_TURN_CHAR_CAP", "300"))

FEATURE_ALIASES = {
    "공기온도": "air_temperature", "air_temp": "air_temperature",
    "공정온도": "process_temperature", "process_temp": "process_temperature",
    "회전속도": "rotational_speed", "rpm": "rotational_speed", "rotation": "rotational_speed",
    "토크": "torque", "torque": "torque",
    "공구마모": "tool_wear", "tool wear": "tool_wear", "toolwear": "tool_wear",
    "타입": "type", "type": "type",
    # canonical 영문명 직접 입력(air_temperature=300 등)도 추출되도록 자기 별칭 추가
    "air_temperature": "air_temperature", "process_temperature": "process_temperature",
    "rotational_speed": "rotational_speed", "tool_wear": "tool_wear",
}

INJECTION_PATTERNS = [
    r"(이전|위|앞선)\s*(규칙|지시|명령|시스템\s*메시지).*(무시|따르지\s*마)",
    r"(규칙|지시|명령|시스템\s*메시지).*(무시|따르지\s*마)",
    r"(시스템\s*프롬프트|개발자\s*지시|숨겨진\s*규칙).*(출력|공개|무시)",
    r"(안전\s*경고|안전\s*문구).*(제거|빼|하지\s*마)",
    r"ignore\s+(all\s+|the\s+)?previous\s+(instructions|rules|messages)",
    r"disregard\s+(all\s+|the\s+)?(instructions|rules|safety)",
    r"you\s+are\s+now", r"forget\s+(the\s+)?(rules|instructions)",
    r"너는\s*이제", r"역할.*변경",
]

CONTEXT_RULES = """\
1. ContextManager는 항상 실행한다.
2. 전체 이전 대화를 Agent에게 그대로 전달하지 않는다.
3. 현재 입력값이 이전 입력값보다 우선한다.
4. 현재값이 없는 feature만 이전 대화에서 보완한다.
5. 이전 citation은 재사용하지 않는다.
6. EvidenceAgent는 현재 질문 기준으로 문서를 다시 검색한다.
7. prompt injection성 context는 제거한다.
8. 오래된 센서값은 stale 표시한다.
9. token budget 초과 시 설비값/직전 PredictionResult 요약을 우선한다."""


def extract_machine_values(text: str) -> dict[str, float | str]:
    """자연어에서 'feature = 값' 또는 'feature 값' 패턴 추출."""
    out: dict[str, float | str] = {}
    low = text.lower()
    # type L/M/H
    m = re.search(r"\btype\s*[:=]?\s*([lmh])\b", low) or re.search(r"타입\s*[:=]?\s*([lmh상중하])", low)
    if m:
        out["type"] = m.group(1).upper().replace("상", "H").replace("중", "M").replace("하", "L")
    for alias, canon in FEATURE_ALIASES.items():
        if canon == "type":
            continue
        # alias 뒤에 조사(은/는/를/이/가/만/도 등)·구분자가 와도 숫자를 잡는다: "토크만 60", "torque=60"
        for mm in re.finditer(re.escape(alias) + r"[은는를이가만도:=\s]*([0-9]+(?:\.[0-9]+)?)", low):
            out[canon] = float(mm.group(1))
    return out


def detect_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)

print("context_policy 정의 완료")
