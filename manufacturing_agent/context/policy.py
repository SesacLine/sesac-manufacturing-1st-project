from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.config import *  # noqa: F401,F403

# ---------- context/context_policy.py ----------
STANDARD_FEATURES = ["type", "air_temperature", "process_temperature",
                     "rotational_speed", "torque", "tool_wear"]

# 재사용한 진단 snapshot이 이 시간보다 오래됐으면 stale로 표시한다(센서값 신선도).
STALE_THRESHOLD_SECONDS = int(os.environ.get("CONTEXT_STALE_THRESHOLD_SECONDS", "3600"))

# 최근 대화 윈도우 정책.
# 기억 깊이는 '턴 수'로 예측 가능하게 제어하고(주), 글자 예산은 비정상적으로 긴 턴에 대한 백스톱(보조)이다.
# 사용자 질문은 짧고 중요(의도 추적), assistant 답변은 길어 토큰을 많이 쓰므로 더 적게 유지한다.
RECENT_USER_TURN_LIMIT = int(os.environ.get("CONTEXT_RECENT_USER_TURN_LIMIT", "8"))   # 주 제어: 유지할 최근 사용자 턴 수
RECENT_SUMMARY_CHAR_BUDGET = int(os.environ.get("CONTEXT_RECENT_SUMMARY_CHAR_BUDGET", "3000"))  # 백스톱(8 user + 4 assistant 수용)
PER_TURN_CHAR_CAP = int(os.environ.get("CONTEXT_PER_TURN_CHAR_CAP", "300"))

FEATURE_ALIASES = {
    "공기온도": "air_temperature", "air_temp": "air_temperature",
    "공정온도": "process_temperature", "process_temp": "process_temperature",
    "회전속도": "rotational_speed", "rpm": "rotational_speed", "rotation": "rotational_speed",
    "토크": "torque", "torque": "torque",
    "공구마모": "tool_wear", "tool wear": "tool_wear", "toolwear": "tool_wear",
    "마모": "tool_wear", "회전": "rotational_speed",   # 약식 표현
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


def _alias_pattern(alias: str) -> str:
    """한국어 별칭은 음절 사이 공백을 허용한다('공구 마모'='공구마모'). 영문 별칭은 정확 매칭."""
    a = alias.strip()
    if re.search(r"[가-힣]", a):
        return r"\s*".join(re.escape(ch) for ch in a.replace(" ", ""))
    return re.escape(a)


def _lev(a: str, b: str) -> int:
    """편집거리(Levenshtein) — 이름 오타 보정용."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# 오타 보정 대상(별칭 중 3글자 이상; 너무 짧으면 오매칭 위험)
_FUZZY_TARGETS = {k.replace(" ", ""): v for k, v in FEATURE_ALIASES.items() if v != "type" and len(k.replace(" ", "")) >= 3}


def _fuzzy_canon(word: str) -> Optional[str]:
    w = word.strip().lower().replace(" ", "")
    if w in _FUZZY_TARGETS:
        return _FUZZY_TARGETS[w]
    best, best_d = None, 99
    for k, canon in _FUZZY_TARGETS.items():
        d = _lev(w, k)
        thr = 1 if len(k) <= 4 else 2          # 긴 단어일수록 1~2글자 오타 허용
        if d <= thr and d < best_d:
            best, best_d = canon, d
    return best


# 숫자 토큰: 바로 뒤에 O/o/l/I(숫자 오타로 흔함)가 오면 거부해 '6O'를 6으로 잘못 읽지 않는다.
_NUM = r"([0-9]+(?:\.[0-9]+)?)(?![OolI0-9])"


def extract_machine_values(text: str) -> dict[str, float | str]:
    """자연어에서 'feature = 값'/'feature 값' 추출. 띄어쓰기·약식·이름오타 허용, 값 오타는 거부."""
    out: dict[str, float | str] = {}
    low = text.lower()
    # type L/M/H
    m = re.search(r"\btype\s*[:=]?\s*([lmh])\b", low) or re.search(r"타입\s*[:=]?\s*([lmh상중하])", low)
    if m:
        out["type"] = m.group(1).upper().replace("상", "H").replace("중", "M").replace("하", "L")
    # 1) 별칭(띄어쓰기 허용) + 조사/구분자 + 숫자
    for alias, canon in FEATURE_ALIASES.items():
        if canon == "type":
            continue
        for mm in re.finditer(_alias_pattern(alias) + r"[은는를이가만도:=\s]*" + _NUM, low):
            out[canon] = float(mm.group(1))
    # 2) 이름 오타 보정: 숫자에 인접한 단어를 feature 별칭과 편집거리로 매칭(정상 추출이 없을 때만 보완)
    for mm in re.finditer(r"([가-힣a-z_]{2,})[은는를이가만도:=\s]*" + _NUM, low):
        canon = _fuzzy_canon(mm.group(1))
        if canon and canon not in out:
            out[canon] = float(mm.group(2))
    return out


def detect_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)

