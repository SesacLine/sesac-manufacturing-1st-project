"""
manufacturing_taxonomy.py

AI4I 예측 변수/고장 모드와 Haas RAG 문서 표현을 연결하는 용어 정규화 레이어.

목적
- 자연어 입력에서 제조 현장 용어를 추출한다.
- AI4I 변수명과 Haas 매뉴얼 용어를 같은 개념으로 매핑한다.
- 예측 결과(failure_types, contributing_features)에 따라 우선 검색할 문서와 검색 쿼리를 생성한다.

대상 문서
- Mill Spindle - Troubleshooting Guide (TG0101)
- Mill Chatter - Troubleshooting Guide (TG0100)
- Mechanical Service Manual 96-0283C RevC English June 2007

주의
- Mechanical Service Manual은 2007년 아카이브 문서이므로 실제 수리 절차의 최신성 검증이 필요하다.
- 본 파일은 RAG 검색 라우팅과 용어 정규화 목적이며, 실제 정비 수행 지시를 대체하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------
# 1. 문서 프로필
# ---------------------------------------------------------------------

DOC_PROFILES: Dict[str, Dict[str, object]] = {
    "mechanical_service": {
        "title": "Mechanical Service Manual 96-0283C RevC English June 2007",
        "source_type": "service_manual_archive",
        "domain": [
            "general_machine_troubleshooting",
            "vibration",
            "accuracy",
            "finish",
            "thermal_growth",
            "overheating",
            "coolant",
            "spindle",
            "axis",
            "ballscrew",
            "backlash",
            "tooling",
            "fixturing",
            "machine_level",
        ],
        "best_for": [
            "general troubleshooting",
            "vibration source isolation",
            "accuracy or positioning error",
            "thermal growth and warm-up",
            "poor finish",
            "spindle overheating",
            "coolant system issue",
            "backlash or ballscrew issue",
            "machine level or fixturing issue",
        ],
    },
    "mill_spindle": {
        "title": "Mill Spindle - Troubleshooting Guide",
        "source_type": "troubleshooting",
        "domain": [
            "spindle",
            "temperature",
            "lubrication",
            "belt",
            "encoder",
            "bearing",
            "tool_load",
            "drawbar",
            "toolholder",
        ],
        "best_for": [
            "spindle overheating",
            "spindle vibration",
            "spindle noise",
            "lubrication failure",
            "bearing issue",
            "drawbar force",
            "tool load exceeded",
            "encoder or belt issue",
        ],
    },
    "mill_chatter": {
        "title": "Mill Chatter - Troubleshooting Guide",
        "source_type": "troubleshooting",
        "domain": [
            "chatter",
            "tool_wear",
            "cutting_force",
            "rpm",
            "feedrate",
            "chip_load",
            "toolpath",
            "surface_finish",
        ],
        "best_for": [
            "excessive tool wear",
            "chatter",
            "surface finish issue",
            "cutting force increase",
            "rpm and feedrate adjustment",
            "chip load issue",
            "tool length or tool holder issue",
            "toolpath engagement issue",
        ],
    },
}


# ---------------------------------------------------------------------
# 2. AI4I 변수별 라우팅 기획
# ---------------------------------------------------------------------

FEATURE_ROUTING: Dict[str, Dict[str, object]] = {
    "air_temperature": {
        "ko": "공기 온도",
        "unit": "K",
        "priority_docs": ["mechanical_service", "mill_spindle"],
        "search_intent": "ambient/shop temperature, thermal growth, cooling context, heat dissipation context",
        "notes": (
            "Haas 문서에 Air temperature가 직접 등장하지는 않지만, Mechanical Service Manual의 "
            "ambient temperature of the shop, thermal growth, warm-up 문맥과 연결된다."
        ),
    },
    "process_temperature": {
        "ko": "공정 온도",
        "unit": "K",
        "priority_docs": ["mill_spindle", "mechanical_service"],
        "search_intent": "spindle overheating, spindle temperature, lubrication, thermal growth, heat dissipation",
        "notes": (
            "HDF와 가장 직접적으로 연결된다. Mill Spindle의 spindle temperature/overheat/lubrication과 "
            "Mechanical Service의 overheating/thermal growth를 함께 검색한다."
        ),
    },
    "rotational_speed": {
        "ko": "회전수",
        "unit": "rpm",
        "priority_docs": ["mill_chatter", "mechanical_service", "mill_spindle"],
        "search_intent": "RPM, spindle speed, speed/feed, chip load, vibration, run-in, encoder/belt",
        "notes": (
            "RPM은 Chatter 문서의 chip load, feedrate, spindle speed 문맥과 직접 연결된다. "
            "Mechanical Service Manual에서는 high continuous RPM, run-in program, speed/feed, vibration과 연결된다."
        ),
    },
    "torque": {
        "ko": "토크",
        "unit": "Nm",
        "priority_docs": ["mill_chatter", "mechanical_service", "mill_spindle"],
        "search_intent": "cutting force, cutting load, tool load, heavy cutting, feedrate, depth/width of cut",
        "notes": (
            "Haas 문서에는 Torque보다 cutting force, cutting load, tool load, heavy milling, "
            "speeds and feeds, depth of cut, feedrate로 나타난다."
        ),
    },
    "tool_wear": {
        "ko": "공구 마모 시간",
        "unit": "min",
        "priority_docs": ["mill_chatter", "mechanical_service", "mill_spindle"],
        "search_intent": "excessive tool wear, tool life, tooling condition, chatter, poor finish, toolholder",
        "notes": (
            "TWF와 직접 연결된다. Mill Chatter의 Excessive Tool Wear를 최우선으로, "
            "Mechanical Service의 tooling condition / damaged tooling / poor finish를 보조로 검색한다."
        ),
    },
}


# ---------------------------------------------------------------------
# 3. 변수별 매칭 용어
# ---------------------------------------------------------------------

FEATURE_ALIASES: Dict[str, List[str]] = {
    "air_temperature": [
        "air temperature",
        "ambient temperature",
        "ambient temperature of the shop",
        "shop temperature",
        "room temperature",
        "ambient",
        "cooling",
        "coolant",
        "temperature",
        "heat",
        "thermal",
        "thermal growth",
        "warm-up",
        "warm up",
        "공기 온도",
        "외기 온도",
        "주변 온도",
        "작업장 온도",
        "상온",
        "냉각",
        "열",
        "열팽창",
        "예열",
    ],
    "process_temperature": [
        "process temperature",
        "spindle temperature",
        "temperature at the top of the spindle taper",
        "temperature probe",
        "temperature gun",
        "temperature",
        "overheating",
        "overheat",
        "spindle getting too hot",
        "heat",
        "thermal",
        "thermal expansion",
        "thermal growth",
        "bearing heat",
        "front cap heat",
        "lubrication",
        "spindle lubrication",
        "oil flow",
        "correct amount of lubrication",
        "coolant",
        "coolant system",
        "공정 온도",
        "스핀들 온도",
        "스핀들 테이퍼 온도",
        "과열",
        "발열",
        "열",
        "열팽창",
        "윤활",
        "오일 흐름",
        "냉각",
    ],
    "rotational_speed": [
        "rotational speed",
        "rpm",
        "RPM",
        "spindle speed",
        "speed",
        "speeds and feeds",
        "spindle speed override",
        "high continuous RPM",
        "lower RPM",
        "max RPM",
        "run-in program",
        "spindle warm-up",
        "encoder",
        "encoder belt",
        "drive belt",
        "orientation",
        "chip load",
        "feedrate",
        "회전수",
        "회전 속도",
        "스핀들 속도",
        "스핀들 회전수",
        "RPM",
        "속도",
        "예열 운전",
        "인코더",
        "벨트",
        "칩로드",
        "이송속도",
    ],
    "torque": [
        "torque",
        "tool load",
        "tool load exceeded",
        "cutting load",
        "cutting force",
        "cutting forces",
        "spikes in cutting forces",
        "heavy cutting",
        "heavy milling",
        "aggressive feedrate",
        "aggressive feedrates",
        "feedrate",
        "feed rates",
        "speeds and feeds",
        "depth of cut",
        "depth-of-cut",
        "width of cut",
        "width-of-cut",
        "radial width of cut",
        "tool engagement",
        "load",
        "load spike",
        "spikes in tool load",
        "drawbar force",
        "part/tool imbalance",
        "imbalance",
        "tool deflection",
        "토크",
        "부하",
        "절삭 부하",
        "절삭력",
        "절삭 조건",
        "공구 부하",
        "과부하",
        "이송속도",
        "절입 깊이",
        "절삭 폭",
        "공구 물림",
        "드로바 힘",
        "공구 휨",
    ],
    "tool_wear": [
        "tool wear",
        "excessive tool wear",
        "tool life",
        "tool life management",
        "tooling",
        "tooling condition",
        "damaged tooling",
        "poor quality tooling",
        "tool",
        "cutter",
        "cutter diameter",
        "toolholder",
        "tool holder",
        "toolholder fretting",
        "fretting",
        "tool length",
        "long tool",
        "longer tools",
        "long tooling",
        "tool stick-out",
        "lack of rigidity",
        "surface finish",
        "poor finish",
        "chatter",
        "runout",
        "tool deflection",
        "공구 마모",
        "공구 수명",
        "공구 상태",
        "손상 공구",
        "공구",
        "커터",
        "툴홀더",
        "채터",
        "표면 조도",
        "가공면 품질",
        "런아웃",
        "강성 부족",
    ],
}


# ---------------------------------------------------------------------
# 4. 고장 모드별 매칭 용어
# ---------------------------------------------------------------------

FAILURE_ALIASES: Dict[str, Dict[str, object]] = {
    "TWF": {
        "ko": "공구 마모 불량",
        "description": "Tool Wear Failure",
        "priority_docs": ["mill_chatter", "mechanical_service", "mill_spindle"],
        "aliases": [
            "tool wear",
            "excessive tool wear",
            "tool life",
            "tooling condition",
            "damaged tooling",
            "toolholder",
            "tool holder",
            "fretting",
            "surface finish",
            "poor finish",
            "chatter",
            "tool deflection",
            "공구 마모",
            "공구 수명",
            "공구 상태",
            "손상 공구",
            "채터",
            "표면 조도",
        ],
    },
    "HDF": {
        "ko": "방열 불량",
        "description": "Heat Dissipation Failure",
        "priority_docs": ["mill_spindle", "mechanical_service"],
        "aliases": [
            "heat",
            "overheat",
            "overheating",
            "temperature",
            "spindle temperature",
            "temperature at the top of the spindle taper",
            "thermal growth",
            "thermal expansion",
            "lubrication",
            "oil flow",
            "bearing heat",
            "front cap heat",
            "coolant",
            "coolant system",
            "warm-up",
            "방열",
            "과열",
            "발열",
            "온도",
            "스핀들 온도",
            "열팽창",
            "윤활",
            "냉각",
        ],
    },
    "OSF": {
        "ko": "과부하 불량",
        "description": "Overstrain Failure",
        "priority_docs": ["mill_chatter", "mechanical_service", "mill_spindle"],
        "aliases": [
            "tool load",
            "cutting load",
            "cutting force",
            "heavy cutting",
            "heavy milling",
            "aggressive feedrate",
            "feedrate",
            "speeds and feeds",
            "depth of cut",
            "width of cut",
            "tool engagement",
            "drawbar force",
            "imbalance",
            "vibration",
            "chatter",
            "과부하",
            "절삭 부하",
            "절삭력",
            "절삭 조건",
            "이송속도",
            "채터",
            "진동",
        ],
    },
    "PWF": {
        "ko": "전력 불량",
        "description": "Power Failure",
        "priority_docs": ["mill_spindle", "mechanical_service"],
        "aliases": [
            "motor",
            "spindle motor",
            "spindle drive",
            "vector drive",
            "drive belt",
            "encoder",
            "software",
            "Wye/Delta",
            "contactor",
            "power",
            "I/O Board",
            "processor",
            "gearbox",
            "전력",
            "모터",
            "스핀들 모터",
            "스핀들 드라이브",
            "벡터 드라이브",
            "인코더",
            "벨트",
            "기어박스",
        ],
    },
    "RNF": {
        "ko": "무작위 불량",
        "description": "Random Failure",
        "priority_docs": ["mechanical_service", "mill_spindle", "mill_chatter"],
        "aliases": [
            "unknown",
            "random",
            "unexpected",
            "inspection",
            "checklist",
            "symptom",
            "find the problem first",
            "general machine troubleshooting",
            "무작위",
            "원인 불명",
            "점검",
            "체크리스트",
            "증상",
            "일반 점검",
        ],
    },
}


# ---------------------------------------------------------------------
# 5. 증상/컴포넌트 보조 사전
# ---------------------------------------------------------------------

SYMPTOM_ALIASES: Dict[str, List[str]] = {
    "chatter": [
        "chatter",
        "chattered surface finish",
        "surface finish shows chatter",
        "spindle vibrates",
        "vibration",
        "excessive vibration",
        "resonate",
        "poor finish",
        "finish issue",
        "채터",
        "진동",
        "떨림",
        "표면 거칠기",
        "가공면 불량",
    ],
    "overheating": [
        "overheating",
        "overheat",
        "spindle temperature",
        "temperature at the top of the spindle taper",
        "getting too hot",
        "heat",
        "temperature",
        "thermal expansion",
        "thermal growth",
        "과열",
        "발열",
        "온도 상승",
        "열팽창",
    ],
    "thermal_growth": [
        "thermal growth",
        "thermal expansion",
        "ballscrew expansion",
        "accuracy error",
        "positioning error",
        "warm-up program",
        "ambient temperature of the shop",
        "program feed rates",
        "열팽창",
        "볼스크류 열팽창",
        "정밀도 저하",
        "위치 오차",
        "예열 프로그램",
    ],
    "tool_sticking": [
        "tools stick",
        "toolholder is sticking",
        "tool does not come out",
        "spindle taper",
        "pull stud",
        "thermal expansion of the toolholder",
        "공구가 안 빠짐",
        "툴홀더 고착",
        "스핀들 테이퍼",
        "풀스터드",
    ],
    "load_exceeded": [
        "tool load exceeded",
        "aggressive feedrates",
        "cutting load",
        "cutting forces",
        "heavy cutting",
        "load spike",
        "spikes in tool load",
        "부하 초과",
        "절삭 부하",
        "절삭력",
        "과한 이송",
    ],
    "lubrication_issue": [
        "lubrication",
        "spindle lubrication",
        "oil supply",
        "oil flow",
        "lube pump",
        "leaks",
        "over lubrication",
        "윤활",
        "윤활유",
        "오일 공급",
        "오일 흐름",
        "과윤활",
        "누유",
    ],
    "coolant_issue": [
        "coolant",
        "coolant system",
        "coolant overflow",
        "low coolant",
        "coolant pump",
        "TSC",
        "through-spindle coolant",
        "coolant pressure",
        "coolant lines",
        "coolant tank",
        "냉각수",
        "쿨런트",
        "냉각수 부족",
        "냉각수 펌프",
        "TSC",
    ],
    "accuracy_issue": [
        "accuracy",
        "positioning error",
        "mis-positioning",
        "out-of-round",
        "backlash",
        "machine level",
        "squareness",
        "spindle sweep",
        "thermal growth",
        "정밀도",
        "위치 오차",
        "백래시",
        "수평",
        "직각도",
        "스핀들 스윕",
    ],
}


# ---------------------------------------------------------------------
# 6. 검색 쿼리 생성용 데이터 구조
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TaxonomyMatch:
    key: str
    ko: str
    aliases: List[str]
    priority_docs: List[str]


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if lower not in seen:
            seen.add(lower)
            out.append(normalized)
    return out


def get_feature_match(feature_name: str) -> Optional[TaxonomyMatch]:
    """AI4I feature명을 받아 검색용 alias와 우선 문서를 반환한다."""
    key = feature_name.strip().lower()
    if key not in FEATURE_ROUTING:
        return None

    routing = FEATURE_ROUTING[key]
    return TaxonomyMatch(
        key=key,
        ko=str(routing["ko"]),
        aliases=FEATURE_ALIASES.get(key, []),
        priority_docs=list(routing.get("priority_docs", [])),
    )


def get_failure_match(failure_code: str) -> Optional[TaxonomyMatch]:
    """AI4I failure code(TWF/HDF/OSF/PWF/RNF)를 받아 검색용 alias와 우선 문서를 반환한다."""
    key = failure_code.strip().upper()
    if key not in FAILURE_ALIASES:
        return None

    info = FAILURE_ALIASES[key]
    return TaxonomyMatch(
        key=key,
        ko=str(info["ko"]),
        aliases=list(info.get("aliases", [])),
        priority_docs=list(info.get("priority_docs", [])),
    )


def route_documents(
    features: Sequence[str] | None = None,
    failure_codes: Sequence[str] | None = None,
    symptoms: Sequence[str] | None = None,
) -> List[str]:
    """예측 변수/고장 모드/증상 기준 우선 검색 문서를 반환한다."""
    docs: List[str] = []

    for code in failure_codes or []:
        match = get_failure_match(code)
        if match:
            docs.extend(match.priority_docs)

    for feature in features or []:
        match = get_feature_match(feature)
        if match:
            docs.extend(match.priority_docs)

    # 증상 기반 라우팅 보정
    for symptom in symptoms or []:
        key = symptom.strip().lower()
        if key in {"thermal_growth", "accuracy_issue"}:
            docs.insert(0, "mechanical_service")
        elif key in {"chatter", "load_exceeded"}:
            docs.insert(0, "mill_chatter")
        elif key in {"overheating", "lubrication_issue", "tool_sticking"}:
            docs.insert(0, "mill_spindle")
        elif key == "coolant_issue":
            docs.extend(["mechanical_service", "mill_spindle"])

    if not docs:
        return ["mechanical_service", "mill_spindle", "mill_chatter"]

    return unique_keep_order(docs)


def build_search_terms(
    features: Sequence[str] | None = None,
    failure_codes: Sequence[str] | None = None,
    symptoms: Sequence[str] | None = None,
    max_terms: int = 24,
) -> List[str]:
    """예측 결과와 사용자 증상 표현을 Haas 문서 검색어로 확장한다."""
    terms: List[str] = []

    for code in failure_codes or []:
        match = get_failure_match(code)
        if match:
            terms.append(match.ko)
            terms.extend(match.aliases)

    for feature in features or []:
        match = get_feature_match(feature)
        if match:
            terms.append(match.ko)
            terms.extend(match.aliases)

    for symptom in symptoms or []:
        symptom_key = symptom.strip().lower()
        if symptom_key in SYMPTOM_ALIASES:
            terms.extend(SYMPTOM_ALIASES[symptom_key])
        else:
            terms.append(symptom)

    return unique_keep_order(terms)[:max_terms]


def build_rag_queries(
    user_question: str,
    features: Sequence[str] | None = None,
    failure_codes: Sequence[str] | None = None,
    symptoms: Sequence[str] | None = None,
    max_queries: int = 8,
) -> List[str]:
    """RAG 검색용 query fan-out을 생성한다."""
    terms = build_search_terms(features, failure_codes, symptoms, max_terms=14)
    docs = route_documents(features, failure_codes, symptoms)

    queries = [user_question.strip()]

    if terms:
        queries.append(" ".join(terms[:7]))
        queries.append("troubleshooting " + " ".join(terms[:6]))

    for code in failure_codes or []:
        match = get_failure_match(code)
        if match:
            queries.append(f"{match.ko} 원인 점검 조치 {' '.join(match.aliases[:5])}")

    for feature in features or []:
        match = get_feature_match(feature)
        if match:
            queries.append(f"{match.ko} 관련 점검 {' '.join(match.aliases[:5])}")

    for symptom in symptoms or []:
        if symptom.strip().lower() in SYMPTOM_ALIASES:
            aliases = SYMPTOM_ALIASES[symptom.strip().lower()]
            queries.append(f"{symptom} 증상 점검 {' '.join(aliases[:5])}")

    # 문서 라우팅 힌트가 필요한 경우 query 뒤에 profile 태그처럼 붙여도 된다.
    if docs:
        queries.append("preferred_docs: " + ", ".join(docs))

    return unique_keep_order(queries)[:max_queries]


# ---------------------------------------------------------------------
# 7. 사용 예시
# ---------------------------------------------------------------------

if __name__ == "__main__":
    question = "토크가 높고 공구 마모가 큰데 OSF 위험이면 무엇을 점검해야 해?"
    features = ["torque", "tool_wear"]
    failures = ["OSF"]
    symptoms = ["chatter", "load_exceeded"]

    print("Priority docs:", route_documents(features=features, failure_codes=failures, symptoms=symptoms))
    print("Search terms:", build_search_terms(features=features, failure_codes=failures, symptoms=symptoms))
    print("RAG queries:")
    for q in build_rag_queries(question, features=features, failure_codes=failures, symptoms=symptoms):
        print("-", q)
