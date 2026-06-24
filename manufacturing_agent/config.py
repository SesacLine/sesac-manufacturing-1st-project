from __future__ import annotations
from manufacturing_agent._common import *  # noqa: F401,F403
from manufacturing_agent.observability import record_llm_usage  # OTel LLM 사용량 계측

# ===================== 환경설정 (.env 로드) =====================
# API 키는 프로젝트 루트의 .env 파일에서 읽습니다. (.env.example 참고)
# 키를 이 노트북에 직접 적지 마세요 — .env 파일에만 저장합니다 (git에 커밋되지 않음).
# 실행 순서: 먼저 01_embed_documents_chroma.ipynb 를 실행한 뒤 이 노트북을 실행합니다.
#   .env 예시:  OPENAI_API_KEY=sk-proj-XXXXXXXX...

def load_dotenv(path: str = ".env") -> bool:
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] not in "\"'" and " #" in value:
                value = value.split(" #", 1)[0].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    return True

_ENV_PATH = ".env"
_ENV_EXISTS = os.path.exists(_ENV_PATH)
_ENV_LOADED = load_dotenv(_ENV_PATH)

# LangSmith tracing/upload 설정 (.env에서 LANGSMITH_*를 읽음)
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "true" if LANGSMITH_API_KEY else "false")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "manufacturing-agent")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

# LangChain/LangGraph 쪽 호환 환경변수도 같이 맞춘다.
os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
# =========================================================

# 설정값
DEFAULT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")               # 채팅 모델. 비용 민감 시 "gpt-4o-mini"
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small") # 임베딩 모델. 고품질은 "text-embedding-3-large"
USE_OPENAI_EMBEDDINGS = os.environ.get("USE_OPENAI_EMBEDDINGS", "true").lower() in {"1", "true", "yes", "on"}
DATA_DIR = "agent_data"
os.makedirs(DATA_DIR, exist_ok=True)

LONGTERM_DB = os.path.join(DATA_DIR, "longterm_memory.sqlite")   # 장기 메모리 (대화/실행 이력)
CHECKPOINT_DB = os.path.join(DATA_DIR, "checkpoints.sqlite")     # 장기 체크포인터(SqliteSaver)
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")                    # 벡터 스토어

# ---- 벡터 백엔드 (chroma | pinecone) ----
# rag_service는 vector_search를 import해 쓰며, 백엔드 전환은 이 설정과 import 경로로만 결정된다.
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "chroma").lower()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "sesacline-agent-docs")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
# text-embedding-3-small=1536, text-embedding-3-large=3072 (Pinecone 인덱스 차원과 일치해야 함)
EMBED_DIM = int(os.environ.get("OPENAI_EMBED_DIM", "1536"))

# 오케스트레이션/검색 튜닝 노브 (.env로 override 가능)
RECURSION_LIMIT      = int(os.environ.get("RECURSION_LIMIT", "50"))      # LangGraph 실행 스텝 상한
TASK_MAX_RETRIES     = int(os.environ.get("TASK_MAX_RETRIES", "2"))      # worker gate RETRYABLE_FAIL 재시도 예산
TASK_MAX_RERUNS      = int(os.environ.get("TASK_MAX_RERUNS", "2"))       # targeted replan rerun 예산
RAG_K_DEFAULT        = int(os.environ.get("RAG_K_DEFAULT", "16"))        # 기본 RAG 검색 문서 수
RAG_K_FALLBACK       = int(os.environ.get("RAG_K_FALLBACK", "8"))        # gate feedback 후 보완검색 문서 수
RECENT_CONTEXT_KEEP  = int(os.environ.get("RECENT_CONTEXT_KEEP", "5"))   # thread별 보관하는 DiagnosisContext 수
MEMORY_SUMMARY_CHAR_CAP = int(os.environ.get("SUMMARY_CHAR_CAP", "4000"))  # 장기메모리 요약 본문 길이 상한
# ---- 근거 없음(NO_EVIDENCE) 시 사용자에게 안내할 담당자 연락처 ----
# 하드코딩하지 않고 .env / 환경변수에서 읽는다. 미설정 시 일반 안내만 노출.
SUPPORT_CONTACT_NAME = os.environ.get("SUPPORT_CONTACT_NAME", "설비 정비 담당자")
SUPPORT_CONTACT_EMAIL = os.environ.get("SUPPORT_CONTACT_EMAIL", "")
SUPPORT_CONTACT_PHONE = os.environ.get("SUPPORT_CONTACT_PHONE", "")


def support_contact_text() -> str:
    """NO_EVIDENCE 안내 문구에 붙일 담당자 연락처 한 줄(설정된 항목만)."""
    parts = []
    if SUPPORT_CONTACT_NAME:
        parts.append(SUPPORT_CONTACT_NAME)
    contacts = [c for c in (SUPPORT_CONTACT_EMAIL, SUPPORT_CONTACT_PHONE) if c]
    base = "담당자: " + (" / ".join(parts) if parts else "설비 정비 담당자")
    if contacts:
        base += " (" + ", ".join(contacts) + ")"
    return base


# RAG 라우팅/검색 디버그 로그 on/off (기본 off: 시나리오 출력 오염 방지)
RAG_DEBUG = os.environ.get("RAG_DEBUG", "false").lower() in {"1", "true", "yes", "on"}

_HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))
print("Vector backend:", VECTOR_BACKEND)
if VECTOR_BACKEND == "pinecone":
    print("Pinecone index:", PINECONE_INDEX_NAME, "| API key:", "OK" if PINECONE_API_KEY else "MISSING")

_LANGSMITH_ENABLED = LANGSMITH_TRACING.lower() in {"1", "true", "yes", "on"}
_LANGSMITH_HAS_KEY = bool(os.environ.get("LANGSMITH_API_KEY"))

if _LANGSMITH_ENABLED and _LANGSMITH_HAS_KEY:
    try:
        from langsmith import Client
        _ls_client = Client(api_url=LANGSMITH_ENDPOINT, api_key=LANGSMITH_API_KEY)
        next(_ls_client.list_projects(limit=1), None)
        print("LangSmith upload check: OK")
    except Exception as e:
        print("LangSmith upload check: FAILED", e)
else:
    print("LangSmith upload check: SKIPPED")

# tier별 모델 분리(1-B): 분류기는 저비용 mini, 최종답변은 긴 출력 허용.
# classifier 모델은 .env CLASSIFIER_MODEL 로 덮어쓸 수 있다. output_safety/evidence 요약은 default(강모델) 유지.
_CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL", "gpt-4o-mini")
_LLM_TIER_SPECS = {
    "classifier": {"model": _CLASSIFIER_MODEL, "max_tokens": 1024},
    "default":    {"model": DEFAULT_MODEL,     "max_tokens": 1024},
    "final":      {"model": DEFAULT_MODEL,     "max_tokens": 2048},
}
_llm_clients: dict = {}
_llm_client = None
_USE_REAL_LLM = False
try:
    if not _HAS_KEY:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. 이 노트북은 LLM 설정이 있는 환경을 전제로 실행합니다.")
    from langchain_openai import ChatOpenAI
    for _tier, _spec in _LLM_TIER_SPECS.items():
        _llm_clients[_tier] = ChatOpenAI(model=_spec["model"], temperature=0, max_tokens=_spec["max_tokens"])
    _llm_client = _llm_clients["default"]  # backward compat
    _USE_REAL_LLM = True
except Exception as e:
    raise RuntimeError(f"실제 LLM 초기화 실패: {e}") from e


import time as _time

_LLM_RETRY_MAX = int(os.environ.get("LLM_RETRY_MAX", "6"))

def _is_transient_llm_error(e: Exception) -> bool:
    """429/timeout/connection 등 재시도로 회복 가능한 일시적 오류인지 판정."""
    name = type(e).__name__.lower()
    text = str(e).lower()
    if 'insufficient_quota' in text or 'exceeded your current quota' in text:
        return False  # 쿼터/결제 소진은 재시도해도 회복 불가 → 즉시 실패
    return (
        "ratelimit" in name or "apitimeout" in name or "apiconnection" in name
        or "serviceunavailable" in name or "internalserver" in name
        or "429" in text or "rate limit" in text or "overloaded" in text
        or "timeout" in text or "temporarily" in text or "503" in text or "502" in text
    )

def call_llm(system: str, user: str, *, tier: str = "default") -> str:
    """system+user 프롬프트 → 실제 LLM 텍스트 응답.
    tier: classifier(저비용 mini) | default | final(긴 출력). 일시적 오류(429/timeout)는 지수 백오프로 재시도."""
    if not (_USE_REAL_LLM and _llm_clients):
        raise RuntimeError("LLM client가 초기화되지 않았습니다.")
    client = _llm_clients.get(tier) or _llm_clients["default"]
    model_name = _LLM_TIER_SPECS.get(tier, _LLM_TIER_SPECS["default"])["model"]
    delay, last_exc = 2.0, None
    for attempt in range(_LLM_RETRY_MAX + 1):
        try:
            msg = client.invoke([("system", system), ("human", user)])
            _um = getattr(msg, "usage_metadata", None) or {}
            try:
                record_llm_usage(model_name, tier,
                                 int(_um.get("input_tokens", 0) or 0),
                                 int(_um.get("output_tokens", 0) or 0))
            except Exception:
                pass
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        except Exception as e:
            last_exc = e
            if attempt >= _LLM_RETRY_MAX or not _is_transient_llm_error(e):
                try:
                    record_llm_usage(model_name, tier, error=True)
                except Exception:
                    pass
                raise
            _time.sleep(min(delay, 30.0))
            delay *= 2
    try:
        record_llm_usage(model_name, tier, error=True)
    except Exception:
        pass
    raise last_exc



# import * 가 밑줄(_x) 이름까지 가져오도록 명시 export
__all__ = [n for n in dir() if not n.startswith("__")]
