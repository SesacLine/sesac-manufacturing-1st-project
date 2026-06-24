"""Pydantic request/response models for the Manufacturing Agent API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    input_features: Optional[dict] = None


class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    answer: str
    citations: list = Field(default_factory=list)
    warnings: list = Field(default_factory=list)
    missing_inputs: list = Field(default_factory=list)
    blocked: bool = False
    # SQL 에이전트가 조회한 고장 이력 행(프론트 카드용). SQL 미사용 턴이면 None.
    sql: Optional[dict] = None
    # 실행된 read-only SQL과 반환 행 스냅샷([D#] 데이터 출처). 프론트 drill-down 칩용. SQL 미사용 턴이면 빈 리스트.
    data_refs: list = Field(default_factory=list)
    # evidence(RAG) 에이전트 레벨 메타(status/요약/문서 수). 문서별 근거는 citations에 있다. 미사용 턴이면 None.
    evidence: Optional[dict] = None
    trace: Optional[dict] = None


class ResumeRequest(BaseModel):
    user_id: str
    thread_id: str
