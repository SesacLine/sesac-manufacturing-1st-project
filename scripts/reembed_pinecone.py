"""RAG 코퍼스를 Pinecone에 업서트(ingest)한다.

`pinecone_store.vector_search`(런타임 검색)와 동일한 메타데이터/임베딩을 쓰도록 맞춘다.
- 코퍼스 로딩/청킹/제외 규칙은 `reembed_corpus.py`를 그대로 재사용한다(haas PDF 3종 전용).
- OpenAI 임베딩(text-embedding-3-small, 1536) -> Pinecone(cosine) upsert.
- 인덱스가 없으면 자동 생성한다.

사전 준비(.env):
    VECTOR_BACKEND=pinecone
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=sesacline-agent-docs   # (선택, 기본값 동일)
    PINECONE_CLOUD=aws                          # (선택)
    PINECONE_REGION=us-east-1                   # (선택)

실행:
    uv run python scripts/reembed_pinecone.py
    uv run python scripts/reembed_pinecone.py --reset     # 인덱스 삭제 후 재생성
    uv run python scripts/reembed_pinecone.py --dry-run   # 대상 청크만 출력(업서트 안 함)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from manufacturing_agent.config import (
    EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
import reembed_corpus as rc  # 코퍼스 로더/청킹/제외 규칙 재사용

EMBED_BATCH = 100
UPSERT_BATCH = 100


def _embed_batch(oai: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _existing_index_names(pc: Pinecone) -> set[str]:
    try:
        return set(pc.list_indexes().names())
    except Exception:
        return {getattr(ix, "name", None) or (ix.get("name") if isinstance(ix, dict) else None)
                for ix in pc.list_indexes()}


def _ensure_index(pc: Pinecone, name: str, dim: int, reset: bool) -> None:
    names = _existing_index_names(pc)
    if reset and name in names:
        print(f"기존 인덱스 삭제: {name}")
        pc.delete_index(name)
        names.discard(name)
        time.sleep(2)
    if name not in names:
        print(f"인덱스 생성: {name} (dim={dim}, metric=cosine, {PINECONE_CLOUD}/{PINECONE_REGION})")
        pc.create_index(
            name=name, dimension=dim, metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # 인덱스 준비 대기
        for _ in range(60):
            try:
                if pc.describe_index(name).status.get("ready"):
                    break
            except Exception:
                pass
            time.sleep(2)
    else:
        print(f"기존 인덱스 사용: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 코퍼스를 Pinecone에 업서트")
    parser.add_argument("--reset", action="store_true", help="인덱스 삭제 후 재생성")
    parser.add_argument("--dry-run", action="store_true", help="임베딩/업서트 없이 대상 청크만 출력")
    args = parser.parse_args()

    chunks = rc.load_document_chunks()
    by_source = Counter(c["metadata"]["source"] for c in chunks)
    by_type = Counter(c["metadata"]["type"] for c in chunks)
    print(f"대상 청크: {len(chunks)} | types={dict(by_type)}")
    for src, n in by_source.most_common():
        print(f"  {src}: {n}")

    if args.dry_run:
        print("\n[dry-run] 업서트는 수행하지 않았습니다.")
        return 0

    if not PINECONE_API_KEY:
        raise SystemExit("PINECONE_API_KEY가 없습니다. .env에 PINECONE_API_KEY를 설정하세요.")

    oai = OpenAI()  # OPENAI_API_KEY는 환경변수에서 읽음
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 1) 임베딩 (배치)
    print("\n임베딩 생성 중...")
    vectors: list[dict] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        part = chunks[i:i + EMBED_BATCH]
        embs = _embed_batch(oai, [c["text"] for c in part])
        for c, emb in zip(part, embs):
            md = dict(c["metadata"])
            md["text"] = c["text"]  # 런타임 검색이 metadata.text를 반환하므로 함께 저장
            vectors.append({"id": c["id"], "values": emb, "metadata": md})
        print(f"  embedded {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)}")

    dim = len(vectors[0]["values"])

    # 2) 인덱스 준비
    _ensure_index(pc, PINECONE_INDEX_NAME, dim, reset=args.reset)
    index = pc.Index(PINECONE_INDEX_NAME)

    # 3) 업서트 (배치)
    print("\n업서트 중...")
    for i in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[i:i + UPSERT_BATCH])
        print(f"  upserted {min(i + UPSERT_BATCH, len(vectors))}/{len(vectors)}")

    stats = index.describe_index_stats()
    print(f"\n완료: index={PINECONE_INDEX_NAME}, dim={dim}, total_vectors={stats.get('total_vector_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
