"""taxonomy 정렬용 RAG 코퍼스 재임베딩 스크립트.

목적
- 현재 chroma 컬렉션을 taxonomy 설계(RAG_RETRIEVAL_ARCHITECTURE.md)에 맞춰 다시 만든다.
- 임베딩 함수/컬렉션 이름/거리 공간은 `manufacturing_agent/rag/chroma.py` 런타임과 동일하게 맞춘다.

코퍼스 정책 (사용자 확정)
- 포함: document/haas 의 3대 핵심 PDF (Mechanical Service Manual, Mill Spindle, Mill Chatter) 전용
- 제외: document/haas_backup/ (옛 HTML 중복 + Vector Drive)
        document/haas 의 "Mill Accuracy" PDF (taxonomy 3대 핵심에 없음)
        (osha/kosha 안전문서는 코퍼스에서 제거됨)
- reset=True 로 옛 stale 청크를 제거하고 새로 임베딩한다.

실행:
    uv run python scripts/reembed_corpus.py
    uv run python scripts/reembed_corpus.py --dry-run   # 임베딩 없이 대상 파일/청크만 출력
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

# repo 루트를 import 경로에 추가 (manufacturing_agent 패키지 사용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

from manufacturing_agent.config import CHROMA_DIR, EMBED_MODEL, USE_OPENAI_EMBEDDINGS

# chroma.py 와 동일한 청킹 상수
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
DOCUMENT_DIR = "document"

# ── 코퍼스 제외 규칙 ─────────────────────────────────────────────────
EXCLUDE_PATH_PARTS = {"haas_backup"}          # 폴더 단위 제외
EXCLUDE_NAME_SUBSTR = ("mill accuracy",)      # 파일명(소문자) 부분일치 제외


def is_excluded(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if parts & EXCLUDE_PATH_PARTS:
        return True
    name = path.name.lower()
    return any(sub in name for sub in EXCLUDE_NAME_SUBSTR)


# ── 문서 로딩/청킹 (01_embed_documents_chroma.ipynb 과 동일 로직) ──────
def doc_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "osha" in parts or "kosha" in parts or "safety" in name or "loto" in name or "guard" in name:
        return "safety"
    if "haas" in parts or "troubleshooting" in name or "diagnostic" in name:
        return "troubleshooting"
    return "concept"


def read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    return soup.get_text("\n")


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) >= 2)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_document_chunks(document_dir: str = DOCUMENT_DIR) -> list[dict]:
    root = Path(document_dir)
    supported = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".html", ".htm", ".pdf", ".txt", ".md"} and not is_excluded(p)
    )
    chunks: list[dict] = []
    for path in supported:
        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"}:
            text = read_html(path)
        elif suffix == ".pdf":
            text = read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        rel = path.relative_to(root).as_posix()
        for idx, chunk in enumerate(chunk_text(text)):
            digest = hashlib.sha1(f"{rel}:{idx}:{chunk[:80]}".encode("utf-8")).hexdigest()[:16]
            chunks.append({
                "id": digest,
                "text": chunk,
                "metadata": {
                    "source": rel,
                    "chunk_index": idx,
                    "type": doc_type(path),
                    "ext": suffix.lstrip("."),
                },
            })
    return chunks


def build_embedding_function():
    """chroma.py 와 동일한 임베딩 함수/컬렉션 이름을 반환한다."""
    if not USE_OPENAI_EMBEDDINGS:
        raise SystemExit(
            "USE_OPENAI_EMBEDDINGS=false 입니다. 이 스크립트는 OpenAI 임베딩 컬렉션을 재구성합니다. "
            ".env에서 USE_OPENAI_EMBEDDINGS=true 로 두고 실행하세요."
        )
    import os
    fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
    )
    return fn, "manufacturing_document_chunks_openai", f"OpenAI({EMBED_MODEL})"


def main() -> int:
    parser = argparse.ArgumentParser(description="taxonomy 정렬용 RAG 코퍼스 재임베딩")
    parser.add_argument("--dry-run", action="store_true", help="임베딩 없이 대상 파일/청크 통계만 출력")
    args = parser.parse_args()

    chunks = load_document_chunks()
    by_source = Counter(c["metadata"]["source"] for c in chunks)
    by_type = Counter(c["metadata"]["type"] for c in chunks)

    print(f"대상 청크: {len(chunks)} | types={dict(by_type)}")
    print("source별 청크 수:")
    for src, n in by_source.most_common():
        print(f"  {src}: {n}")

    if args.dry_run:
        print("\n[dry-run] 임베딩은 수행하지 않았습니다.")
        return 0

    embed_fn, collection_name, embed_label = build_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # reset: 기존 컬렉션 삭제 후 재생성 (옛 stale 청크 제거)
    try:
        client.delete_collection(collection_name)
        print(f"\n기존 컬렉션 삭제: {collection_name}")
    except Exception:
        print(f"\n기존 컬렉션 없음(신규 생성): {collection_name}")

    collection = client.get_or_create_collection(
        collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # chroma.py 검색 score=1-distance 전제와 동일
    )

    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
        )
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}")

    print(f"\n재임베딩 완료: collection={collection_name}, embedding={embed_label}, total={collection.count()}")
    final = Counter(m["source"] for m in collection.get(include=["metadatas"])["metadatas"])
    print("최종 source 분포:")
    for src, n in final.most_common():
        print(f"  {src}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
