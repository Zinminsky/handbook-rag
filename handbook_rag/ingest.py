"""Document ingestion: load -> split (section-aware) -> embed -> store in Chroma."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SUPPORTED = {".md", ".txt"}
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def load_documents(data_dir: Path) -> list[Document]:
    """Read every .md / .txt file under data_dir into a LangChain Document."""
    docs = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(Path(data_dir).rglob("*"))
        if p.suffix.lower() in SUPPORTED
    ]
    if not docs:
        raise FileNotFoundError(f"No {sorted(SUPPORTED)} files found in {data_dir}")
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Two-stage split: by markdown headers first (keeps section metadata and topical coherence),
    then by size with a recursive character splitter."""
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for doc in docs:
        sections = header_splitter.split_text(doc.page_content)
        for sec in sections:
            title = " > ".join(sec.metadata[k] for _, k in _HEADERS if k in sec.metadata)
            sec.metadata = {**doc.metadata, "section": title}
        chunks.extend(char_splitter.split_documents(sections))

    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk_id"] = f"{src}:{counters[src]}"
    return chunks


def _open_store(embeddings: Embeddings, collection: str, persist_dir: Path | None) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir) if persist_dir else None,
        collection_metadata={"hnsw:space": "cosine"},  # distances in [0, 2]; 0 = identical
    )


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    collection: str = "handbook",
    persist_dir: Path | None = None,
) -> Chroma:
    """Create a fresh Chroma collection. persist_dir=None gives an in-memory store with a unique
    collection name (used by tests and the chunk-size sweep)."""
    if persist_dir is not None:
        shutil.rmtree(persist_dir, ignore_errors=True)  # rebuild from scratch
    else:
        collection = f"{collection}-{uuid.uuid4().hex[:8]}"
    store = _open_store(embeddings, collection, persist_dir)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return store


def load_vectorstore(embeddings: Embeddings, collection: str, persist_dir: Path) -> Chroma:
    if not Path(persist_dir).exists():
        raise FileNotFoundError(f"No index at {persist_dir}. Run `python -m handbook_rag ingest` first.")
    return _open_store(embeddings, collection, persist_dir)
