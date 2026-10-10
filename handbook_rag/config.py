"""Central configuration. Every setting can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- data / index ---
    data_dir: Path = ROOT / "data" / "handbook"
    persist_dir: Path = ROOT / ".chroma"
    collection: str = "handbook"

    # --- chunking / retrieval ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # --- providers ---
    # embeddings: hash | huggingface | openai | ollama
    # llm:        extractive | huggingface | openai | ollama
    # "hash" + "extractive" run fully offline with zero downloads: they exist for tests and
    # for checking the plumbing. Use a real provider to get meaningful answers and metrics.
    embeddings: str = "hash"
    llm: str = "extractive"
    embed_model: str = ""   # empty -> provider default (see providers.py)
    llm_model: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            data_dir=Path(_env("RAG_DATA_DIR", str(d.data_dir))),
            persist_dir=Path(_env("RAG_PERSIST_DIR", str(d.persist_dir))),
            collection=_env("RAG_COLLECTION", d.collection),
            chunk_size=int(_env("RAG_CHUNK_SIZE", str(d.chunk_size))),
            chunk_overlap=int(_env("RAG_CHUNK_OVERLAP", str(d.chunk_overlap))),
            top_k=int(_env("RAG_TOP_K", str(d.top_k))),
            embeddings=_env("RAG_EMBEDDINGS", d.embeddings),
            llm=_env("RAG_LLM", d.llm),
            embed_model=_env("RAG_EMBED_MODEL", d.embed_model),
            llm_model=_env("RAG_LLM_MODEL", d.llm_model),
        )

    @property
    def offline_demo(self) -> bool:
        return self.embeddings == "hash" or self.llm == "extractive"
