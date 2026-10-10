import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handbook_rag.config import Settings  # noqa: E402
from handbook_rag.ingest import build_vectorstore, load_documents, split_documents  # noqa: E402
from handbook_rag.providers import HashingEmbeddings, make_extractive_llm  # noqa: E402
from handbook_rag.rag import RagPipeline  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return Settings()


@pytest.fixture(scope="session")
def docs(settings):
    return load_documents(settings.data_dir)


@pytest.fixture(scope="session")
def chunks(docs):
    return split_documents(docs, chunk_size=500, chunk_overlap=50)


@pytest.fixture(scope="session")
def pipeline(chunks):
    store = build_vectorstore(chunks, HashingEmbeddings())
    return RagPipeline(store, make_extractive_llm(), k=4)
