"""Retrieval-grounded question answering."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions about the Meridian Labs employee handbook.
Rules:
1. Use ONLY the numbered context passages. Never use outside knowledge.
2. If the context does not contain the answer, reply exactly: {REFUSAL}
3. Be concise and cite the passages you used like [1] or [2]."""

HUMAN_PROMPT = "Context:\n{context}\n\nQuestion: {question}"

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


@dataclass
class Retrieved:
    document: Document
    distance: float  # cosine distance: lower = more similar

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "?")


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved: list[Retrieved] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for r in self.retrieved:
            if r.source not in seen:
                seen.append(r.source)
        return seen

    @property
    def refused(self) -> bool:
        return REFUSAL.lower() in self.answer.lower()


def format_context(items: list[Retrieved]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {r.source} | section: {r.document.metadata.get('section', '')})\n"
        f"{r.document.page_content}"
        for i, r in enumerate(items, start=1)
    )


class RagPipeline:
    def __init__(self, store, llm, k: int = 4, max_distance: float | None = None):
        """max_distance: optional guard - if even the best chunk is farther than this, refuse
        without calling the LLM. Tune it per embedding model (distances are not comparable)."""
        self.store, self.llm, self.k, self.max_distance = store, llm, k, max_distance
        self._chain = PROMPT | llm | StrOutputParser()

    def retrieve(self, question: str, k: int | None = None) -> list[Retrieved]:
        hits = self.store.similarity_search_with_score(question, k=k or self.k)
        items = [Retrieved(doc, float(dist)) for doc, dist in hits]
        if self.max_distance is not None and items and items[0].distance > self.max_distance:
            return []
        return items

    def answer(self, question: str) -> RagAnswer:
        items = self.retrieve(question)
        if not items:
            return RagAnswer(question, REFUSAL, [])
        text = self._chain.invoke({"context": format_context(items), "question": question})
        return RagAnswer(question, text.strip(), items)
