"""Embedding and chat-model factories.

Real providers (huggingface / openai / ollama) are imported lazily so the project installs and
tests without them. Two offline stand-ins exist purely so the whole pipeline can run in CI:

* HashingEmbeddings - a bag-of-words hashing embedder (lexical, not semantic).
* extractive LLM    - returns the best-matching sentence from the retrieved context.
"""
from __future__ import annotations

import re
import zlib
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from .config import Settings

_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been do does did how what when where which "
    "who whom why can could should would i my me we our you your it its this that these those with "
    "at by from as if per any much many".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords, with a crude plural strip."""
    toks = re.findall(r"[a-z0-9€@.:]+", text.lower())
    out = []
    for t in toks:
        t = t.strip(".:")
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        if t and t not in _STOPWORDS:
            out.append(t)
    return out


# --------------------------------------------------------------------------- embeddings
class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings. Lexical only - for tests and offline demos."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = zlib.crc32(tok.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dim] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_DEFAULT_EMBED_MODEL = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}
_DEFAULT_LLM_MODEL = {
    "huggingface": "Qwen/Qwen2.5-1.5B-Instruct",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}


def get_embeddings(settings: Settings) -> Embeddings:
    kind = settings.embeddings
    model = settings.embed_model or _DEFAULT_EMBED_MODEL.get(kind, "")
    if kind == "hash":
        return HashingEmbeddings()
    if kind == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    if kind == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown embeddings provider: {kind!r}")


# --------------------------------------------------------------------------- chat models
def _messages_of(value: Any) -> list[BaseMessage]:
    """Normalise a PromptValue / message list / string into a list of messages."""
    if hasattr(value, "to_messages"):
        return value.to_messages()
    if isinstance(value, str):
        from langchain_core.messages import HumanMessage

        return [HumanMessage(content=value)]
    return list(value)


_ROLE = {"system": "system", "human": "user", "ai": "assistant"}


def make_extractive_llm(min_overlap: int = 2) -> RunnableLambda:
    """Offline 'LLM' for tests: answer with the context sentence that best overlaps the question."""
    from .rag import REFUSAL  # local import avoids a cycle

    def run(value: Any) -> AIMessage:
        text = _messages_of(value)[-1].content
        if "Context:" not in text or "Question:" not in text:
            return AIMessage(content=REFUSAL)
        context = text.split("Context:", 1)[1].rsplit("Question:", 1)[0]
        question = text.rsplit("Question:", 1)[1]
        q_tokens = set(tokenize(question))
        best, best_score = None, 0
        for block in re.split(r"\n(?=\[\d+\] )", context.strip()):
            m = re.match(r"\[(\d+)\] \(source:[^\n]*\)\n(.*)", block, re.S)
            if not m:
                continue
            idx, body = m.group(1), m.group(2)
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                if sent.lstrip().startswith("#"):  # skip markdown headings
                    continue
                score = len(q_tokens & set(tokenize(sent)))
                if score > best_score:
                    best, best_score = (sent.strip(), idx), score
        if best is None or best_score < min_overlap:
            return AIMessage(content=REFUSAL)
        return AIMessage(content=f"{best[0]} [{best[1]}]")

    return RunnableLambda(run)


def make_hf_local_chat(model_name: str, max_new_tokens: int = 512) -> RunnableLambda:
    """Greedy local chat model on top of transformers (needs transformers>=4.56 and torch)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device).eval()

    def run(value: Any) -> AIMessage:
        msgs = [{"role": _ROLE.get(m.type, "user"), "content": m.content} for m in _messages_of(value)]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return AIMessage(content=tok.decode(new_tokens, skip_special_tokens=True))

    return RunnableLambda(run)


def get_llm(settings: Settings):
    kind = settings.llm
    model = settings.llm_model or _DEFAULT_LLM_MODEL.get(kind, "")
    if kind == "extractive":
        return make_extractive_llm()
    if kind == "huggingface":
        return make_hf_local_chat(model)
    if kind == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    if kind == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)
    raise ValueError(f"Unknown LLM provider: {kind!r}")
