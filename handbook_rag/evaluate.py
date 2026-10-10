"""Evaluation: retrieval metrics, answer accuracy, refusal behaviour, and a chunking sweep."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from langchain_core.embeddings import Embeddings

from .ingest import build_vectorstore, split_documents
from .rag import REFUSAL, RagPipeline


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def keywords_ok(answer: str, keywords: list[str]) -> bool:
    """Every keyword must appear (case-insensitive). 'a|b' means either alternative is fine."""
    low = answer.lower()
    return all(any(alt.strip().lower() in low for alt in kw.split("|")) for kw in keywords)


def _rank_of_gold(sources: list[str], gold: list[str]) -> int | None:
    for i, s in enumerate(sources, start=1):
        if s in gold:
            return i
    return None


def retrieval_eval(retrieve, questions: list[dict], ks=(1, 2, 4)) -> dict:
    """retrieve(question, k) -> list[Retrieved]. Only answerable questions count.

    hit@k  : a chunk from a gold source FILE is in the top k (coarse: documents are small).
    ctx@k  : the text of the top-k chunks contains every gold keyword, i.e. the answer is actually
             available to the LLM. This is the metric chunking changes most.
    mrr@kmax: mean reciprocal rank of the first chunk from a gold source file."""
    answerable = [q for q in questions if q["gold_sources"]]
    kmax = max(ks)
    ranks, ctx = [], {k: 0 for k in ks}
    for q in answerable:
        items = retrieve(q["question"], kmax)
        ranks.append(_rank_of_gold([r.source for r in items], q["gold_sources"]))
        for k in ks:
            text = " ".join(r.document.page_content for r in items[:k])
            ctx[k] += keywords_ok(text, q["keywords"])
    n = len(answerable)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out.update({f"ctx@{k}": ctx[k] / n for k in ks})
    out[f"mrr@{kmax}"] = sum(1 / r for r in ranks if r) / n
    return out


def evaluate_rag(pipeline: RagPipeline, questions: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        res = pipeline.answer(q["question"])
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "answer": res.answer,
               "sources": res.sources, "refused": res.refused}
        row["correct"] = res.refused if not q["gold_sources"] else (not res.refused and keywords_ok(res.answer, q["keywords"]))
        rows.append(row)
    return _summarise(rows), rows


def evaluate_agent(agent, questions: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        res = agent.run(q["question"])
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "answer": res.answer,
                     "steps": len(res.steps), "tools": [s.action for s in res.steps], "finished": res.finished,
                     "correct": res.finished and keywords_ok(res.answer, q["keywords"])})
    return {"accuracy": sum(r["correct"] for r in rows) / len(rows),
            "avg_steps": sum(r["steps"] for r in rows) / len(rows)}, rows


def _summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "answer_accuracy": sum(r["correct"] for r in ans) / len(ans),
        "false_refusal_rate": sum(r["refused"] for r in ans) / len(ans),
    }
    if una:
        summary["refusal_accuracy"] = sum(r["correct"] for r in una) / len(una)
    for t in sorted({r["type"] for r in ans}):
        sub = [r for r in ans if r["type"] == t]
        summary[f"acc[{t}]"] = sum(r["correct"] for r in sub) / len(sub)
    return summary


def sweep(docs, embeddings: Embeddings, questions: list[dict], chunk_sizes, overlaps, ks=(1, 4)) -> list[dict]:
    """Retrieval-only grid search over chunking parameters (no LLM needed, so it is fast and cheap)."""
    rows = []
    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
            chunks = split_documents(docs, size, overlap)
            store = build_vectorstore(chunks, embeddings)

            def retrieve(question, k, _store=store):
                from .rag import Retrieved
                return [Retrieved(d, s) for d, s in _store.similarity_search_with_score(question, k=k)]

            metrics = retrieval_eval(retrieve, questions, ks)
            rows.append({"chunk_size": size, "overlap": overlap, "n_chunks": len(chunks), **metrics})
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
