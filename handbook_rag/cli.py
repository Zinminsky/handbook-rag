"""Command line interface:  python -m handbook_rag <command> ..."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .agent import ReActAgent
from .config import ROOT, Settings
from .evaluate import (evaluate_agent, evaluate_rag, keywords_ok, load_questions, retrieval_eval, save_csv,
                       sweep, to_markdown)
from .ingest import build_vectorstore, load_documents, load_vectorstore, split_documents
from .providers import get_embeddings, get_llm
from .rag import RagPipeline
from .tools import default_tools

EVAL_DIR = ROOT / "eval"


def _banner(s: Settings) -> None:
    if s.offline_demo:
        print("[offline demo mode: hash embeddings and/or extractive LLM - set RAG_EMBEDDINGS / RAG_LLM "
              "for real models; see README]\n")


def _pipeline(s: Settings) -> RagPipeline:
    store = load_vectorstore(get_embeddings(s), s.collection, s.persist_dir)
    return RagPipeline(store, get_llm(s), k=s.top_k)


def cmd_ingest(s: Settings, a) -> None:
    docs = load_documents(s.data_dir)
    chunks = split_documents(docs, s.chunk_size, s.chunk_overlap)
    build_vectorstore(chunks, get_embeddings(s), s.collection, s.persist_dir)
    print(f"Ingested {len(docs)} documents -> {len(chunks)} chunks "
          f"(size={s.chunk_size}, overlap={s.chunk_overlap}) into {s.persist_dir}")


def cmd_ask(s: Settings, a) -> None:
    res = _pipeline(s).answer(a.question)
    print(f"\n{res.answer}\n\nSources: {', '.join(res.sources) or '-'}")
    if a.show_context:
        for i, r in enumerate(res.retrieved, 1):
            print(f"\n--- [{i}] {r.source} | {r.document.metadata.get('section')} | distance={r.distance:.3f}")
            print(r.document.page_content)


def _require_real_llm(s: Settings) -> None:
    if s.llm == "extractive":
        raise SystemExit("The ReAct agent needs a real chat model. Set RAG_LLM=huggingface | openai | ollama "
                         "(see README). The offline 'extractive' stand-in cannot follow the ReAct format.")


def cmd_agent(s: Settings, a) -> None:
    _require_real_llm(s)
    pipeline = _pipeline(s)
    result = ReActAgent(get_llm(s), default_tools(pipeline)).run(a.question)
    print(result.trace() if a.trace else f"\n{result.answer}")


def cmd_eval(s: Settings, a) -> None:
    pipeline = _pipeline(s)
    if a.agent:
        _require_real_llm(s)
        qs = load_questions(EVAL_DIR / "agent_questions.jsonl")
        summary, rows = evaluate_agent(ReActAgent(get_llm(s), default_tools(pipeline)), qs)
    else:
        qs = load_questions(EVAL_DIR / "questions.jsonl")
        retr = retrieval_eval(pipeline.retrieve, qs)
        if a.retrieval_only:
            print(json.dumps(retr, indent=2))
            return
        summary, rows = evaluate_rag(pipeline, qs)
        summary = {**retr, **summary}
    print(json.dumps({k: round(v, 3) for k, v in summary.items()}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Per-question results written to {a.out}")
    for r in rows:
        if not r["correct"]:
            print(f"  FAIL {r['id']} [{r['type']}] {r['question']}\n       -> {r['answer'][:160]}")


def cmd_sweep(s: Settings, a) -> None:
    docs = load_documents(s.data_dir)
    qs = load_questions(EVAL_DIR / "questions.jsonl")
    rows = sweep(docs, get_embeddings(s), qs, a.sizes, a.overlaps)
    print(to_markdown(rows))
    if a.out:
        save_csv(rows, Path(a.out))
        print(f"\nSaved {a.out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="handbook_rag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="chunk, embed and index the handbook")
    sp.add_argument("--chunk-size", type=int)
    sp.add_argument("--chunk-overlap", type=int)
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("ask", help="RAG question answering")
    sp.add_argument("question")
    sp.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("agent", help="ReAct agent with tools")
    sp.add_argument("question")
    sp.add_argument("--trace", action="store_true", help="print every Thought/Action/Observation")
    sp.set_defaults(fn=cmd_agent)

    sp = sub.add_parser("eval", help="run the evaluation set")
    sp.add_argument("--retrieval-only", action="store_true", help="skip the LLM; only retrieval metrics")
    sp.add_argument("--agent", action="store_true", help="evaluate the agent on multi-step questions")
    sp.add_argument("--out", help="write per-question results as JSON")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("sweep", help="retrieval metrics across chunk sizes / overlaps")
    sp.add_argument("--sizes", type=int, nargs="+", default=[150, 250, 350, 500, 800])
    sp.add_argument("--overlaps", type=int, nargs="+", default=[0, 50, 100])
    sp.add_argument("--out", help="save results as CSV")
    sp.set_defaults(fn=cmd_sweep)

    args = p.parse_args(argv)
    s = Settings.from_env()
    if args.cmd == "ingest":
        s = replace(s, chunk_size=args.chunk_size or s.chunk_size, chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else s.chunk_overlap)
    _banner(s)
    args.fn(s, args)


if __name__ == "__main__":
    main()
