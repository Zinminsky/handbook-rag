"""Web interface for the handbook RAG project.

Run from the project folder:   streamlit run app.py     (opens http://localhost:8501)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handbook_rag.agent import ReActAgent  # noqa: E402
from handbook_rag.config import ROOT, Settings  # noqa: E402
from handbook_rag.evaluate import evaluate_rag, load_questions, retrieval_eval, sweep  # noqa: E402
from handbook_rag.ingest import build_vectorstore, load_documents, split_documents  # noqa: E402
from handbook_rag.providers import get_embeddings, get_llm  # noqa: E402
from handbook_rag.rag import RagPipeline  # noqa: E402
from handbook_rag.tools import default_tools  # noqa: E402

st.set_page_config(page_title="Handbook RAG", page_icon="📚", layout="wide")

EMBEDDINGS = ["hash", "huggingface", "openai", "ollama"]
LLMS = ["extractive", "huggingface", "openai", "ollama"]
RAG_EXAMPLES = [
    "How many vacation days do full-time employees get per year?",
    "What is the hotel price cap per night in London?",
    "My laptop got stolen on the train. Who do I tell and how fast?",
    "Can I expense a glass of wine at a client dinner?",
    "What is the CEO's name?",
]
AGENT_EXAMPLES = [
    "What is the total of the referral bonus and the yearly learning budget?",
    "What is the maximum I can be reimbursed for 3 nights in a London hotel plus 4 days of international meal allowance?",
    "I have worked full-time for 4 years. I take vacation from 2026-07-06 to 2026-07-17. "
    "How many working days is that, and how many vacation days remain from my annual entitlement?",
]

base = Settings.from_env()


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading embedding model (the first run downloads it)...")
def load_embeddings(kind: str, model: str):
    return get_embeddings(replace(base, embeddings=kind, embed_model=model))


@st.cache_resource(show_spinner="Loading language model (the first run downloads it)...")
def load_llm(kind: str, model: str):
    return get_llm(replace(base, llm=kind, llm_model=model))


@st.cache_resource(show_spinner="Building the index...")
def load_store(kind: str, model: str, chunk_size: int, overlap: int):
    docs = load_documents(base.data_dir)
    chunks = split_documents(docs, chunk_size, overlap)
    return build_vectorstore(chunks, load_embeddings(kind, model))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 Handbook RAG")
    st.caption("Retrieval-augmented QA and a ReAct agent over a fictional company handbook.")
    emb_kind = st.selectbox("Embeddings", EMBEDDINGS, index=EMBEDDINGS.index(base.embeddings))
    llm_kind = st.selectbox("LLM", LLMS, index=LLMS.index(base.llm))
    emb_model = st.text_input("Embedding model (blank = default)", base.embed_model)
    llm_model = st.text_input("LLM model (blank = default)", base.llm_model)
    chunk_size = st.slider("Chunk size", 100, 1000, base.chunk_size, 50)
    overlap = st.slider("Chunk overlap", 0, 200, base.chunk_overlap, 10)
    top_k = st.slider("Top-k passages", 1, 8, base.top_k)

if overlap >= chunk_size:
    st.error("Chunk overlap must be smaller than the chunk size.")
    st.stop()

if emb_kind == "hash" or llm_kind == "extractive":
    st.info(
        "**Offline demo mode** (hash embeddings and/or the extractive stand-in). It shows the plumbing, not real "
        "quality. Pick `huggingface`, `ollama` or `openai` in the sidebar for real answers."
    )

try:
    embeddings = load_embeddings(emb_kind, emb_model)
    store = load_store(emb_kind, emb_model, chunk_size, overlap)
    llm = load_llm(llm_kind, llm_model)
except Exception as exc:  # missing package, no network, bad model name, missing API key ...
    st.error(f"Could not load the selected providers: {type(exc).__name__}: {exc}")
    st.stop()

pipeline = RagPipeline(store, llm, k=top_k)
st.sidebar.success(f"Index ready: {store._collection.count()} chunks")

ask_tab, agent_tab, eval_tab = st.tabs(["Ask (RAG)", "Agent", "Evaluation"])

# ----------------------------------------------------------------------------- tab 1: RAG
with ask_tab:
    st.subheader("Ask the handbook")
    typed = st.text_input("Your question", key="ask_q", placeholder="e.g. How many vacation days do I get?")
    example = st.selectbox("...or pick an example", [""] + RAG_EXAMPLES, key="ask_example")
    question = typed.strip() or example
    if st.button("Ask", type="primary", key="ask_btn") and question:
        with st.spinner("Retrieving and answering..."):
            res = pipeline.answer(question)
        if res.refused:
            st.warning(res.answer)
        else:
            st.markdown(res.answer)
        st.caption("Sources: " + (", ".join(res.sources) or "none"))
        with st.expander(f"Retrieved passages ({len(res.retrieved)})"):
            for i, r in enumerate(res.retrieved, 1):
                st.markdown(f"**[{i}] {r.source}** · {r.document.metadata.get('section', '')} · distance `{r.distance:.3f}`")
                st.text(r.document.page_content)

# ----------------------------------------------------------------------------- tab 2: agent
with agent_tab:
    st.subheader("ReAct agent")
    st.caption("For multi-step questions: it searches the handbook, counts working days and does arithmetic with tools.")
    if llm_kind == "extractive":
        st.warning("The agent needs a real chat model. Choose `huggingface`, `ollama` or `openai` as the LLM in the sidebar.")
    else:
        typed_a = st.text_input("Your question", key="agent_q")
        example_a = st.selectbox("...or pick an example", [""] + AGENT_EXAMPLES, key="agent_example")
        question_a = typed_a.strip() or example_a
        if st.button("Run agent", type="primary", key="agent_btn") and question_a:
            with st.spinner("The agent is thinking..."):
                result = ReActAgent(llm, default_tools(pipeline)).run(question_a)
            (st.markdown if result.finished else st.warning)(result.answer)
            st.caption(f"{len(result.steps)} tool call(s)")
            for i, s in enumerate(result.steps, 1):
                with st.expander(f"Step {i}: {s.action}"):
                    st.markdown(f"**Thought:** {s.thought or '-'}")
                    st.code(json.dumps(s.action_input, ensure_ascii=False, indent=2), language="json")
                    st.markdown("**Observation:**")
                    st.text(s.observation)

# ----------------------------------------------------------------------------- tab 3: evaluation
with eval_tab:
    st.subheader("Evaluation")
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    st.caption(f"{len(questions)} labelled questions in eval/questions.jsonl. Results use the settings in the sidebar.")

    c1, c2 = st.columns(2)
    if c1.button("Retrieval metrics (fast, no LLM)", key="eval_retr"):
        metrics = retrieval_eval(pipeline.retrieve, questions)
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.2f}")

    if c2.button("Full evaluation (calls the LLM)", key="eval_full"):
        bar = st.progress(0.0, text="Answering questions...")
        done = []

        class _Progress(RagPipeline):  # report progress while evaluate_rag loops over the questions
            def answer(self, q):
                out = super().answer(q)
                done.append(1)
                bar.progress(len(done) / len(questions), text=f"{len(done)}/{len(questions)}")
                return out

        summary, rows = evaluate_rag(_Progress(store, llm, k=top_k), questions)
        bar.empty()
        cols = st.columns(len(summary))
        for col, (name, value) in zip(cols, summary.items()):
            col.metric(name, f"{value:.2f}")
        frame = pd.DataFrame(rows)
        frame["sources"] = frame["sources"].map(", ".join)
        st.dataframe(frame[["id", "type", "question", "answer", "correct", "sources"]], use_container_width=True)

    st.markdown("---")
    st.markdown("**Chunking sweep** (retrieval only): how do chunk size and overlap change retrieval?")
    sizes = st.multiselect("Chunk sizes", [100, 150, 250, 350, 500, 800], default=[150, 250, 350, 500], key="sw_sizes")
    overlaps = st.multiselect("Overlaps", [0, 50, 100], default=[0, 50], key="sw_overlaps")
    if st.button("Run sweep", key="sweep_btn") and sizes and overlaps:
        with st.spinner("Re-indexing for every combination..."):
            rows = sweep(load_documents(base.data_dir), embeddings, questions, sizes, overlaps)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("ctx@k = the retrieved text actually contains the answer. Compare small and large chunks.")
