from handbook_rag.evaluate import keywords_ok, load_questions, retrieval_eval
from handbook_rag.rag import REFUSAL
from handbook_rag.config import ROOT


def test_retrieval_finds_right_document(pipeline):
    top = pipeline.retrieve("How many vacation days do I get per year?")
    assert top[0].source == "01_leave_policy.md"


def test_answer_is_grounded_and_cited(pipeline):
    res = pipeline.answer("What is the daily meal allowance on international business trips?")
    assert "70" in res.answer and "[" in res.answer
    assert "02_expenses_and_travel.md" in res.sources


def test_unanswerable_question_is_refused(pipeline):
    assert pipeline.answer("What is the CEO's name?").answer == REFUSAL


def test_max_distance_guard_skips_llm(chunks):
    from handbook_rag.ingest import build_vectorstore
    from handbook_rag.providers import HashingEmbeddings, make_extractive_llm
    from handbook_rag.rag import RagPipeline

    p = RagPipeline(build_vectorstore(chunks, HashingEmbeddings()), make_extractive_llm(), max_distance=0.0001)
    res = p.answer("How many vacation days do I get per year?")
    assert res.refused and res.retrieved == []


def test_keyword_matcher_alternatives():
    assert keywords_ok("It is 1,500 euros", ["1,500|1500"])
    assert not keywords_ok("It is 99", ["1,500|1500"])
    assert keywords_ok("June and December", ["june", "December"])


def test_retrieval_metrics_sane_on_eval_set(pipeline):
    qs = load_questions(ROOT / "eval" / "questions.jsonl")
    m = retrieval_eval(pipeline.retrieve, qs)
    assert m["hit@4"] >= 0.8          # lexical baseline should already find most gold docs
    assert m["hit@1"] <= m["hit@2"] <= m["hit@4"]


def test_context_recall_metric_present_and_bounded(pipeline):
    qs = load_questions(ROOT / "eval" / "questions.jsonl")
    m = retrieval_eval(pipeline.retrieve, qs)
    assert all(0 <= v <= 1 for v in m.values())
    assert m["ctx@1"] <= m["ctx@2"] <= m["ctx@4"]
