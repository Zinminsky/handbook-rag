"""Headless test of the Streamlit app in offline mode (skipped when streamlit is not installed)."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from handbook_rag.config import ROOT  # noqa: E402


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_app_loads_without_errors():
    at = _app()
    assert not at.exception
    assert any("Offline demo mode" in i.value for i in at.info)


def test_ask_tab_answers_with_sources():
    at = _app()
    at.text_input(key="ask_q").set_value("What is the daily meal allowance on international business trips?")
    at.button(key="ask_btn").click().run()
    assert not at.exception
    assert any("70" in m.value for m in at.markdown)
    assert any("02_expenses_and_travel.md" in c.value for c in at.caption)


def test_unanswerable_question_shows_refusal_warning():
    at = _app()
    at.text_input(key="ask_q").set_value("What is the CEO's name?")
    at.button(key="ask_btn").click().run()
    assert any("don't know" in w.value for w in at.warning)


def test_agent_tab_requires_real_llm_in_offline_mode():
    at = _app()
    assert any("needs a real chat model" in w.value for w in at.warning)


def test_retrieval_metrics_button():
    at = _app()
    at.button(key="eval_retr").click().run()
    assert not at.exception
    assert any(m.label == "hit@4" for m in at.metric)
