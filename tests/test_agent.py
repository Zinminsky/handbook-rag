"""The agent is tested with a scripted LLM, so the loop logic is verified deterministically."""
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from handbook_rag.agent import ReActAgent
from handbook_rag.tools import default_tools


def scripted(*replies):
    it = iter(replies)
    seen = []

    def run(messages):
        seen.append(messages)
        return AIMessage(content=next(it))

    llm = RunnableLambda(run)
    llm.seen = seen
    return llm


def test_multi_step_run(pipeline):
    llm = scripted(
        'Thought: I need the entitlement.\nAction: search_handbook\nAction Input: {"query": "vacation days after 3 years"}',
        'Thought: Count the days.\nAction: business_days_between\nAction Input: {"start_date": "2026-07-06", "end_date": "2026-07-17"}',
        'Thought: Subtract.\nAction: calculator\nAction Input: {"expression": "27 - 10"}',
        "Thought: I now know the final answer\nFinal Answer: 10 working days, 17 days remain.",
    )
    res = ReActAgent(llm, default_tools(pipeline)).run("vacation question")
    assert res.finished and "17" in res.answer
    assert [s.action for s in res.steps] == ["search_handbook", "business_days_between", "calculator"]
    assert res.steps[1].observation.startswith("10 working days")
    assert res.steps[2].observation == "17"
    # observations are fed back to the model as the next human turn
    assert llm.seen[-1][-1].content == "Observation: 17"


def test_hallucinated_observation_is_discarded(pipeline):
    llm = scripted(
        'Thought: calc\nAction: calculator\nAction Input: {"expression": "2+2"}\nObservation: 5\nFinal Answer: 5',
        "Final Answer: 4",
    )
    res = ReActAgent(llm, default_tools(pipeline)).run("2+2?")
    assert res.steps[0].observation == "4" and res.answer == "4"


def test_unknown_tool_and_bad_args_become_observations(pipeline):
    llm = scripted(
        "Action: teleport\nAction Input: {}",
        'Action: calculator\nAction Input: {"expression": "import os"}',
        "Final Answer: gave up",
    )
    res = ReActAgent(llm, default_tools(pipeline)).run("x")
    assert "unknown tool" in res.steps[0].observation
    assert res.steps[1].observation.startswith("Error:")
    assert res.answer == "gave up"


def test_bare_string_action_input(pipeline):
    llm = scripted("Action: calculator\nAction Input: 6 * 250", "Final Answer: 1500")
    res = ReActAgent(llm, default_tools(pipeline)).run("x")
    assert res.steps[0].observation == "1500"


def test_invalid_format_is_nudged_then_step_limit(pipeline):
    llm = scripted("blah", "blah", "blah")
    res = ReActAgent(llm, default_tools(pipeline), max_steps=3).run("x")
    assert not res.finished and "Stopped after 3 steps" in res.answer
