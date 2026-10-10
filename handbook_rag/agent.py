"""A from-scratch ReAct (Reason + Act) agent loop over LangChain tools.

Why hand-rolled instead of a prebuilt agent? The loop is ~100 lines, works with any chat model
(including small local ones that cannot do native tool calling), and every step is inspectable,
which makes it easy to explain, test and debug.

Protocol (text based):
    Thought: ...            <- the model reasons
    Action: <tool name>     <- the model picks a tool
    Action Input: {...}     <- JSON arguments
    Observation: ...        <- WE run the tool and feed the result back
    ... repeat ...
    Final Answer: ...
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

SYSTEM_TEMPLATE = """You are a ReAct agent that answers questions about the Meridian Labs employee handbook using tools.

Available tools:
{tool_descriptions}

Use EXACTLY this format:

Thought: reason about what to do next
Action: one of [{tool_names}]
Action Input: a JSON object with the tool's arguments
Observation: (filled in by the system - NEVER write this yourself)
... (Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the answer to the question, mentioning handbook sources you used

Rules:
- Policy facts must come from search_handbook. Do not rely on memory.
- Do all arithmetic with the calculator tool.
- Take one action at a time and wait for the Observation.
- If the handbook does not contain the information, say you don't know."""

_ACTION_RE = re.compile(r"Action:\s*([\w\-]+)\s*\n\s*Action Input:\s*(.*)", re.S)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.S)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\Z)", re.S)
MAX_OBSERVATION_CHARS = 1500


@dataclass
class Step:
    thought: str
    action: str
    action_input: object
    observation: str


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = True

    def trace(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            lines += [f"[{i}] Thought: {s.thought}", f"    Action: {s.action} {json.dumps(s.action_input, ensure_ascii=False)}",
                      f"    Observation: {s.observation[:300]}{'...' if len(s.observation) > 300 else ''}"]
        lines.append(f"Final Answer: {self.answer}")
        return "\n".join(lines)


def _parse_action_input(raw: str, tool) -> object:
    """Accept clean JSON, JSON wrapped in code fences / trailing text, or a bare string."""
    raw = raw.strip().strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    first_arg = next(iter(tool.args), "input")  # bare string -> first parameter
    return {first_arg: raw.strip("\"' \n")}


class ReActAgent:
    def __init__(self, llm, tools: list, max_steps: int = 6):
        self.llm, self.tools, self.max_steps = llm, {t.name: t for t in tools}, max_steps
        descriptions = "\n".join(
            f"- {t.name}: {t.description} Args: {json.dumps({k: v.get('type', 'string') for k, v in t.args.items()})}"
            for t in tools
        )
        self.system = SYSTEM_TEMPLATE.format(tool_descriptions=descriptions, tool_names=", ".join(self.tools))

    def _run_tool(self, name: str, raw_input: str) -> tuple[object, str]:
        tool = self.tools.get(name)
        if tool is None:
            return raw_input, f"Error: unknown tool '{name}'. Valid tools: {', '.join(self.tools)}."
        args = _parse_action_input(raw_input, tool)
        try:
            return args, str(tool.invoke(args))[:MAX_OBSERVATION_CHARS]
        except Exception as exc:  # tool errors become observations so the model can recover
            return args, f"Error: {type(exc).__name__}: {str(exc)[:300]}"

    def run(self, question: str) -> AgentResult:
        messages = [SystemMessage(content=self.system), HumanMessage(content=f"Question: {question}")]
        steps: list[Step] = []
        for _ in range(self.max_steps):
            raw = self.llm.invoke(messages).content
            text = raw.split("Observation:")[0].strip()  # discard hallucinated observations

            action = _ACTION_RE.search(text)
            final = _FINAL_RE.search(text)
            # If the model emitted both, the action comes first: run it, ignore the premature answer.
            if final and not (action and action.start() < final.start()):
                return AgentResult(question, final.group(1).strip(), steps, finished=True)

            if action:
                name, raw_input = action.group(1), action.group(2).split("\nThought:")[0]
                m = _THOUGHT_RE.search(text)
                thought = m.group(1).strip() if m else ""
                args, observation = self._run_tool(name, raw_input)
                steps.append(Step(thought, name, args, observation))
                messages += [AIMessage(content=text[: action.end()].split("\nThought:")[0]),
                             HumanMessage(content=f"Observation: {observation}")]
            else:
                messages += [AIMessage(content=text or "(empty)"),
                             HumanMessage(content="Observation: Invalid format. Reply with either "
                                                  "'Action:' + 'Action Input:' or 'Final Answer:'.")]
        return AgentResult(question, f"Stopped after {self.max_steps} steps without a final answer.", steps, finished=False)
