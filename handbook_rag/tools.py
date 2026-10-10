"""Custom tools for the ReAct agent."""
from __future__ import annotations

import ast
import operator
from datetime import date, timedelta

from langchain_core.tools import tool

from .rag import RagPipeline

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate +,-,*,/,//,%,** on numbers only (no names, calls or attributes)."""
    cleaned = expression.replace(",", "").replace("€", "").strip()

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    return ev(ast.parse(cleaned, mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '3 * 250 + 4 * 70'. Supports + - * / ** % and parentheses."""
    result = safe_eval(expression)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(round(result, 4) if isinstance(result, float) else result)


@tool
def business_days_between(start_date: str, end_date: str) -> str:
    """Count working days (Monday-Friday, both ends inclusive) between two ISO dates (YYYY-MM-DD).
    Does NOT subtract public holidays."""
    start, end = date.fromisoformat(start_date.strip()), date.fromisoformat(end_date.strip())
    if end < start:
        raise ValueError("end_date is before start_date")
    days = sum(1 for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5)
    return f"{days} working days (Mon-Fri) from {start} to {end} inclusive"


def make_search_tool(pipeline: RagPipeline):
    @tool
    def search_handbook(query: str) -> str:
        """Search the Meridian Labs employee handbook (leave, expenses, security, on-call, remote work,
        onboarding). Input is a natural-language query. Returns the most relevant passages with sources."""
        items = pipeline.retrieve(query)
        if not items:
            return "No relevant passages found."
        return "\n\n".join(f"({r.source} | {r.document.metadata.get('section', '')}) {r.document.page_content}" for r in items)

    return search_handbook


def default_tools(pipeline: RagPipeline) -> list:
    return [make_search_tool(pipeline), calculator, business_days_between]
