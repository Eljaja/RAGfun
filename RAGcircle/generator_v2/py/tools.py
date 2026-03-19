"""Tool executors and OpenAI-compatible tool definitions.

Only the AST-based calculator is kept — it's actually safe.
execute_code was removed: the substring blocklist was trivially bypassable
(blocks "os" but chr(111)+chr(115) escapes it). A RAG service should not
run arbitrary code; if needed, use a proper sandbox.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

_SAFE_MATH = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "abs": abs, "round": round,
    "min": min, "max": max,
}


def run_calculator(expression: str) -> dict[str, Any]:
    expr = (expression or "").strip()
    if not expr:
        return {"ok": False, "error": "Empty expression", "result": None}
    if not re.match(r"^[\d\s+\-*/().a-z_]+$", expr):
        return {"ok": False, "error": "Invalid characters in expression", "result": None}
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in _SAFE_MATH:
                return {"ok": False, "error": f"Unknown symbol: {node.id}", "result": None}
        result = eval(expr, {"__builtins__": {}}, _SAFE_MATH)  # noqa: S307
        return {"ok": True, "result": result, "error": None}
    except ZeroDivisionError:
        return {"ok": False, "error": "Division by zero", "result": None}
    except Exception as e:
        return {"ok": False, "error": str(e), "result": None}


def execute_tool(name: str, args: dict[str, Any]) -> str:
    if name == "calculator":
        out = run_calculator(args.get("expression", ""))
        return str(out.get("result")) if out.get("ok") else f"Error: {out.get('error', 'unknown')}"
    return "Unknown tool"


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, log, sin, cos, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
]
