"""Tool executors and OpenAI-compatible tool definitions."""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from typing import Any

_CODE_BLOCKED = frozenset({
    "import", "from", "open", "exec", "eval", "compile", "execfile",
    "__import__", "getattr", "setattr", "delattr", "globals", "locals",
    "input", "raw_input", "file", "reload", "exit", "quit", "os", "sys",
    "subprocess", "socket", "requests", "urllib", "builtins", "code",
    "frame", "traceback", "ctypes", "pickle", "marshal", "dis",
})

_SAFE_MATH = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "abs": abs, "round": round,
    "min": min, "max": max,
}

# ── Executors ────────────────────────────────────────────


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


def run_execute_code(code: str, timeout_s: float = 5.0) -> dict[str, Any]:
    code = (code or "").strip()
    if not code:
        return {"ok": False, "error": "Empty code", "stdout": "", "stderr": ""}
    code_lower = code.lower()
    for blocked in _CODE_BLOCKED:
        if blocked in code_lower:
            return {"ok": False, "error": f"Blocked: {blocked}", "stdout": "", "stderr": ""}
    if "__" in code:
        return {"ok": False, "error": "Blocked: __", "stdout": "", "stderr": ""}
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout_s,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "ok": result.returncode == 0,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timeout", "stdout": "", "stderr": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}


# ── Tool dispatch ────────────────────────────────────────


def execute_tool(name: str, args: dict[str, Any]) -> str:
    if name == "calculator":
        out = run_calculator(args.get("expression", ""))
        return str(out.get("result")) if out.get("ok") else f"Error: {out.get('error', 'unknown')}"
    if name == "execute_code":
        out = run_execute_code(args.get("code", ""))
        return out.get("stdout", "") or (f"stderr: {out.get('stderr', '')}" if not out.get("ok") else "ok")
    return "Unknown tool"


# ── OpenAI-compatible tool definitions ───────────────────

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
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in a sandbox. No network or file I/O.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
]
