"""
Tool implementations for agent-search: calculator, code execution (sandbox).

LLM invokes these when reasoning requires computation.
"""
from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from typing import Any


# Blocked identifiers for code execution
_CODE_BLOCKED = frozenset(
    {
        "import", "from", "open", "exec", "eval", "compile", "execfile",
        "__import__", "getattr", "setattr", "delattr", "globals", "locals",
        "input", "raw_input", "file", "reload", "exit", "quit", "os", "sys",
        "subprocess", "socket", "requests", "urllib", "builtins", "code",
        "frame", "traceback", "ctypes", "pickle", "marshal", "dis",
    }
)


def run_calculator(expression: str) -> dict[str, Any]:
    """
    Safely evaluate a math expression. Supports: +, -, *, /, **, sqrt, log, sin, cos, etc.
    """
    expr = (expression or "").strip()
    if not expr:
        return {"ok": False, "error": "Empty expression", "result": None}

    # Allow only safe chars: numbers, spaces, +-*/()**. and math function names
    if not re.match(r"^[\d\s+\-*/().a-z_]+$", expr):
        return {"ok": False, "error": "Invalid characters in expression", "result": None}

    # Allowed names for eval
    safe_dict = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
    }
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in safe_dict:
                    return {"ok": False, "error": f"Unknown symbol: {node.id}", "result": None}
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return {"ok": True, "result": result, "error": None}
    except SyntaxError as e:
        return {"ok": False, "error": str(e), "result": None}
    except ZeroDivisionError:
        return {"ok": False, "error": "Division by zero", "result": None}
    except Exception as e:
        return {"ok": False, "error": str(e), "result": None}


def run_execute_code(code: str, timeout_s: float = 5.0) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed subprocess. No network, no file I/O.
    """
    code = (code or "").strip()
    if not code:
        return {"ok": False, "error": "Empty code", "stdout": "", "stderr": ""}

    code_lower = code.lower()
    for blocked in _CODE_BLOCKED:
        if blocked in code_lower:
            return {"ok": False, "error": f"Blocked: {blocked}", "stdout": "", "stderr": ""}

    # Block double underscore
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
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        return {
            "ok": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timeout", "stdout": "", "stderr": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}
