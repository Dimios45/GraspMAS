"""Sandboxed execution of Coder-generated code.

Replaces the legacy ``exec(code, globals())`` (agents/graspmas.py) with:
  1. cleanup of markdown fences
  2. AST parse + automatic repair of the known Qwen2-VL failure modes
     (module-scope ``return``, unwrapped script bodies, stray trailing lines)
  3. static whitelist check: only registry tools + benign builtins may be
     called; hallucinated ImagePatch methods are caught *before* any GPU
     work instead of exploding mid-round
  4. execution in a fresh, restricted namespace with an optional timeout

With all feature flags off the sandbox reproduces legacy behavior exactly:
no repair, no whitelist, syntax errors propagate (aborting the query), and
runtime errors inside execute_command are returned as error strings.
"""

import ast
import math
import re
import textwrap
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

# attribute calls that are always fine (builtins / list / str / dict / numpy)
_BENIGN_ATTRS = {
    "sort", "append", "extend", "copy", "get", "keys", "values", "items",
    "index", "count", "split", "strip", "lower", "upper", "format", "join",
    "median", "tolist", "cpu", "numpy", "squeeze", "reverse", "pop", "add",
    "startswith", "endswith", "replace", "round",
}
_FORBIDDEN_CALLS = {"exec", "eval", "open", "__import__", "compile", "input",
                    "globals", "locals", "vars", "setattr", "delattr"}
_ALLOWED_IMPORTS = {"math", "numpy"}


@dataclass
class SandboxResult:
    ok: bool
    out: Any = None
    error: Optional[str] = None
    error_kind: str = ""            # syntax | whitelist | runtime | timeout
    code_used: str = ""
    repaired: bool = False
    repair_note: str = ""
    violations: List[str] = field(default_factory=list)


def _clean(code: str) -> str:
    code = re.sub(r"^\s*```(?:python)?\s*$", "", code, flags=re.MULTILINE)
    return code.strip("\n")


def _repair(code: str, err: SyntaxError) -> tuple[str, str]:
    """Attempt mechanical fixes for known Qwen2-VL syntax failures."""
    msg = str(err)
    if "'return' outside function" in msg:
        if "def execute_command" in code:
            # drop stray top-level return/expression lines after the def
            fixed = "\n".join(
                l for l in code.split("\n")
                if not re.match(r"^return\b", l)
            )
            return fixed, "dropped module-scope return lines"
        # bare script body: wrap it
        body = textwrap.indent(code, "    ")
        return f"def execute_command(image):\n{body}", "wrapped bare script in execute_command"
    if "unexpected indent" in msg or "unindent" in msg:
        return textwrap.dedent(code), "dedented block"
    return code, ""


def _whitelist_violations(code_ast: ast.AST, allowed_tools: set[str]) -> List[str]:
    violations = []
    for node in ast.walk(code_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in node.names] \
                if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            for m in mods:
                if m not in _ALLOWED_IMPORTS:
                    violations.append(f"forbidden import: {m}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                violations.append(f"forbidden call: {f.id}()")
            elif isinstance(f, ast.Attribute):
                name = f.attr
                if name not in allowed_tools and name not in _BENIGN_ATTRS:
                    violations.append(f"unknown method: .{name}() — not in the ImagePatch API")
    return violations


def _make_namespace(image_patch_cls: Any = None) -> dict:
    """Fresh execution namespace per round (legacy leaked into module globals)."""
    if image_patch_cls is None:
        from image_patch import ImagePatch  # heavy import deferred to first use
        image_patch_cls = ImagePatch
    return {"ImagePatch": image_patch_cls, "math": math, "np": np}


def run_code(code: str,
             image: Any,
             allowed_tools: Optional[set] = None,
             repair: bool = False,
             whitelist: bool = False,
             timeout_s: float = 0.0,
             image_patch_cls: Any = None) -> SandboxResult:
    code = _clean(code)
    repaired, note = False, ""

    # 1) parse + compile (+ repair). Note: ast.parse alone does NOT catch
    # "'return' outside function" — that fires at compile() — so both steps
    # live inside the repair loop.
    try:
        tree = ast.parse(code)
        compiled = compile(tree, "<coder>", "exec")
    except SyntaxError as e:
        if not repair:
            return SandboxResult(ok=False, error=f"SyntaxError: {e}",
                                 error_kind="syntax", code_used=code)
        code2, note = _repair(code, e)
        try:
            tree = ast.parse(code2)
            compiled = compile(tree, "<coder>", "exec")
            code, repaired = code2, bool(note)
        except SyntaxError as e2:
            return SandboxResult(ok=False, error=f"SyntaxError after repair: {e2}",
                                 error_kind="syntax", code_used=code2,
                                 repaired=bool(note), repair_note=note)

    # 2) static whitelist
    if whitelist and allowed_tools:
        violations = _whitelist_violations(tree, set(allowed_tools))
        if violations:
            return SandboxResult(ok=False, error="; ".join(violations),
                                 error_kind="whitelist", code_used=code,
                                 repaired=repaired, repair_note=note,
                                 violations=violations)

    # 3) execute the definition + call execute_command
    ns = _make_namespace(image_patch_cls)
    try:
        exec(compiled, ns)
        fn = ns.get("execute_command")
        if fn is None:
            return SandboxResult(ok=False, error="code defines no execute_command(image)",
                                 error_kind="runtime", code_used=code,
                                 repaired=repaired, repair_note=note)

        if timeout_s and timeout_s > 0:
            box: dict = {}

            def _call():
                try:
                    box["out"] = fn(image)
                except Exception as e:      # noqa: BLE001
                    box["err"] = e

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout_s)
            if t.is_alive():
                return SandboxResult(ok=False, error=f"execution exceeded {timeout_s}s",
                                     error_kind="timeout", code_used=code,
                                     repaired=repaired, repair_note=note)
            if "err" in box:
                raise box["err"]
            out = box.get("out")
        else:
            out = fn(image)
        return SandboxResult(ok=True, out=out, code_used=code,
                             repaired=repaired, repair_note=note)
    except Exception as e:                  # noqa: BLE001
        return SandboxResult(ok=False, error=str(e), error_kind="runtime",
                             code_used=code, repaired=repaired, repair_note=note)
