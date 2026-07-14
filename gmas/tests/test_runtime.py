"""Micro-tests for the gmas runtime (no GPU: execution namespace is stubbed).

Run: python -m pytest gmas/tests/test_runtime.py -q
"""

import numpy as np
import pytest

import gmas.runtime.sandbox as sandbox
from gmas.config import Flags
from gmas.runtime.registry import default_registry
from gmas.runtime.session import Session
from gmas.runtime.validator import validate_grasp


@pytest.fixture(autouse=True)
def stub_namespace(monkeypatch):
    """Avoid importing image_patch (loads GPU models) inside run_code."""
    monkeypatch.setattr(sandbox, "_make_namespace",
                        lambda image_patch_cls=None: {"ImagePatch": None, "np": np})


IMG = np.zeros((480, 640, 3), dtype=np.uint8)


# ── sandbox: repair of known Qwen2-VL failures ──────────────────────────────

def test_module_scope_return_repaired():
    code = ("def execute_command(image):\n    return [0.9, 1, 2, 3, 4, 5]\n"
            "return execute_command(image)")
    res = sandbox.run_code(code, IMG, repair=True)
    assert res.ok and res.repaired
    assert res.out == [0.9, 1, 2, 3, 4, 5]


def test_bare_script_wrapped():
    code = "x = [0.5, 10, 20, 5, 5, 0]\nreturn x"
    res = sandbox.run_code(code, IMG, repair=True)
    assert res.ok and res.repaired
    assert res.out == [0.5, 10, 20, 5, 5, 0]


def test_syntax_error_propagates_when_repair_off():
    res = sandbox.run_code("def f(:\n  pass", IMG, repair=False)
    assert not res.ok and res.error_kind == "syntax"


def test_markdown_fences_stripped():
    code = "```python\ndef execute_command(image):\n    return 1\n```"
    res = sandbox.run_code(code, IMG)
    assert res.ok and res.out == 1


def test_missing_execute_command_reported():
    res = sandbox.run_code("x = 1", IMG)
    assert not res.ok and "execute_command" in res.error


# ── sandbox: whitelist ───────────────────────────────────────────────────────

def test_hallucinated_method_blocked():
    code = ("def execute_command(image):\n"
            "    p = ImagePatch(image)\n"
            "    return p.detect_grasp_pose('apple')")   # not a real method
    res = sandbox.run_code(code, IMG, allowed_tools=set(default_registry().names()),
                           whitelist=True)
    assert not res.ok and res.error_kind == "whitelist"
    assert "detect_grasp_pose" in res.error


def test_real_api_passes_whitelist():
    code = ("def execute_command(image):\n"
            "    p = ImagePatch(image)\n"
            "    ps = p.find('apple')\n"
            "    return p.grasp_detection(ps[0])")
    tree_only = sandbox.run_code(code, IMG,
                                 allowed_tools=set(default_registry().names()),
                                 whitelist=True)
    # fails at runtime (ImagePatch stubbed to None) but NOT at the whitelist
    assert tree_only.error_kind != "whitelist"


def test_forbidden_import_blocked():
    code = "import os\ndef execute_command(image):\n    return os.getcwd()"
    res = sandbox.run_code(code, IMG, allowed_tools={"find"}, whitelist=True)
    assert not res.ok and "forbidden import" in res.error


# ── validator ────────────────────────────────────────────────────────────────

def test_validator_rejects_out_of_image():
    assert validate_grasp([0.9, 900, 200, 30, 20, 0], (480, 640))
    assert validate_grasp([0.9, -5, 200, 30, 20, 0], (480, 640))


def test_validator_rejects_degenerate():
    assert validate_grasp([0.9, 100, 100, 0.5, 20, 0], (480, 640))


def test_validator_accepts_good_grasp():
    assert validate_grasp([0.9, 320, 240, 40, 20, 15], (480, 640)) == []


def test_validator_box_containment():
    boxes = [[100, 100, 200, 200]]
    assert validate_grasp([0.9, 500, 400, 30, 20, 0], (480, 640), boxes)
    assert validate_grasp([0.9, 150, 150, 30, 20, 0], (480, 640), boxes) == []


# ── session rollback ─────────────────────────────────────────────────────────

def test_best_round_prefers_valid_verdict():
    s = Session("q")
    r0 = s.new_round(); r0.grasp = [0.9, 1, 2, 3, 4, 5]; r0.verdict = "INVALID"
    r1 = s.new_round(); r1.grasp = [0.4, 9, 9, 3, 4, 5]; r1.verdict = "VALID"
    r2 = s.new_round(); r2.verdict = "INVALID"           # no grasp
    assert s.best_round() is r1


def test_best_round_falls_back_to_quality():
    s = Session("q")
    r0 = s.new_round(); r0.grasp = [0.9, 1, 2, 3, 4, 5]
    r1 = s.new_round(); r1.grasp = [0.4, 9, 9, 3, 4, 5]
    assert s.best_round() is r0


def test_best_round_none_when_no_grasps():
    s = Session("q")
    s.new_round()
    assert s.best_round() is None


# ── cache ────────────────────────────────────────────────────────────────────

def test_cache_memoizes_by_coords_and_args(monkeypatch):
    import sys, types
    calls = {"find": 0}

    class FakePatch:
        def __init__(self, left=0, lower=0, right=640, upper=480):
            self.left, self.lower, self.right, self.upper = left, lower, right, upper

        def find(self, name):
            calls["find"] += 1
            return [f"patch-{name}"]

    fake_mod = types.ModuleType("image_patch")
    fake_mod.ImagePatch = FakePatch
    monkeypatch.setitem(sys.modules, "image_patch", fake_mod)

    from gmas.runtime.cache import make_cached_image_patch
    s = Session("q")
    Cached = make_cached_image_patch(s)

    p = Cached()
    assert p.find("apple") == ["patch-apple"]
    assert p.find("apple") == ["patch-apple"]      # hit
    assert p.find("banana") == ["patch-banana"]    # different args → miss
    q = Cached(10, 10, 50, 50)
    assert q.find("apple") == ["patch-apple"]      # different coords → miss
    assert calls["find"] == 3
    assert s.cache_hits == 1 and s.cache_misses == 3


# ── flags ────────────────────────────────────────────────────────────────────

def test_flags_parse():
    f = Flags.from_csv("repair,cache,timeout_s=120")
    assert f.repair and f.cache and f.timeout_s == 120 and not f.gate
    assert Flags.from_csv("").csv() == "-"
    a = Flags.from_csv("all,gate_q=0.9")
    assert a.repair and a.gate and a.rollback and a.registry_prompt
    assert a.gate_q == 0.9
    with pytest.raises(ValueError):
        Flags.from_csv("nonsense_flag")


def test_registry_prompt_injection():
    from gmas.agents.orchestrator import GmasPipeline
    aug = GmasPipeline.__new__(GmasPipeline)   # avoid llm init
    aug.flags = Flags.from_csv("registry_prompt")
    aug.registry = default_registry()
    prompt = aug._coder_prompt()
    assert "Authoritative API" in prompt.CODE
    assert "grasp_detection_directional" in prompt.CODE
    # placeholders must survive for str.format
    formatted = prompt.CODE.format(plan="P", example="E")
    assert "Plan at this step: P" in formatted
