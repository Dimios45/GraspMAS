"""Lightweight per-query telemetry for the GraspMAS ↔ gmas A/B evaluation.

Usage (eval harness):
    tel = Telemetry.start(meta={"query": q})
    ... run pipeline ...
    tel = Telemetry.stop()
    row.update(tel.summary())

The hook in ``local_vlm.vlm_chat`` records every LLM call (input/output
tokens, wall time) into the active collector, tagged with the agent that
issued it (set by the instrumented wrappers). When no collector is active
the hooks are no-ops, so normal runs are unaffected.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class LLMCall:
    tag: str                # planner / coder / observer / vqa / unknown
    wall_s: float
    input_tokens: int
    output_tokens: int
    n_images: int = 0


class Telemetry:
    """Per-query event collector. One active instance at a time (the
    pipeline is fully serialized on a single local VLM, so no need for
    anything fancier)."""

    _active: Optional["Telemetry"] = None

    def __init__(self, meta: Optional[Dict[str, Any]] = None):
        self.meta = meta or {}
        self.t0 = time.time()
        self.llm_calls: List[LLMCall] = []
        self.events: List[Dict[str, Any]] = []
        self.current_tag: str = "unknown"
        self.vlm_load_s: float = 0.0

    # ── lifecycle ──────────────────────────────────────────────────────
    @classmethod
    def start(cls, meta: Optional[Dict[str, Any]] = None) -> "Telemetry":
        cls._active = Telemetry(meta)
        return cls._active

    @classmethod
    def stop(cls) -> Optional["Telemetry"]:
        cur, cls._active = cls._active, None
        return cur

    @classmethod
    def active(cls) -> Optional["Telemetry"]:
        return cls._active

    # ── recording ──────────────────────────────────────────────────────
    def record_llm(self, wall_s: float, input_tokens: int, output_tokens: int,
                   n_images: int = 0, tag: Optional[str] = None) -> None:
        self.llm_calls.append(LLMCall(
            tag=tag or self.current_tag, wall_s=wall_s,
            input_tokens=input_tokens, output_tokens=output_tokens,
            n_images=n_images,
        ))

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"kind": kind, "t": round(time.time() - self.t0, 3), **data})

    # ── reporting ──────────────────────────────────────────────────────
    def summary(self) -> Dict[str, Any]:
        per_tag: Dict[str, Dict[str, float]] = {}
        for c in self.llm_calls:
            d = per_tag.setdefault(c.tag, {"calls": 0, "wall_s": 0.0,
                                           "input_tokens": 0, "output_tokens": 0})
            d["calls"] += 1
            d["wall_s"] = round(d["wall_s"] + c.wall_s, 2)
            d["input_tokens"] += c.input_tokens
            d["output_tokens"] += c.output_tokens
        rounds = sum(1 for e in self.events if e["kind"] == "planner_ok")
        return {
            "llm_calls": len(self.llm_calls),
            "llm_wall_s": round(sum(c.wall_s for c in self.llm_calls), 2),
            "input_tokens": sum(c.input_tokens for c in self.llm_calls),
            "output_tokens": sum(c.output_tokens for c in self.llm_calls),
            "llm_per_tag": per_tag,
            "rounds": rounds,
            "vlm_load_s": round(self.vlm_load_s, 1),
            "agent_errors": [e for e in self.events if e["kind"].endswith("_error")],
        }

    def dump(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "llm_calls": [asdict(c) for c in self.llm_calls],
            "events": self.events,
            "summary": self.summary(),
        }
