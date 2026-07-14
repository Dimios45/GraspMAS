"""Typed message schemas for the gmas runtime.

Every hop in the agent loop is a typed message published on the bus and
mirrored to the audit log. Free-text only lives *inside* fields, never as
the transport itself.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _mid() -> str:
    return uuid.uuid4().hex[:12]


class Msg(BaseModel):
    session_id: str
    sender: str
    topic: str
    ts: float = Field(default_factory=time.time)
    msg_id: str = Field(default_factory=_mid)
    round: int = -1


class QueryMsg(Msg):
    topic: str = "query"
    query: str = ""
    image_path: str = ""


class PlanMsg(Msg):
    topic: str = "plan"
    thought: str = ""
    plan: str = ""
    terminate: bool = False   # planner said "return to user"


class CodeMsg(Msg):
    topic: str = "code"
    code: str = ""
    repaired: bool = False
    repair_note: str = ""


class ExecResult(Msg):
    topic: str = "exec"
    ok: bool = False
    grasp: Optional[List[float]] = None
    approach: str = "center"
    error: Optional[str] = None
    raw_type: str = ""        # type name of what execute_command returned
    vis_path: Optional[str] = None


class ToolCallRecord(Msg):
    topic: str = "tool"
    tool: str = ""
    args: str = ""
    cache_hit: bool = False
    wall_s: float = 0.0
    result_summary: str = ""
    boxes: Optional[List[List[float]]] = None   # for find/find_part: [l,u,r,lo]


class ValidationResult(Msg):
    topic: str = "validation"
    approved: bool = True
    violations: List[str] = Field(default_factory=list)


class ObservationMsg(Msg):
    topic: str = "observation"
    verdict: str = "UNKNOWN"      # VALID | INVALID | AUTO_VALID | SKIPPED
    checklist: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    auto: bool = False            # produced by the gate, not the VLM


class FinalResult(Msg):
    topic: str = "final"
    grasp: Optional[List[float]] = None
    approach: str = "center"
    source_round: int = -1        # which round the returned grasp came from
    via_rollback: bool = False
