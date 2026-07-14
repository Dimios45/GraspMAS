"""Instrumentation wrapper for the legacy GraspMAS pipeline.

Wraps the Planner/Coder/Observer callables of an existing ``GraspMAS``
instance with proxies that tag LLM calls, time each agent turn, and record
per-round events into the active ``Telemetry`` collector. Behavior is
unchanged: outputs and exceptions pass through untouched.
"""

import time
from typing import Any, Optional

from gmas.telemetry import Telemetry


class _AgentProxy:
    def __init__(self, agent: Any, tag: str):
        self._agent = agent
        self._tag = tag

    def __getattr__(self, name):
        return getattr(self._agent, name)

    async def __call__(self, *args, **kwargs):
        tel = Telemetry.active()
        if tel is not None:
            tel.current_tag = self._tag
        t0 = time.time()
        try:
            out = await self._agent(*args, **kwargs)
        except Exception as e:
            if tel is not None:
                tel.event(f"{self._tag}_error", error=repr(e)[:500],
                          wall_s=round(time.time() - t0, 2))
            raise
        else:
            if tel is not None:
                tel.event(f"{self._tag}_ok", wall_s=round(time.time() - t0, 2))
                if self._tag == "observer" and args:
                    # args[0] is the execution-result dict; its error_logs field
                    # is the only place runtime code errors surface without
                    # modifying the legacy loop.
                    result = args[0]
                    err = result.get("error_logs") if isinstance(result, dict) else None
                    # execute_command legitimately returns None when nothing is
                    # found — that is a detection miss, not a code error
                    if err and str(err) != "None":
                        tel.event("code_runtime_error", error=str(err)[:500])
            return out
        finally:
            if tel is not None:
                tel.current_tag = "unknown"


def instrument(graspmas: Any) -> Any:
    """Wrap a GraspMAS instance's agents in telemetry proxies (idempotent)."""
    if not isinstance(graspmas.planner, _AgentProxy):
        graspmas.planner = _AgentProxy(graspmas.planner, "planner")
    if not isinstance(graspmas.coder, _AgentProxy):
        graspmas.coder = _AgentProxy(graspmas.coder, "coder")
    if not isinstance(graspmas.observer, _AgentProxy):
        graspmas.observer = _AgentProxy(graspmas.observer, "observer")
    return graspmas


def classify_failure(error: Optional[str], grasp_pose, success: bool,
                     iou: float, tel_summary: dict) -> str:
    """Failure taxonomy for the baseline table.

    Categories: success / code_syntax_error / agent_parse_error /
    code_runtime_error / no_detection / wrong_location / bad_grasp /
    other_error
    """
    if success:
        return "success"
    if error:
        e = error.lower()
        if "invalid syntax" in e or "syntaxerror" in e or "outside function" in e \
                or "indentation" in e or "never closed" in e:
            return "code_syntax_error"
        agent_errors = tel_summary.get("agent_errors", [])
        kinds = {a["kind"] for a in agent_errors}
        if "planner_error" in kinds or "observer_error" in kinds:
            return "agent_parse_error"
        if "list index out of range" in e or "jsondecode" in e or "expecting value" in e:
            return "agent_parse_error"
        return "other_error"
    if grasp_pose is None:
        # look for runtime code errors recorded during rounds
        for ev in tel_summary.get("agent_errors", []):
            if ev["kind"] == "code_runtime_error":
                return "code_runtime_error"
        return "no_detection"
    if iou == 0.0:
        return "wrong_location"
    return "bad_grasp"
