"""GmasPipeline — the GraspMAS loop re-hosted on the gmas runtime.

Drop-in replacement for ``agents.graspmas.GraspMAS`` (same ``query()``
signature) that reuses the legacy Planner/Coder/Observer verbatim — same
prompts, same LLM — but routes every hop through typed messages, executes
Coder output in the sandbox instead of ``exec(code, globals())``, and adds
the P2/P3 features behind ``gmas.config.Flags`` (all off = parity mode).

Deliberate parity notes (legacy quirks reproduced when flags are off):
  * a Coder syntax error aborts the whole query (legacy ran exec() at the
    top level, outside try) — with ``repair`` on it becomes a recoverable
    round instead;
  * ``grasp_pose`` persists across rounds: a failed round does not clear a
    previous round's grasp;
  * the Observer runs even after error rounds (that is how the Planner
    learns what went wrong).

One intentional divergence: legacy leaked ``execute_command`` into module
globals, so a round whose code failed to define it silently re-ran the
previous round's function. The sandbox uses a fresh namespace per round and
reports "code defines no execute_command" instead.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from agents.coder import Coder
from agents.graspmas import GraspMAS
from agents.llm import OpenAILLM
from agents.observer import Observer
from agents.planner import Planner
from agents.prompt import coder_prompt, observer_prompt, planner_prompt
from grasp.utils import visualize_grasp_pose

from gmas.config import Flags
from gmas.runtime.audit import AuditLog
from gmas.runtime.bus import MessageBus
from gmas.runtime.messages import (CodeMsg, ExecResult, FinalResult,
                                   ObservationMsg, PlanMsg, QueryMsg,
                                   ValidationResult)
from gmas.runtime.registry import default_registry
from gmas.runtime.sandbox import run_code
from gmas.runtime.session import Session
from gmas.runtime.validator import validate_grasp


class GmasPipeline:
    def __init__(self, api_file: str, max_round: int = 5,
                 flags: Optional[Flags] = None):
        llm = OpenAILLM(api_file)
        self.flags = flags or Flags()
        self.max_round = max_round
        self.registry = default_registry()
        self.planner = Planner(planner_prompt, llm)
        self.coder = Coder(self._coder_prompt(), llm)
        self.observer = Observer(observer_prompt, llm)
        self.plan = None
        self.observation = None
        self.code = None
        self.last_approach: str = "center"

    def _coder_prompt(self):
        """The Coder prompt module, optionally augmented with the registry's
        authoritative API list so the prompt can never drift from the real
        callable surface (P3 `registry_prompt`)."""
        if not self.flags.registry_prompt:
            return coder_prompt
        # escape braces: CODE goes through str.format(plan=..., example=...)
        api = self.registry.api_doc().replace("{", "{{").replace("}", "}}")
        block = ("** Authoritative API (auto-generated from the tool registry) **\n"
                 "The ONLY callable ImagePatch methods are:\n" + api +
                 "\nAny method not listed above DOES NOT EXIST and must not be called.\n\n")
        anchor = "Write a function using Python"
        code = coder_prompt.CODE.replace(anchor, block + anchor, 1)
        return SimpleNamespace(CODE=code, EXAMPLES_CODER=coder_prompt.EXAMPLES_CODER)

    async def query(
        self,
        query: str,
        image: Union[str, np.ndarray],
        save_folder: Union[str, Path] = "imgs/",
    ) -> Tuple[Any, Optional[List]]:
        img_np, image_path = GraspMAS._prepare_image(image, save_folder)
        flags = self.flags

        session = Session(query, meta={"flags": flags.csv()})
        image_patch_cls = None
        if flags.cache:
            from gmas.runtime.cache import make_cached_image_patch
            image_patch_cls = make_cached_image_patch(session)
        bus = MessageBus()
        audit = AuditLog(str(Path(save_folder) / "audit.jsonl"))
        bus.subscribe("*", audit)
        sid = session.id

        await bus.publish(QueryMsg(session_id=sid, sender="user",
                                   query=query, image_path=str(image_path)))

        grasp_pose: Optional[List] = None
        out: Any = None
        try:
            for idx in range(self.max_round):
                snap = session.new_round()
                print("=" * 10, f"Round {idx}", "=" * 10)

                # ── Planner ──────────────────────────────────────────────
                self.thought, self.plan = await self.planner(
                    query=query, previous_plan=self.plan,
                    observation=self.observation)
                snap.plan = self.plan
                print("----- Thought -----\n", self.thought)
                print("----- Plan -----\n", self.plan)
                terminate = "return to user" in (self.plan or "").lower()
                await bus.publish(PlanMsg(session_id=sid, sender="planner",
                                          round=idx, thought=self.thought,
                                          plan=self.plan, terminate=terminate))
                if terminate:
                    break

                # ── Coder + sandbox ─────────────────────────────────────
                self.code = await self.coder(self.plan)
                print("----- Code -----\n", self.code)
                res = run_code(
                    self.code, img_np,
                    allowed_tools=set(self.registry.names()),
                    repair=flags.repair, whitelist=flags.whitelist,
                    timeout_s=flags.timeout_s, image_patch_cls=image_patch_cls)
                snap.code = res.code_used
                await bus.publish(CodeMsg(session_id=sid, sender="coder",
                                          round=idx, code=res.code_used,
                                          repaired=res.repaired,
                                          repair_note=res.repair_note))

                if not res.ok and res.error_kind == "syntax" and not flags.repair:
                    # parity: legacy exec() raised SyntaxError at the top level
                    raise SyntaxError(res.error)

                # ── unpack execution result (same as legacy) ─────────────
                grasp_list = None
                if res.ok:
                    out_raw = res.out
                    if isinstance(out_raw, dict):
                        grasp_list = out_raw.get("grasp")
                        self.last_approach = out_raw.get("approach", "center")
                    elif isinstance(out_raw, list):
                        grasp_list = out_raw
                        self.last_approach = "center"
                else:
                    out_raw = res.error
                    print("Error:", out_raw)
                if isinstance(grasp_list, list):
                    grasp_pose = grasp_list.copy()
                    snap.grasp = grasp_pose
                    snap.approach = self.last_approach
                snap.error = None if res.ok else res.error

                result = {"grasp": None, "image": str(image_path), "error_logs": None}
                if isinstance(grasp_list, list):
                    result["grasp"] = grasp_list
                    vis_path = visualize_grasp_pose(img_np, grasp_list, save_folder)
                    print("Grasp Pose Visualization saved at:", vis_path)
                    result["image"] = str(vis_path)
                else:
                    result["error_logs"] = str(out_raw)
                await bus.publish(ExecResult(
                    session_id=sid, sender="sandbox", round=idx, ok=res.ok,
                    grasp=snap.grasp if isinstance(grasp_list, list) else None,
                    approach=self.last_approach, error=snap.error,
                    raw_type=type(res.out).__name__ if res.ok else "",
                    vis_path=result["image"]))
                print("----- Execution Result -----\n", result)

                # ── Validator (flag) ─────────────────────────────────────
                if flags.validator and isinstance(grasp_list, list):
                    violations = validate_grasp(
                        grasp_list, img_np.shape[:2], snap.found_boxes or None)
                    snap.validator_ok = not violations
                    await bus.publish(ValidationResult(
                        session_id=sid, sender="validator", round=idx,
                        approved=not violations, violations=violations))
                    if violations:
                        result["error_logs"] = ("validator rejected grasp: "
                                                + "; ".join(violations))

                # ── Observer (confidence-gated when flags.gate) ──────────
                quality = float(grasp_list[0]) if isinstance(grasp_list, list) \
                    and grasp_list else 0.0
                gated = (flags.gate and res.ok and snap.grasp
                         and quality >= flags.gate_q
                         and snap.validator_ok is not False)
                if gated:
                    snap.verdict = "AUTO_VALID"
                    self.observation = (f"Execution succeeded; grasp quality "
                                        f"{quality:.2f} ≥ {flags.gate_q:g} — "
                                        f"auto-approved without Observer.")
                    print(f"[gmas] observer gate: quality {quality:.2f}, skipping VLM call")
                else:
                    observation = await self.observer(result, query)
                    print("----- Observation -----\n", observation)
                    self.observation = observation.get("summary") \
                        if isinstance(observation, dict) else str(observation)
                    snap.verdict = (observation.get("verdict") or "UNKNOWN").upper() \
                        if isinstance(observation, dict) else "UNKNOWN"
                snap.observer_summary = self.observation or ""
                await bus.publish(ObservationMsg(
                    session_id=sid, sender="observer", round=idx,
                    verdict=snap.verdict, summary=snap.observer_summary,
                    auto=bool(gated)))

                # ── Early termination (flag) ─────────────────────────────
                if flags.early_term and snap.verdict in ("VALID", "AUTO_VALID") \
                        and snap.validator_ok is not False and snap.grasp:
                    print(f"[gmas] early termination: round {idx} verdict {snap.verdict}")
                    break

            # ── Rollback (flag): return best candidate, not last ─────────
            source_round, via_rollback = -1, False
            if flags.rollback:
                best = session.best_round()
                if best is not None and best.grasp != grasp_pose:
                    grasp_pose = list(best.grasp)
                    self.last_approach = best.approach
                    via_rollback = True
                if best is not None:
                    source_round = best.round
            await bus.publish(FinalResult(
                session_id=sid, sender="orchestrator", grasp=grasp_pose,
                approach=self.last_approach, source_round=source_round,
                via_rollback=via_rollback))
        finally:
            audit.close()
        return out, grasp_pose
