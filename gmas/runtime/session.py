"""Per-query session state: round snapshots, best-candidate rollback,
and the cross-round tool-result cache."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoundSnapshot:
    round: int
    plan: Optional[str] = None
    code: Optional[str] = None
    grasp: Optional[List[float]] = None
    approach: str = "center"
    verdict: str = "UNKNOWN"          # VALID | INVALID | AUTO_VALID | UNKNOWN
    observer_summary: str = ""
    validator_ok: Optional[bool] = None
    error: Optional[str] = None
    found_boxes: List[List[float]] = field(default_factory=list)


# verdict ranking for rollback_to_best: prefer validated grasps, then any grasp
_VERDICT_RANK = {"AUTO_VALID": 3, "VALID": 3, "UNKNOWN": 1, "INVALID": 0}


class Session:
    def __init__(self, query: str, meta: Optional[Dict[str, Any]] = None):
        self.id = uuid.uuid4().hex[:12]
        self.query = query
        self.meta = meta or {}
        self.t0 = time.time()
        self.rounds: List[RoundSnapshot] = []
        self.tool_cache: Dict[Any, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def new_round(self) -> RoundSnapshot:
        snap = RoundSnapshot(round=len(self.rounds))
        self.rounds.append(snap)
        return snap

    def best_round(self) -> Optional[RoundSnapshot]:
        """The best grasp-bearing round: highest verdict rank, then
        validator approval, then grasp quality score, then recency."""
        candidates = [r for r in self.rounds if r.grasp]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (
            _VERDICT_RANK.get(r.verdict, 1),
            1 if r.validator_ok else 0,
            r.grasp[0] if r.grasp else 0.0,
            r.round,
        ))
