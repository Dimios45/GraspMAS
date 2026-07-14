"""Feature flags for the gmas runtime.

Every flag defaults to OFF: with all flags off the runtime must reproduce
legacy GraspMAS behavior (the P1 parity gate). Each flag is also an ablation
switch for the A/B eval.
"""

from dataclasses import dataclass, fields


@dataclass
class Flags:
    repair: bool = False        # AST auto-repair of Coder syntax errors + re-prompt
    whitelist: bool = False     # static registry whitelist check before execution
    validator: bool = False     # deterministic pre-return grasp checks
    cache: bool = False         # per-session tool-result cache
    gate: bool = False          # confidence-gated Observer skip
    rollback: bool = False      # return best round's grasp instead of last
    registry_prompt: bool = False  # inject registry-generated API list into Coder prompt
    early_term: bool = False    # stop on validator-approved VALID verdict
    timeout_s: float = 0.0      # per-round code execution timeout (0 = none)
    gate_q: float = 0.85        # min grasp quality for the Observer gate

    @classmethod
    def from_csv(cls, csv: str) -> "Flags":
        """Parse "repair,cache,timeout_s=120" style flag strings; "all" turns
        every boolean flag on."""
        f = cls()
        csv = (csv or "").strip()
        if not csv:
            return f
        bool_names = {x.name for x in fields(cls) if x.type == "bool" or x.type is bool}
        for part in csv.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                for n in bool_names:
                    setattr(f, n, True)
            elif "=" in part:
                k, v = part.split("=", 1)
                setattr(f, k.strip(), float(v))
            elif part in bool_names:
                setattr(f, part, True)
            else:
                raise ValueError(f"unknown flag: {part}")
        return f

    def csv(self) -> str:
        on = [x.name for x in fields(self)
              if isinstance(getattr(self, x.name), bool) and getattr(self, x.name)]
        if self.timeout_s:
            on.append(f"timeout_s={self.timeout_s:g}")
        return ",".join(on) or "-"
