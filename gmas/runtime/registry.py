"""ToolRegistry — the single source of truth for what the Coder may call.

Tools are the public methods of ImagePatch. The registry provides:
  * the sandbox whitelist (which attribute calls are legal in Coder code)
  * capability discovery: the API section of the Coder prompt is generated
    from the registry (P3 feature flag `registry_prompt`), so the prompt can
    never drift from the actually-callable surface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolSpec:
    name: str
    signature: str
    returns: str
    description: str
    notes: str = ""


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> List[str]:
        return list(self._tools)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    # ── capability discovery: Coder prompt API section ──────────────────
    def api_doc(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"    {t.name}{t.signature} -> {t.returns}")
            lines.append(f"        {t.description}")
            if t.notes:
                lines.append(f"        NOTE: {t.notes}")
        return "\n".join(lines)


def default_registry() -> ToolRegistry:
    """The current ImagePatch surface (mirrors image_patch.py)."""
    r = ToolRegistry()
    r.register(ToolSpec(
        "find", "(object_name: str)", "list[ImagePatch]",
        "Detect objects by name; returns crops centered on each match.",
        "Returns [None] if nothing found — always guard: if patches[0] is None: return None",
    ))
    r.register(ToolSpec(
        "find_part", "(object_name: str, part_name: str)", "ImagePatch",
        "Locate a specific part of an object (e.g. handle of a knife).",
    ))
    r.register(ToolSpec(
        "exists", "(object_name: str)", "bool",
        "True if the named object is present in the crop.",
    ))
    r.register(ToolSpec(
        "verify_property", "(object_name: str, visual_property: str)", "bool",
        "Check a visual property (color, shape, material) of an object.",
    ))
    r.register(ToolSpec(
        "simple_query", "(question: str = None)", "str",
        "Answer a basic perception question about the crop.",
    ))
    r.register(ToolSpec(
        "llm_query", "(question: str)", "str",
        "Ask a complex/external-knowledge question about the image.",
    ))
    r.register(ToolSpec(
        "compute_depth", "()", "float",
        "Median monocular depth of the crop (relative units).",
    ))
    r.register(ToolSpec(
        "best_image_match", "(list_patches: list[ImagePatch], content: list[str], return_index: bool = False)",
        "ImagePatch | int",
        "Return the patch most likely to contain the content.",
    ))
    r.register(ToolSpec(
        "grasp_detection", "(object_patch: ImagePatch)", "List[float]",
        "Best grasp rectangle for the patch: [quality, x, y, w, h, angle].",
        "Returns a plain list (or None), NOT an ImagePatch.",
    ))
    r.register(ToolSpec(
        "grasp_detection_directional", "(object_patch: ImagePatch, direction: str)", "List[float]",
        "Grasp rectangle biased toward one side; direction in "
        "{'top','bottom','left','right','center'}.",
        "Returns a plain list (or None), NOT an ImagePatch.",
    ))
    # geometry helpers usable in code but not 'tools' per se
    for name in ("crop", "overlaps_with"):
        r.register(ToolSpec(name, "(...)", "ImagePatch | bool",
                            "Geometry helper on ImagePatch."))
    return r
