"""Deterministic pre-return validation of grasp proposals.

Cheap geometric checks that catch obviously-broken grasps without spending
a VLM Observer call, and give the Planner precise, actionable feedback.
Deliberately not an LLM: these constraints must be verifiable.
"""

from typing import List, Optional, Tuple


def validate_grasp(grasp: Optional[List[float]],
                   image_hw: Tuple[int, int],
                   found_boxes: Optional[List[List[float]]] = None,
                   box_pad: float = 30.0) -> List[str]:
    """Return a list of violation strings (empty = approved).

    grasp: [quality, cx, cy, w, h, angle]
    image_hw: (height, width)
    found_boxes: [l, u, r, lo] boxes of objects found this round (image coords)
    """
    violations: List[str] = []
    if grasp is None:
        return ["no grasp produced"]
    if len(grasp) < 6:
        return [f"malformed grasp (expected 6 values, got {len(grasp)})"]

    q, cx, cy, w, h, angle = [float(v) for v in grasp[:6]]
    H, W = image_hw

    if not (0 <= cx < W and 0 <= cy < H):
        violations.append(f"grasp center ({cx:.0f},{cy:.0f}) outside image {W}x{H}")
    if w <= 1 or h <= 1:
        violations.append(f"degenerate rectangle w={w:.1f} h={h:.1f}")
    if w > W or h > H:
        violations.append(f"rectangle larger than image (w={w:.0f}, h={h:.0f})")

    if found_boxes:
        inside_any = any(
            (l - box_pad) <= cx <= (r + box_pad) and (u - box_pad) <= cy <= (lo + box_pad)
            for l, u, r, lo in found_boxes
        )
        if not inside_any:
            violations.append(
                "grasp center lies outside every detected object box — "
                "likely grasping the wrong region")
    return violations
