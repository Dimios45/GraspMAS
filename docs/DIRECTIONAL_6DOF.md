# Directional 6-DoF Grasping Extension

Language-conditioned approach direction for YCB object grasping in ManiSkill.
Built on top of the GraspMAS multi-agent pipeline.

---

## Motivation

The base GraspMAS paper outputs a **2D grasp rectangle** `(quality, cx, cy, w, h, θ)` and projects it to 3D using camera depth + a fixed top-down approach direction `[0, 0, −1]`. Every prompt — "pick from the top", "pick from the right", "pick from the bottom" — produces the same gripper trajectory. Only the jaw angle `θ` changes.

This extension maps the **direction word in the natural language prompt** to a distinct 3D approach vector, giving the robot genuinely different trajectories for each directional prompt.

---

## Pipeline Changes

### 1. Directional candidate selection in RAGT (`detect_grasp_directional`)

`grasp/unit_grasp_pose_generation.py` — `detect_grasp_directional(grasp_model, image, mask, device, direction)`

Standard `detect_grasp()` picks the single highest-quality candidate from RAGT GraspNet. The directional version:

1. Lowers the confidence threshold progressively (`0.5 → 0.2 → 0.05`) to get ≥ 3 candidates.
2. Computes a **target point** 30% in from the requested edge of the object mask:
   ```
   top    → (cx_mid,  ys.min + (cy_mid − ys.min) × 0.3)
   bottom → (cx_mid,  ys.max − (ys.max − cy_mid) × 0.3)
   left   → (xs.min + (cx_mid − xs.min) × 0.3,  cy_mid)
   right  → (xs.max − (xs.max − cx_mid) × 0.3,  cy_mid)
   ```
3. Picks the candidate whose center `(cx, cy)` is closest to that target point.

The jaw angle `θ` from the selected candidate is used for the 3D closing direction.

### 2. ImagePatch API extension

`image_patch.py` — `ImagePatch.grasp_detection_directional(object_patch, direction)`

Wraps `detect_grasp_directional` with the same resize / coordinate-rescale logic as the existing `grasp_detection()`. Exposed to the VLM Coder agent via the prompt docstring with examples.

Coder prompt rule added (`coder_prompt.py`):
> "Use `grasp_detection_directional(patch, direction)` when the query mentions a specific side (top/bottom/left/right). Use `grasp_detection(patch)` otherwise."

### 3. 2D → 6-DoF pose construction

The 2D grasp rectangle `(q, cx, cy, w, h, θ)` is projected to a full 6-DoF `sapien.Pose` as follows:

```
cx, cy  ← discarded (patch-relative artefacts from PickSingleYCB crop geometry)
θ       ← jaw closing direction in world XY plane
```

**Step A — 3D grasp center:**

```python
obj_pos  = env.unwrapped._objs[0].pose.p[0]          # GT simulator position
bounds   = collision_mesh.bounding_box.bounds          # AABB in local frame
hy       = (bounds[1,1] − bounds[0,1]) / 2 × 0.55    # half Y-extent (55%)
hz       = (bounds[1,2] − bounds[0,2]) / 2 × 0.55    # half Z-extent (55%)

dir_offset = {
    "top"   : [0,   0,  hz ],   # upper face
    "bottom": [0,   0, −hz×0.5],# lower face (table blocks full −hz)
    "right" : [0,  hy,  0  ],   # +Y face  (camera-right = world +Y)
    "left"  : [0, −hy,  0  ],   # −Y face  (camera-left  = world −Y)
}
grasp_center = obj_pos + dir_offset[direction]
```

**Step B — approach direction:**

The scene coordinate frame has the **robot base at X = −0.615** and the **camera at (0.3, 0, 0.6) looking toward (−0.1, 0, 0.1)**. Because the camera looks in the −X direction, its right axis is world **+Y** (not +X).

```python
approach_dir = {
    "top"   : [ 0,  0, −1] / 1,        # straight down
    "bottom": [ 0,  0, −1] / 1,        # straight down, lower contact point
    "right" : [ 0, −1, −1] / √2,       # from +Y side at 45°
    "left"  : [ 0, +1, −1] / √2,       # from −Y side at 45°
}
```

> **Why not X-axis for left/right?**
> The robot base is at X = −0.615. Approaching from +X (the far side) requires
> the arm to reach past the object — outside the comfortable workspace. Approaches
> in Y are symmetric and reachable.

**Step C — Gram-Schmidt orthogonalisation:**

`build_grasp_pose` requires `approach_dir ⊥ closing_dir`. After changing `approach_dir`
from `[0,0,−1]` to an angled vector, the XY-plane closing direction is no longer orthogonal:

```python
closing_dir = [cos(θ), sin(θ), 0]
# project out the approach component
closing_dir = closing_dir − dot(closing_dir, approach_dir) × approach_dir
closing_dir = closing_dir / ‖closing_dir‖
```

**Step D — build pose and execute:**

```python
grasp_6d = env.unwrapped.agent.build_grasp_pose(approach_dir, closing_dir, grasp_center)

planner.move_to_pose_with_screw(grasp_6d * Pose([0, 0, −0.05]))  # pre-grasp
planner.move_to_pose_with_screw(grasp_6d * Pose([0.005, 0, 0.015]))  # close in
planner.close_gripper()
planner.move_to_pose_with_screw(grasp_6d * Pose([0, 0, −0.4]))   # lift
```

---

## Coordinate Frame Reference

```
Top-down view of the ManiSkill PickSingleYCB scene:

         +Y  (camera-right, "pick from right" direction)
          │
          │
 Robot ───┼──────── Object ─────── Camera
X=−0.615  │         X≈0.07          X=+0.3
          │         Y≈0.08
         −Y  (camera-left, "pick from left" direction)

         −X ────────────────────── +X
       (robot side)             (camera side)

+Z = up (out of page)
```

Camera is at `eye=(0.3, 0, 0.6)`, `target=(−0.1, 0, 0.1)` → forward ≈ (−0.625, 0, −0.781), right = +Y (Z-up cross product).

---

## Results

### Full batch: 49 YCB objects × 4 directions (196 attempts, seed=42)

> Note: multi-variant objects (4 cups, 2 toy_airplane, 7 lego_duplo) share a folder name
> and resume-skip logic de-duplicates them. Effective unique objects run: 39–40 per direction.

| Direction | Approach vector | Success |
|-----------|----------------|---------|
| top       | `[0,  0, −1]`  | 24/39 **(62%)** |
| right     | `[0, −1, −1]/√2` | 13/39 **(33%)** |
| left      | `[0, +1, −1]/√2` | 6/39 **(15%)** |
| bottom    | `[0,  0, −1]`  | 24/40 **(60%)** |
| **Total** | | **67/157 (42.7%)** |

### Perfect objects (4/4 directions successful)

| Object | Notes |
|--------|-------|
| apple | round, symmetric |
| lemon | oval, graspable from all sides |
| orange | round, symmetric |
| softball | round, symmetric |
| tennis ball | round, symmetric |
| cups (065-f) | cylindrical, stable geometry |

### Zero-success objects (0/4)

cracker_box, sponge, power_drill, scissors, adjustable_wrench, hammer, extra_large_clamp, foam_brick, toy_airplane — all elongated, thin, or geometrically complex.

### Ablation: direction axis bug history

| Run | right approach | right success | total |
|-----|---------------|---------------|-------|
| Broken (X-axis) | `[−1, 0, −1]/√2` | 0/39 **(0%)** | 29.5% |
| Fixed (Y-axis)  | `[0, −1, −1]/√2` | 13/39 **(33%)** | 42.7% |

The original mapping used ±X for left/right. Because the robot sits at X = −0.615 and the
object at X ≈ 0, approaching from +X puts the pre-grasp pose *past* the object away from the
robot — kinematically inaccessible. Switching to ±Y (the correct camera-frame axis) fixed
the right direction entirely.

---

## Semantic Part Grasping (`run_clamp_parts_demo.py`)

For objects with named parts, `find_part(object, part)` (backed by VLpart) locates a sub-region
mask. The same 6-DoF pipeline applies with part-specific approach directions:

```python
# Medium clamp (050_medium_clamp)
prompts = [
    ("loop",         "Pick the medium clamp up by grasping the loop"),
    ("right_handle", "Pick the medium clamp up by grasping the right handle"),
    ("left_handle",  "Pick the medium clamp up by grasping the left handle"),
]

approach_dir = {
    "loop"        : [0,  0, −1],           # top-down for the spring loop
    "right_handle": [0, −1, −1] / √2,      # from +Y (camera-right)
    "left_handle" : [0, +1, −1] / √2,      # from −Y (camera-left)
}
```

Part-specific contact offsets are derived from VLM jaw angle `θ` rather than bbox extents,
since the part mask already localises the finger contact region.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_mustard_demo.py` | 4-direction demo for any single YCB object (`--object-id`) |
| `run_all_objects_demo.py` | Full batch: 49 objects × 4 directions with crash safety + resume |
| `run_clamp_parts_demo.py` | 3 semantic-part prompts on `050_medium_clamp` |

See [SCRIPTS.md](../SCRIPTS.md) for CLI flags and output layout.

---

## Known Limitations

- **left direction (15%)**: `[0, +1, −1]/√2` is geometrically correct but the Panda arm finds
  it harder than the old empirically-tuned `[1, 0, −1]/√2`. Further tuning or IK seeding may help.
- **GT position only**: grasp center uses the simulator's ground-truth object pose, not depth
  projection. A real-robot deployment would need camera-to-world depth unprojection.
- **Elongated objects (0%)**: thin or irregular geometry (scissors, drill, hammer) produces
  unreliable RAGT candidates regardless of direction. Part-based grasping (`find_part`) is
  recommended for these.
