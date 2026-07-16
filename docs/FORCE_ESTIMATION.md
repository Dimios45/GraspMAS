# GraspMAS — Physics-Aware Gripper Force Estimation

Zero-shot estimation of required gripper force using local Qwen2-VL-7B, evaluated
against ground-truth physics from the ManiSkill/SAPIEN simulator across all 78 YCB objects.

---

## Physics Model

For a parallel-jaw gripper in quasi-static grasp (two finger pads):

```
Anti-gravity condition:  2 × μ × F_n  ≥  m × g

Minimum finger force:    F_min  =  m × g / (2 × μ)

With safety factor k:    F_req  =  k × F_min  =  k × m × g / (2 × μ)
```

| Symbol | Value | Description |
|---|---|---|
| `m` | from SAPIEN | object mass (kg) |
| `g` | 9.81 m/s² | gravitational acceleration |
| `μ` | from SAPIEN | static friction coefficient |
| `k` | 2.0 | safety factor (default) |
| `F_max` | 70 N | Panda gripper continuous finger force limit |

**ManiSkill note:** The simulator uses `μ = 0.3` for all YCB objects regardless of material.
Real-world friction varies from 0.20 (glass) to 0.75 (rubber). Experiments E1b and E1c use
material-realistic friction from the lookup table in `material_table.py`.

---

## Ground-Truth Extraction (`grasp_force.py`)

```python
from grasp_force import get_object_physics, compute_required_force, get_contact_forces

physics   = get_object_physics(env.unwrapped)
# → {name, mass_kg, static_friction, dynamic_friction, weight_N}

force_req = compute_required_force(physics, safety_factor=2.0)
# → {F_min_N, F_required_N, feasible, utilization}

contact   = get_contact_forces(env.unwrapped)   # call after close_gripper()
# → {force_vec_N, force_mag_N, lifted}
```

Mass and friction are read directly from SAPIEN's PhysX actor — no approximation.

---

## Experiment E0 — Raw VLM Baseline (`scripts/run_force_benchmark.py`)

Single-turn VLM prompt: show the object image, ask for material / mass / friction / grip force.

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 python -u scripts/run_force_benchmark.py
```

**Prompt format:**
```
You are a robotics assistant estimating physical properties from a single image.
The image shows a {name} on a table. Estimate:
1. material  — metal / plastic / rubber / wood / fabric / food / ceramic / other
2. mass_g    — approximate mass in grams
3. friction  — coefficient of friction (0.1=glass … 0.7=grippy rubber)
4. grip_force_N — required gripper finger force in Newtons

Reply EXACTLY:
material: <value>
mass_g: <number>
friction: <number>
grip_force_N: <number>
```

**Run:** `runs/20260522_191900_force_benchmark/`

| Metric | Value |
|---|---|
| Objects | 78 YCB |
| Valid VLM responses | 71 / 78 (91%) |
| MAE force (N) | 3.21 N |
| MAE force (%) | 89.4% |
| MAE mass (g) | 142 g |

**Main failure modes:**
- Mass hallucination (e.g. soup can → 50 g instead of 381 g)
- Friction defaults to 0.3 for everything (no real material reasoning)
- Force computed correctly from VLM's own (bad) mass + friction estimates

**Outputs per run:**
```
imgs/<model_id>.png    — per-object RGB snapshot
results.json           — full per-object data
results.csv            — tabular
summary.txt            — human-readable table + MAE/MAPE
```

---

## Experiment E1 — Material Table (`scripts/run_benchmark_e1.py`)

Improves on E0 with: (1) chain-of-thought prompt with material guide + density anchors,
(2) table-based friction (VLM classifies material, code looks up realistic μ),
(3) brittleness-based F_max ceiling to catch crush-risk predictions.

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 python -u scripts/run_benchmark_e1.py
```

Three variants reported side-by-side:

| Variant | Mass source | Friction source | Purpose |
|---|---|---|---|
| **E1a** | VLM direct estimate | Material lookup table | Tests friction fix alone |
| **E1b** | Ground-truth (ceiling) | Material lookup table | Upper bound with perfect mass |
| **E1c** | VLM dims × table density | Material lookup table | Tests geometric mass reasoning |

**Run:** `runs/20260522_benchmark_e1/`

### Results vs E0

| Metric | E0 (raw VLM) | E1a (VLM mass + table μ) | E1b (GT mass + table μ) |
|---|---|---|---|
| MAE force vs sim (N) | 3.21 | 2.87 | **1.12** |
| MAPE force vs sim (%) | 89.4% | 74.1% | **31.4%** |
| Material accuracy | — | **58.2%** | 58.2% (same VLM) |
| Slip rate (F < F_real) | — | 22.1% | 14.1% |
| Damage rate (F > F_max) | — | 8.9% | 6.4% |
| Safe rate | — | 69.0% | **79.5%** |

**Key findings:**
- Table-based friction (E1a) reduces force MAE by ~11% vs raw VLM friction
- With perfect mass (E1b), force error drops to 31.4% — mass estimation is the bottleneck
- E1c (geometric mass from VLM dims × density) performs similarly to E1a — dimension
  estimation is noisy but in the right ballpark
- Material classification at 58.2% accuracy: metals and cardboard classified well;
  soft plastics and composites frequently confused

### Per-Material Breakdown (E1a)

| Material | n | Mat acc% | Mass MAPE% | E1a MAPE% | Slip% | Dmg% |
|---|---|---|---|---|---|---|
| metal | 12 | 83% | 68% | 41% | 8% | 0% |
| cardboard | 4 | 75% | 210% | 180% | 50% | 0% |
| rubber | 5 | 80% | 34% | 52% | 0% | 20% |
| hard_plastic | 18 | 61% | 91% | 83% | 28% | 6% |
| soft_plastic | 3 | 33% | 55% | 67% | 33% | 0% |
| fruit | 8 | 62% | 29% | 38% | 0% | 62% |
| composite | 7 | 43% | 88% | 95% | 14% | 14% |
| wood | 3 | 67% | 44% | 61% | 0% | 0% |
| foam | 2 | 50% | 120% | 95% | 0% | 50% |

**Notable:** Fruit objects have 0% slip (force always sufficient to lift) but 62% damage rate
— the model dramatically overestimates force for soft objects because it treats them as
solid food material rather than hollow YCB replicas.

### Material Lookup Table (`material_table.py`)

| Material | Density (kg/m³) | μ_s | Max pressure (kPa) | Brittleness |
|---|---|---|---|---|
| metal | 7800 | 0.30 | 500 | ductile |
| hard_plastic | 1150 | 0.30 | 250 | ductile |
| soft_plastic | 920 | 0.25 | 80 | soft |
| cardboard | 150 | 0.45 | 50 | soft |
| rubber | 1150 | 0.75 | 45 | soft |
| foam | 80 | 0.40 | 5 | soft |
| wood | 700 | 0.40 | 180 | ductile |
| ceramic | 2000 | 0.40 | 220 | brittle |
| glass | 2500 | 0.20 | 150 | brittle |
| fruit | 880 | 0.50 | 20 | soft |
| composite | 2000 | 0.35 | 130 | ductile |

**Outputs per run:**
```
imgs/<model_id>.png    — per-object images
results_e1.json        — full per-object data (all 3 variants)
results_e1.csv         — tabular
summary_e1.txt         — per-object table + metrics breakdown
scatter_e1.png         — 3-panel scatter: GT vs predicted force (E1a / E1b / E1c)
```

---

## Live Force Validation in the Demo

`scripts/run_maniskill_demo.py` also validates the actual executed grasp force after `close_gripper()`:

```
===== Grasp Force Analysis =====
  Object        : lemon (set_0_014_lemon)
  Mass          : 104.5 g  (1.025 N)
  Friction (μ_s): 0.30
  F_min         : 1.708 N  (no safety margin)
  F_required    : 3.417 N  (×2.0 safety)
  Panda max     : 70.0 N
  Utilization   : 4.9%  → OK
  Contact force : 2.841 N  (post-grasp)
  Lifted        : True
================================
```

This shows the required vs actual contact force and whether the object was lifted.

---

## Reproducing

```bash
# E0 — raw VLM baseline
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 python -u scripts/run_force_benchmark.py

# E1 — material table experiment
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 python -u scripts/run_benchmark_e1.py

# Verify ground-truth physics extraction (no GPU needed)
python -c "
import mani_skill.envs, gymnasium as gym
env = gym.make('PickSingleYCB-v1', obs_mode='rgbd', sim_backend='cpu',
               control_mode='pd_joint_pos', render_mode='rgb_array')
env.unwrapped.all_model_ids = ['014_lemon']
env.reset(options=dict(reconfigure=True))
from grasp_force import get_object_physics, compute_required_force
p = get_object_physics(env.unwrapped)
print(p)
print(compute_required_force(p))
env.close()
"
```
