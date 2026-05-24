# GraspMAS — YCB Object Mass Prediction

Zero-shot mass estimation for 78 YCB objects using local Qwen2-VL-7B on AMD MI300X (ROCm).  
Evaluated against the exact masses the ManiSkill simulator computes at runtime.

---

## Ground-Truth API

Mass is computed directly from simulation assets — no live ManiSkill run needed:

```python
import json, trimesh
from pathlib import Path

ASSET = Path("~/.maniskill/data/assets/mani_skill2_ycb").expanduser()

with open(ASSET / "info_pick_v0.json") as f:
    info = json.load(f)                          # density (kg/m³) + scale per object

mesh  = trimesh.load(str(ASSET / "models" / obj_id / "collision.ply"))
if not mesh.is_watertight:
    mesh = mesh.convex_hull

scale   = info[obj_id]["scales"][0]
density = info[obj_id]["density"]                # kg/m³  (sim value, may differ from real-world)
mass_kg = density * mesh.volume * scale**3       # ← exact value SAPIEN/PhysX uses
```

Verified: 78 objects, mass range 0.005–1.37 kg, all nonzero.  
Run `python property_inference.py --verify-gt` to print the full table.

**Important:** The sim uses simplified densities for box-shaped objects (`density=50 kg/m³`)
to prevent instability. These are not real-world masses.

| Object category | Sim density | Example |
|---|---|---|
| Cans, bottles, fruits, tools | 1000 kg/m³ | tomato soup can = 0.381 kg |
| Card/food boxes | 50 kg/m³ | cracker box = 0.123 kg (vs real ~0.411 kg) |

---

## Prediction Pipeline

```
Object label + bbox dims + k anchors
            │
            ▼  Qwen2-VL-7B (local, vlm_chat)
   Chain-of-thought decomp:
     (a) material → density
     (b) packing factor (hollow/solid/thin)
     (c) volume = bbox_volume × packing_factor
     (d) mass = density × volume
     (e) anchor check & reconcile
            │
            ▼  parse JSON → bounds check [0.001, 50 kg]
            │
            ▼  anchor blend (if |ratio| > 3×)
            │
            ▼  self-consistency: median of N=5 samples
            │
            ▼  (mass_kg, confidence)
```

**Key design choices:**

| Choice | Rationale |
|---|---|
| Bbox dimensions injected into prompt | Eliminates dimension hallucination (#1 error source in v1) |
| Embedding-nearest k=5 anchors (SentenceTransformer) | Grounds estimate in calibrated sim masses |
| Anchor-range constraint in prompt | Forces model to stay within 5× of nearest reference |
| Adaptive geometric blend post-processing | Corrects solid-metal misidentification (pitcher, skillet) |
| Expression evaluator in JSON parser | Handles `"mass_kg": 7800 * 0.002 * 0.5` LLM outputs |
| Physical bounds [0.001, 50 kg] | Wide enough to not reject valid predictions |

---

## Results — Full Eval (78 objects, 5-fold CV)

**Run:** `runs/mass_eval_v2/`  
**Model:** Qwen2-VL-7B-Instruct (local, AMD MI300X, ROCm 6.2.4, bfloat16)  
**Settings:** N=5 samples, T=0.9, k=5 anchors, anchor blend ON, no vision

### v1 → v2 Improvement Summary

| Metric | v1 (no bbox, wide bounds) | v2 (bbox + blend + better prompt) | Change |
|---|---|---|---|
| Detection rate | 87.2% (68/78) | **100.0% (78/78)** | +12.8pp |
| MAE_kg | 0.2308 kg | **0.1803 kg** | −22% |
| MAPE | 244.1% | **126.4%** | −48% |
| Median error | 83.3% | **70.0%** | −13pp |
| <25% error | 2.9% (2/68) | **20.5% (16/78)** | +7× |
| <50% error | 19.1% (13/68) | **32.1% (25/78)** | +1.7× |
| Failures | 10 objects | **0** | −10 |

### Error Distribution

| Error bucket | Count | % of detected |
|---|---|---|
| <25% | 16 | 20.5% |
| 25–50% | 9 | 11.5% |
| 50–100% | 37 | 47.4% |
| 100–200% | 5 | 6.4% |
| >200% | 11 | 14.1% |

### Best Predictions (<25% error)

| Object | True (kg) | Pred (kg) | Error |
|---|---|---|---|
| power drill | 0.7326 | 0.6974 | **4.8%** |
| lemon | 0.1045 | 0.0980 | **6.2%** |
| lego duplo (×5 variants) | 0.042–0.127 | 0.041–0.126 | **6.7–11.5%** |
| extra large clamp | 0.6829 | 0.7603 | 11.3% |
| knife | 0.0446 | 0.0501 | 12.1% |
| mini soccer ball | 0.1400 | 0.1576 | 12.5% |
| windex bottle | 0.9081 | 1.0312 | 13.6% |
| peach | 0.1260 | 0.1051 | 16.6% |
| plate | 0.5448 | 0.4398 | 19.3% |

### Worst Predictions (>200% error)

| Object | True (kg) | Pred (kg) | Error | Root cause |
|---|---|---|---|---|
| dice | 0.0049 | 0.0545 | 1007% | Anchor blend overshoots tiny object |
| toy airplane (×3) | 0.023–0.049 | 0.163–0.257 | 235–1003% | Anchor blend maps to wrong (large) anchor |
| pudding box | 0.0101 | 0.0718 | 611% | Sim density=50 vs real-world food density |
| colored wood blocks | 0.0182 | 0.1201 | 560% | Sim density=50 |
| adjustable wrench | 0.0462 | 0.2917 | 532% | Wrong packing + large bbox |
| large marker | 0.0355 | 0.2112 | 495% | Anchor blend overshoots |
| hammer | 0.3713 | 1.6150 | 335% | Solid-steel assumption, large bbox |

---

## Failure Mode Analysis

### 1. Dimension hallucination → FIXED by bbox injection

In v1, the model defaulted to 0.1×0.1×0.1m for everything:
- Apple: predicted 0.0024 kg (coin-sized) → FAILED (below bounds)
- Baseball: predicted rubber sheet (0.07×0.07×0.003m) → FAILED
- Pitcher base: predicted 26 kg (solid steel 0.15³) → FAILED

In v2, with actual bbox dimensions, all 10 previously-failed objects now produce predictions.
Apple now predicts 0.170 kg (true: 0.267 kg, 36% error vs previously FAILED).

### 2. Anchor blend: helps 19 objects, hurts 10

The geometric blend post-processes the raw LLM estimate when it diverges >3× from the nearest anchor:

**Where it helped:** Objects misidentified as solid metal (pitcher_base 0.34 kg, skillet_lid 0.24 kg, cracker_box, spoon). The blend pulled extreme solid-steel predictions down toward calibrated anchors.

**Where it hurt:** Very small objects where the raw prediction was already accurate:

| Object | Raw pred | After blend | True | Verdict |
|---|---|---|---|---|
| large marker | 0.041 (−15% err) | 0.211 (+495% err) | 0.036 | blend wrong anchor |
| toy airplane | 0.049 (0% err!) | 0.164 (+235% err) | 0.049 | nearest anchor is a big object |
| dice | 0.005 (−10% err) | 0.055 (+1007% err) | 0.005 | nearest anchor is much heavier |
| banana | 0.217 (+4% err) | 0.100 (−52% err) | 0.208 | embedding pulled to wrong anchor |

**Root cause:** SentenceTransformer embedding similarity does not always align with mass similarity. "toy airplane" embeds closest to heavier objects because the sentence representation does not encode scale. The blend then pulls a correct small prediction toward a wrong large anchor.

**Fix (future work):** Apply blend only when `raw_pred > nearest_anchor` (only correct overestimates, not underestimates). For underestimates, trust the raw prediction.

### 3. Sim density=50 boxes (structural limitation)

Four objects use density=50 kg/m³ in the sim (not physically realistic):

| Object | Sim density | Sim GT | Model pred | Real-world mass |
|---|---|---|---|---|
| pudding box | 50 | 0.010 kg | 0.072 kg | ~0.20 kg |
| cracker box | 50 | 0.123 kg | 0.018 kg | ~0.41 kg |
| gelatin box | 50 | 0.010 kg | 0.023 kg | ~0.10 kg |
| sugar box | 50 | 0.037 kg | 0.006 kg | ~0.51 kg |

The model predicts real-world masses; the ground truth is a simulation artifact. Use `--sim-density-hint` to partially address this.

### 4. Confidence always 1.0

Temperature=0.9 produces zero variance in structured JSON outputs with this 7B model. All 5 self-consistency samples return identical values for every object. The confidence metric (IQR-based) correctly reports 1.0 but is therefore uninformative. **Use N=1 — 5× faster with identical results.**

---

## Ablation Flags

```bash
# Default (full pipeline):
python -u eval_mass.py

# Ablation A — random anchors instead of NN retrieval:
python -u eval_mass.py --no-retrieval

# Ablation B — direct guess (no chain-of-thought):
python -u eval_mass.py --direct

# Ablation C — no anchor blend post-processing:
python -u eval_mass.py --no-anchor-blend

# Sim density hint (tells model about density=50 boxes):
python -u eval_mass.py --sim-density-hint

# With vision (uses saved RGB crops from single-obj eval):
python -u eval_mass.py --with-vision \
  --image-dir runs/20260522_215724_graspmas_single/imgs

# Quick smoke test:
python -u eval_mass.py --limit 10 --n-samples 1
```

---

## Running

```bash
# Full 78-object eval (~25 min on AMD MI300X):
conda run -n graspmas --no-capture-output \
    python -u eval_mass.py --out-dir runs/mass_eval_v2 2>/dev/null \
    | tee runs/mass_eval_v2.log

# Print GT table (no GPU needed):
python property_inference.py --verify-gt
```

**Outputs per run:**
- `mass_results.csv` — per-object: true, pred, abs_err, pct_err, confidence, elapsed_s
- `true_vs_pred.png` — scatter plot (colour = confidence, outlier labels annotated)
- `trace.jsonl` — one entry per object: raw LLM responses, parsed samples, all metadata

---

## Speed

| System | Objects | Time | Avg/object |
|---|---|---|---|
| Ours (Qwen2-VL-7B, local, N=5) | 78 | ~25 min | ~19s |
| Ours (Qwen2-VL-7B, local, N=1) | 78 | ~13 min | ~10s |

At N=1 (samples are identical anyway at T=0.9), eval time halves with no accuracy loss.

---

## Recommendations

| Priority | Change | Expected gain |
|---|---|---|
| High | Apply anchor blend only for overestimates, not underestimates | Fixes dice, toy airplane, large marker |
| High | Use N=1 (confidence metric broken at T=0.9) | Halves runtime, no accuracy cost |
| Medium | `--sim-density-hint` for box objects | Fixes density=50 outliers |
| Medium | `--with-vision` for size-ambiguous objects | Better packing factor estimation |
| Low | Higher temperature (1.2+) or prompt variation | Would make confidence metric meaningful |
