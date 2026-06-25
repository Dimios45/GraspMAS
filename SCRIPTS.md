# GraspMAS — Script Reference & Environment Guide

Covers every Python script in the project root, the `graspmas` conda environment,
and the exact commands needed to run each script on the AMD MI300X (ROCm) setup.

---

## Conda Environment: `graspmas`

### Activate

```bash
conda activate graspmas
```

### Key packages (pinned versions)

| Package | Version | Note |
|---|---|---|
| Python | 3.9.23 | conda-forge |
| PyTorch | 2.5.1+rocm6.2 | AMD ROCm build |
| numpy | **1.26.4** | must stay <2.0 — mplib & toppra are compiled against 1.x |
| mplib | **0.1.1** | mani-skill 3.0.1 requires exactly this version |
| toppra | 0.6.3 | must be built from source (`--no-binary toppra`) against numpy 1.26.x |
| mani-skill | 3.0.1 | ManiSkill simulator |
| sapien | 3.0.1 | SAPIEN physics engine |
| transformers | **4.48.0** | pinned; 4.49+ blocks `torch.load` on torch<2.6 (CVE-2025-32434) |
| torch | 2.5.1+rocm6.2 | |
| scipy | 1.13.1 | conda-forge build (pip manylinux wheels conflict with ROCm) |
| opencv-python | 4.8.0.76 | pip wheel; conda-forge `py-opencv` also works |
| sentence-transformers | 5.1.2 | for embedding-based anchor retrieval |
| qwen-vl-utils | 0.0.14 | Qwen2-VL image preprocessing |
| trimesh | 4.12.2 | YCB mesh volume computation |
| gymnasium | 1.1.1 | |
| imageio / imageio-ffmpeg | 2.37.2 / 0.6.0 | required by mani-skill for video recording |

### Critical version constraints

```
numpy==1.26.4   # mplib 0.1.1 and toppra Cython extensions are compiled against numpy 1.x
mplib==0.1.1    # mani-skill 3.0.1 hard-requires this; 0.2.x has a different API
transformers==4.48.0  # 4.49+ raises SafeTensorsError on torch 2.5
```

If numpy ever gets upgraded (e.g. by pip resolving a scipy/matplotlib dependency),
reinstall with:

```bash
pip install "mplib==0.1.1" "numpy==1.26.4"
pip install toppra --force-reinstall --no-binary toppra   # rebuild Cython against 1.26.4
```

### GPU routing (AMD MI300X, 8 GPUs)

The machine runs sglang inference servers on GPUs 1 and 4–7, which hold KFD-level
ROCm contexts. Use `CUDA_VISIBLE_DEVICES` to expose only a free GPU, and
`VLM_DEVICE` to tell `local_vlm.py` which device index to use within that view.

```bash
# Expose GPU 2 only; VLM loads on cuda:0 (= physical GPU 2)
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 python run_maniskill_demo.py

# Expose two GPUs (GPU 0 for sim, GPU 2 for VLM)
CUDA_VISIBLE_DEVICES=0,2 VLM_DEVICE=cuda:1 python run_maniskill_demo.py
```

ManiSkill always uses `sim_backend='cpu'` in this repo — the GPU is only needed
for Qwen2-VL inference.

### VLM model path

```
/mnt/data/mritunjoyh/models/Qwen2-VL-7B-Instruct
```

Loaded on first call to `vlm_chat()`. Takes ~30 s on first import; subsequent
calls reuse the singleton.

---

## Entry-Point Scripts

### `run_maniskill_demo.py` — End-to-end robot grasp demo

Runs the full GraspMAS → motion-planning pipeline on one ManiSkill episode:

1. Resets `PickClutterYCB-v1`, saves RGB/depth/third-view images.
2. Extracts ground-truth physics (mass, friction, required grip force).
3. Runs the GraspMAS Planner→Coder→Observer loop to get a 2D grasp rectangle.
4. Lifts the rectangle into a 6-DoF pose via camera intrinsics/extrinsics.
5. Executes the grasp with `PandaArmMotionPlanningSolver` and records a video.

**Command:**

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u run_maniskill_demo.py [--seed SEED]
```

| Flag | Default | Description |
|---|---|---|
| `--seed` | random | Fix the ManiSkill episode seed for reproducible runs |

Known-good seed: **17** (lemon on table, clean grasp).

**Outputs** (under `runs/<YYYYMMDD_HHMMSS>/`):

```
imgs/observation.png   — RGB / depth / third-view snapshot
imgs/grasp_*.png       — GraspMAS intermediate visualisations
video/0.mp4            — full episode recording
```

---

### `eval_mass.py` — YCB mass prediction evaluation (5-fold CV)

Evaluates `MassPredictor` (from `property_inference.py`) on all 78 YCB objects
using 5-fold cross-validation. Calls Qwen2-VL-7B via `local_vlm.vlm_chat`.

**Command:**

```bash
# Full run (~25 min on MI300X, all 78 objects, 5 samples each)
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u eval_mass.py --out-dir runs/mass_eval

# Quick smoke test (first 10 objects, 1 sample)
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u eval_mass.py --limit 10 --n-samples 1
```

| Flag | Default | Description |
|---|---|---|
| `--n-samples` | 5 | Self-consistency samples per object |
| `--temperature` | 0.9 | LLM sampling temperature |
| `--k-anchors` | 5 | Retrieval anchor count |
| `--no-retrieval` | off | Ablation A: random anchors instead of NN |
| `--direct` | off | Ablation B: skip chain-of-thought decomposition |
| `--no-anchor-blend` | off | Ablation C: skip geometric post-blend |
| `--sim-density-hint` | off | Tell model about sim's simplified densities |
| `--with-vision` | off | Attach object RGB crop to the prompt |
| `--image-dir` | `runs/20260522_.../imgs` | Root dir with `<obj_id>/input.png` |
| `--out-dir` | `runs/<ts>_mass_eval` | Output directory |
| `--limit` | all | Evaluate only first N objects |

**Outputs:**

```
mass_results.csv      — per-object predictions and errors
true_vs_pred.png      — scatter plot (true vs predicted mass, colour = confidence)
trace.jsonl           — raw LLM outputs per object
```

---

### `run_force_benchmark.py` — Force estimation baseline (E0)

For each YCB object in `PickSingleYCB-v1`: loads the object in ManiSkill, reads
ground-truth mass + friction from SAPIEN, then asks Qwen2-VL-7B to estimate
material, mass, friction, and grip force from a single RGB image.

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u run_force_benchmark.py
```

No flags. Iterates all available model IDs automatically.

**Outputs** (under `runs/<ts>_force_benchmark/`):

```
imgs/<model_id>.png   — one image per YCB object
results.json          — full per-object data
results.csv           — same (tabular)
summary.txt           — human-readable table + aggregate MAE/MAPE
```

---

### `run_benchmark_e1.py` — Force estimation experiment 1 (material table)

Improves on E0 by: (1) better VLM prompt with chain-of-thought + material anchors,
(2) table-based friction (VLM classifies material, we look up realistic μ),
(3) brittleness-based F_max ceiling.

Reports three variants side-by-side:

| Variant | Mass source | Friction source |
|---|---|---|
| E1a | VLM direct estimate | Material lookup table |
| E1b | Ground-truth (ceiling) | Material lookup table |
| E1c | VLM dims × table density | Material lookup table |

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u run_benchmark_e1.py
```

**Outputs** (under `runs/<ts>_benchmark_e1/`):

```
imgs/<model_id>.png   — per-object images
results_e1.json / results_e1.csv
summary_e1.txt        — per-object table + E1a/E1b/E1c metrics + per-material breakdown
scatter_e1.png        — 3-panel scatter: GT vs predicted force for each variant
```

---

### `run_graspmas_eval.py` — GraspMAS benchmark on PickSingleYCB

Runs GraspMAS on all 74 YCB objects in `PickSingleYCB-v1`, records whether the
Observer returns VALID, and produces a per-object summary CSV.

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u run_graspmas_eval.py
```

**Results:** 66.2% VALID rate (49/74), 75.7% detection rate. See `docs/GRASPMAS_QWEN.md`.

**Outputs** (under `runs/<ts>_graspmas_single/`):
```
imgs/<obj_id>/input.png, query_image.png, grasp_pose_visualization.png
results.json / results.csv / summary.txt
```

---

### `run_graspmas_cluttered_eval.py` — GraspMAS benchmark on PickClutterYCB

Runs GraspMAS on 100 random seeds of `PickClutterYCB-v1` (multi-object scenes).

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u run_graspmas_cluttered_eval.py [--n-seeds 100] [--start-seed 0]
```

**Results:** 35.0% VALID rate (35/100). See `docs/GRASPMAS_QWEN.md`.

---

### `run_ocidvlg_eval.py` — OCID-VLG real-world benchmark

Evaluates on the OCID-VLG dataset (89k image-text-grasp tuples, 30 categories).
Dataset path: `/mnt/data/mritunjoyh/datasets/ocid-vlg/`

```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 \
    python -u run_ocidvlg_eval.py [--n-samples 100] [--split test] [--seed 42]
```

| Flag | Default | Description |
|---|---|---|
| `--n-samples` | 100 | Total samples (stratified across categories) |
| `--split` | `test` | Dataset split |
| `--seed` | 42 | Sampling seed (deterministic) |
| `--easy` | off | Restrict to 8 best categories + simple queries |

**Results:** 13% success (full), 34% (easy subset). See `docs/QWEN_OCIDVLG.md`.

---

### `run_mustard_demo.py` — Single-object directional demo

Runs 4 directional grasp prompts (top / right / left / bottom) on any YCB object and saves one video per direction. Loads VLMs once; all 4 directions share the process.

**Command:**

```bash
CUDA_VISIBLE_DEVICES=3 VLM_DEVICE=cuda:0 PYOPENGL_PLATFORM=egl \
    python run_mustard_demo.py --object-id 013_apple --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--object-id` | `006_mustard_bottle` | Any YCB object ID, e.g. `077_rubiks_cube` |
| `--seed` | `42` | ManiSkill episode seed |

**Outputs** (under `object_demo/<object_name>_<timestamp>/`):

```
imgs_top/observation.png      — RGB / depth / third-view per direction
imgs_top/grasp_*.png          — GraspMAS intermediate visualisations
videos/top/0.mp4              — episode video (one per direction)
videos/right/0.mp4
videos/left/0.mp4
videos/bottom/0.mp4
```

Browse via HTTP: `python -m http.server 8787 --directory object_demo` then open `localhost:8787`.

---

### `run_all_objects_demo.py` — Full 49-object directional batch

Runs `run_mustard_demo` logic for all 49 VALID YCB objects in a single process. VLMs are loaded once. Supports crash recovery (resume from partial run) and per-direction video-skip logic.

**Command:**

```bash
# Fresh run
CUDA_VISIBLE_DEVICES=3 VLM_DEVICE=cuda:0 PYOPENGL_PLATFORM=egl \
    nohup python run_all_objects_demo.py --seed 42 \
    > logs/batch.log 2>&1 &

# Resume a crashed run
CUDA_VISIBLE_DEVICES=3 VLM_DEVICE=cuda:0 PYOPENGL_PLATFORM=egl \
    python run_all_objects_demo.py --resume object_demo/all_20260624_215600
```

| Flag | Default | Description |
|---|---|---|
| `--seed` | `42` | Shared ManiSkill episode seed |
| `--resume` | — | Path to an existing batch dir to resume from |

**Crash safety:** each direction is wrapped in `try/except`; failures are logged and the batch continues. Results are saved incrementally to `results.json` after every direction so a crash never loses completed work.

**Outputs** (under `object_demo/all_<timestamp>/`):

```
results.json                  — all direction results (success, height, angle, mass, timing)
<object>/videos/top/0.mp4     — one video per object per direction
<object>/imgs_top/            — GraspMAS visualisations
```

**Results (seed=42, 49 objects):**

| Direction | Approach | Success |
|---|---|---|
| top    | `[0, 0, −1]`     | 24/39 (62%) |
| right  | `[0, −1, −1]/√2` | 13/39 (33%) |
| left   | `[0, +1, −1]/√2` | 6/39  (15%) |
| bottom | `[0,  0, −1]`    | 24/40 (60%) |
| **Total** | | **67/157 (42.7%)** |

Perfect 4/4 objects: apple, lemon, orange, softball, tennis ball, cups.

Check progress during a run:
```bash
tail -f logs/batch.log
grep "is_grasped" logs/batch.log | tail -20
```

---

### `run_clamp_parts_demo.py` — Semantic part grasping

Runs 3 semantic-part prompts on `050_medium_clamp`: loop, right handle, left handle. Uses `find_part()` (VLpart) for sub-object localisation.

**Command:**

```bash
CUDA_VISIBLE_DEVICES=3 VLM_DEVICE=cuda:0 PYOPENGL_PLATFORM=egl \
    python run_clamp_parts_demo.py
```

No flags. Object and seed are fixed (`050_medium_clamp`, seed=42).

**Outputs** (under `object_demo/clamp_parts_<timestamp>/`):

```
videos/loop/0.mp4
videos/right_handle/0.mp4
videos/left_handle/0.mp4
imgs_loop/ imgs_right_handle/ imgs_left_handle/
```

---

### `run_gemini_grasp.py` — Gemini Direct baseline

Single-turn Gemini API call for grasp detection — no multi-agent loop, no local GPU.
Requires `gemini.key` in the project root.

```bash
python -u run_gemini_grasp.py \
    --image-path path/to/image.png \
    --query "Grasp the banana"
```

---

### `main_simple.py` — Single-image GraspMAS inference

Minimal CLI wrapper: runs GraspMAS on one image and prints the resulting grasp pose.

```bash
python main_simple.py \
    --api-file api.key \
    --query "Grasp the banana" \
    --image-path path/to/image.png \
    --max-round 5
```

| Flag | Default | Description |
|---|---|---|
| `--api-file` | `api.key` | Path to the API key file |
| `--query` | required | Natural-language grasp instruction |
| `--image-path` | required | Input image path |
| `--max-round` | 4 | Max Planner→Coder→Observer iterations |

---

### `property_inference.py` — GT table verification (CLI mode)

Used primarily as a library (`MassPredictor`, `build_gt_table`, etc.).
The `--verify-gt` flag prints the full ground-truth mass table and verifies
the trimesh API without needing a running ManiSkill environment.

```bash
# No GPU needed — reads sim asset files directly
python property_inference.py --verify-gt
```

Prints all 78 YCB objects with true mass (kg) and bounding-box dimensions.

---

## Library Modules

These are imported by the scripts above and are not meant to be run directly.

| Module | Purpose |
|---|---|
| `local_vlm.py` | Singleton loader for Qwen2-VL-7B-Instruct. Exposes `vlm_chat(messages, ...)`. Device set via `VLM_DEVICE` env var (default `cuda:1`). |
| `grasp_force.py` | Ground-truth physics extraction from a live SAPIEN env (`get_object_physics`), force computation (`compute_required_force`), contact force readout (`get_contact_forces`), and pretty-printing (`print_force_report`). |
| `property_inference.py` | `MassPredictor` class: embedding-based anchor retrieval (SentenceTransformer), chain-of-thought prompt construction, self-consistency (median of N samples), anchor-blend post-processing. Also provides `build_gt_table` and `build_bbox_table`. |
| `material_table.py` | Lookup table of material properties (μ_s friction, density, max contact pressure) for YCB object categories. Used by `run_benchmark_e1.py`. |
| `image_patch.py` | `ImagePatch` class: wraps an RGB array with cropping, VLM querying, and grasp-detection tools. Used by the GraspMAS agents. |
| `utils.py` | Grasp evaluation utilities: `rotated_rect_to_polygon`, `calculate_iou_and_angle`, `eval_grasp`. |
| `agents/` | GraspMAS multi-agent system: `Planner`, `Coder`, `Observer`, `GraspMAS` orchestrator, and `OpenAILLM` (points at local Qwen2-VL). |

---

## Known Broken

| Script | Issue |
|---|---|
| `main_batch.py` | Imports `visualize_grasp_pose` from `utils` — function never existed. Broken since the first commit. Not actively used. |

---

## Output Directory Structure

Every run writes to `runs/<timestamp>_<run-type>/`:

```
runs/
  20260525_130000/               ← run_maniskill_demo.py
    imgs/
      observation.png
      grasp_*.png
    video/
      0.mp4
  20260525_131000_mass_eval/     ← eval_mass.py
    mass_results.csv
    true_vs_pred.png
    trace.jsonl
  20260525_132000_force_benchmark/   ← run_force_benchmark.py
    imgs/
    results.json
    results.csv
    summary.txt
  20260525_133000_benchmark_e1/  ← run_benchmark_e1.py
    imgs/
    results_e1.json
    results_e1.csv
    summary_e1.txt
    scatter_e1.png
```
