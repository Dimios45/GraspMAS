<div align="center"><h1> GraspMAS: Zero-Shot Language-driven Grasp Detection with Multi-Agent System<br>
</h1>
<p align="center">
    <a href="https://scholar.google.com/citations?user=F5Fr2ysAAAAJ&hl=vi" style="text-decoration: none;">Quang Nguyen</a> •
    <a href="https://scholar.google.com/citations?user=t6RXOWgAAAAJ&hl=vi" style="text-decoration: none;">Tri Le</a> •
    <a href="https://scholar.google.com/citations?user=T_LryjgAAAAJ&hl=en" style="text-decoration: none;">Huy Nguyen</a> •
    <a href="https://sites.google.com/tdtu.edu.vn/vongocthieu" style="text-decoration: none;">Thieu Vo</a> •
    <a href="https://scholar.google.it/citations?user=KUqlbGUAAAAJ&hl=en" style="text-decoration: none;">Tung Ta</a> •
    <a href="https://scholar.google.com/citations?user=unbPvWAAAAAJ&hl=zh-CN" style="text-decoration: none;">Baoru Huang</a> •
    <a href="https://scholar.google.com/citations?hl=th&user=qyExc4QAAAAJ&view_op=list_works" style="text-decoration: none;">Minh Vu</a> •
    <a href="https://www.csc.liv.ac.uk/~anguyen/" style="text-decoration: none;">Anh Nguyen</a>
</p>
<h1><sub><sup><a href="https://www.iros25.org/">IROS 2025</a></sup></sub></h1>

[![Website](https://img.shields.io/badge/Website-Demo-fedcba?style=flat-square)](https://zquang2202.github.io/GraspMAS/) 
[![arXiv](https://img.shields.io/badge/arXiv-2403.07487-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2506.18448)

</div>

# Introduction
![image](static/method9.jpg)
In this paper, we introduce GraspMAS, a new multi-agent system framework for language-driven grasp detection. GraspMAS is designed to reason through ambiguities and improve decision-making in real-world scenarios.

![image](static/compare_fig.jpg)

Our method consistently produces more plausible grasp poses than existing methods.
# Installation
Follow these steps to install the GraspMAS framework:

1. **Clone recursively:**
    ```bash
    git clone --recurse-submodules https://github.com/Fsoft-AIC/GraspMAS.git
    cd GraspMAS
    ```

2. **OpenAI key:** To run the GraspMAS framework, you will need an OpenAI key. This can be done by signing up for an account and then creating a key in account/api-keys. Create a file api.key in the root of this project and store the key in it.
    ```
    echo YOUR_OPENAI_API_KEY_HERE > api.key
    ```

3. **Prepare environment:**
   ```bash
    conda create -n graspmas python=3.9 -y
    conda activate graspmas
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    cd detectron2
    pip install -e .
    cd ..
   ```

4. **Download pretrained model:**

    ```bash 
    bash download.sh
    ```
# Quickstart
- You can start checking the notebook ```simple_demo.ipynb``` for simple demo inference. This notebook includes details instructions and executing queries with visualization. You can run either the complete closed-loop pipeline or the open-loop mode with Coder.
- If you want to run inference on a single image, use the following:
```bash
python main_simple.py \
    --api-file "api.key" \
    --max-round 5 \
    --query "Grasp the knife at its handle" \
    --image-path PATH-TO-INPUT-IMAGE \
    --save-folder PATH-TO-SAVE-FOLDER
```

# Configuration
If you want to customize tools or model hyperparameters and configurations, please refer to **`image_patch.py`**. We have only developed sufficient tools for language-driven grasp detection. The GraspMAS framework heavily depends on the effectiveness of pretrained models as tools, so results may be biased. Feel free to add or remove any pretrained models related to image or video processing, including any up-to-date models. Note that some models, such as BLIP or VLM, may require significant GPU memory.

# Maniskill Demo with GraspMAS
<p align="center">
    <img src="static/robot_exp.jpg" alt="image" />
</p>

We provide the notebook demo **`Maniskill_demo.ipynb`** and the script **`run_maniskill_demo.py`** for simulating language-driven grasp detection on the ManiSkill simulator. The simulation runs in a tabletop environment using a Panda robot arm equipped with a wrist camera.

**Run the demo:**
```bash
CUDA_VISIBLE_DEVICES=2 VLM_DEVICE=cuda:0 python -u run_maniskill_demo.py --seed 17
```

## Demo Pipeline

```
Seed → PickClutterYCB-v1 (SAPIEN/PhysX, CPU sim)
            │
            ▼  1. Observation
   RGB + depth + third-person view saved to imgs/
            │
            ▼  2. Ground-truth physics (grasp_force.py)
   Read from SAPIEN: mass, friction → F_required = safety × m × g / (2μ)
            │
            ▼  3. GraspMAS multi-agent loop (Qwen2-VL-7B)
   ┌─────────────────────────────────────────────────┐
   │  PLANNER  →  step-by-step natural language plan │
   │  CODER    →  Python code using ImagePatch API   │
   │                 find()       ← GroundingDINO + SAM   │
   │                 grasp_detection() ← GraspNet (RAGT) │
   │                 find_part()  ← VLpart            │
   │  OBSERVER →  checks grasp on RGB overlay        │
   │              VALID: done  |  INVALID: retry      │
   │              (up to max_round=5 iterations)      │
   └─────────────────────────────────────────────────┘
            │  2D grasp: (quality, cx, cy, w, h, angle)
            ▼  4. 2D → 6DoF projection
   depth[cy, cx] → 3D camera coords
   camera intrinsics (K) + extrinsics → world coords
   jaw axis angle → gripper orientation
            │
            ▼  5. Motion planning + execution (mplib + toppra)
   PandaArmMotionPlanningSolver:
     (a) Reach  — approach 5 cm above grasp
     (b) Grasp  — move in, close gripper
     (c) Lift   — move to goal 40 cm up
            │
            ▼  6. Contact force validation
   get_contact_forces() — compare actual vs F_required
            │
            ▼  Outputs
   runs/<timestamp>/imgs/   — observations + grasp overlays
   runs/<timestamp>/video/  — full episode recording
```

**What can fail:**
- Object not detected by GroundingDINO → script exits with "No grasp detected"
- Observer rejects all rounds → falls through with last best grasp
- Bad depth at grasp centre (background pixel) → warning printed, planner still attempts
- Motion planner finds no collision-free path → SAPIEN error, no lift

# Note
This is a research project, so the code may not be optimized, regularly updated, or actively maintained.

# Mass Property Inference (PoC Extension)

This fork adds a zero-shot YCB object mass predictor (`property_inference.py`) and
a 5-fold cross-validation evaluation harness (`eval_mass.py`).

## ManiSkill mass-readout API

The ground-truth mass for each YCB object is computed directly from the
simulation assets — no live ManiSkill run required:

```python
import json, trimesh
from pathlib import Path

ASSET_DIR = Path("~/.maniskill/data/assets/mani_skill2_ycb").expanduser()

with open(ASSET_DIR / "info_pick_v0.json") as f:
    info = json.load(f)                          # density + scale per object

mesh  = trimesh.load(str(ASSET_DIR / "models" / obj_id / "collision.ply"))
if not mesh.is_watertight:
    mesh = mesh.convex_hull

scale   = info[obj_id]["scales"][0]
density = info[obj_id]["density"]                # kg/m³ (sim value)
mass_kg = density * mesh.volume * scale**3       # exact PhysX value
```

This matches what SAPIEN/PhysX computes at actor-build time. Verified: all 78
YCB objects return nonzero masses in the range 0.005–1.37 kg.

## Results (Qwen2-VL-7B, 78 YCB objects, 5-fold CV)

| Metric | Value |
|---|---|
| Detection rate | 100% (78/78) |
| MAE_kg | 0.1803 kg |
| MAPE | 126.4% |
| Median error | 70.0% |
| <25% error | 20.5% (16/78) |
| <50% error | 32.1% (25/78) |

Run: `runs/mass_eval_v2/` — see `docs/MASS_EVAL.md` for full analysis.

## Running the mass eval

```bash
# print ground-truth table and verify the API (no GPU needed)
python property_inference.py --verify-gt

# full 5-fold eval (~25 min on AMD MI300X)
conda run -n graspmas --no-capture-output \
    python -u eval_mass.py --out-dir runs/mass_eval_v2 2>/dev/null \
    | tee runs/mass_eval_v2.log

# ablations
conda run -n graspmas python -u eval_mass.py --no-retrieval    # random anchors
conda run -n graspmas python -u eval_mass.py --direct          # no chain-of-thought
conda run -n graspmas python -u eval_mass.py --no-anchor-blend # no post-blend

# quick smoke test (first 10 objects, 1 sample)
python eval_mass.py --limit 10 --n-samples 1
```

Outputs per run: `mass_results.csv`, `true_vs_pred.png`, `trace.jsonl`.

---

# Local Implementation & Results (AMD ROCm)

This fork runs GraspMAS entirely locally on AMD MI300X using Qwen2-VL-7B-Instruct
(ROCm 6.2.4, no cloud API required) and adds physics-aware force estimation.

### Documentation

| Doc | Contents |
|---|---|
| [SCRIPTS.md](SCRIPTS.md) | Every script: purpose, CLI flags, outputs, conda env setup |
| [docs/GRASPMAS_QWEN.md](docs/GRASPMAS_QWEN.md) | System architecture, benchmark results vs paper |
| [docs/GRASP_EXAMPLES.md](docs/GRASP_EXAMPLES.md) | Grasp pose format, worked examples, visualisations |
| [docs/MASS_EVAL.md](docs/MASS_EVAL.md) | YCB mass prediction: pipeline, results, failure analysis |
| [docs/QWEN_OCIDVLG.md](docs/QWEN_OCIDVLG.md) | OCID-VLG evaluation: 13% (full) / 34% (easy) success rate |

### Benchmark Summary

| Benchmark | Ours (Qwen2-VL-7B, local) | Paper (GPT-4o) |
|---|---|---|
| PickSingleYCB VALID rate | **66.2%** (49/74) | 82.0% |
| PickClutterYCB VALID rate | **35.0%** (35/100) | 72.0% |
| OCID-VLG success (full) | **13.0%** (13/100) | 62.0% |
| OCID-VLG success (easy) | **34.0%** (34/100) | 62.0% |
| YCB mass MAE | **0.180 kg** (78/78) | — |

---

# Citation
If you find our work useful for your research, please cite:
```
@inproceedings{nguyen2025graspmas,
      title = {GraspMAS: Zero-Shot Language-driven Grasp Detection with Multi-Agent System},
      author = {Nguyen, Quang and Le, Tri and Nguyen, Huy and Vo, Thieu and Ta, Tung D and Huang, Baoru and Vu, Minh N and Nguyen, Anh},
      booktitle = IROS,
      year      = {2025}
  }
```
# Acknowledgement
We thank the valuable work of [ViperGPT](https://github.com/cvlab-columbia/viper), [ViperDuality](https://github.com/duality-robotics/viper/tree/main) that inspired and enabled this research.
