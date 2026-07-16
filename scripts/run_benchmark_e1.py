"""
Experiment 1 — Material Classification + Lookup Table
======================================================
Improves on E0 (raw VLM guessing) by:
  - Better VLM prompt: chain-of-thought with material guide + density anchors
  - Table-based friction: VLM classifies material, we look up realistic mu_s
  - Brittleness bounds: F_max ceiling per material to prevent crush damage

Three variants reported side-by-side:
  E1a  VLM mass  + table friction               (tests friction fix)
  E1b  GT mass   + table friction               (ceiling: perfect mass + real friction)
  E1c  VLM dims  × table density → mass         (tests explicit geometric reasoning)

Metrics vs E0 baseline (runs/20260522_191900_force_benchmark/):
  Mass MAE/MAPE, Force MAE/MAPE vs sim-GT, material accuracy,
  slip rate (F < F_min_real), damage rate (F > F_max), safe rate.

Note on friction:
  ManiSkill uses mu=0.3 for ALL YCB objects (simulator simplification).
  F_gt_sim is therefore mass-only dependent.
  Table friction reflects real-world physics; E1b/E1c are the "real" force targets.
"""

import os, sys, re, json, csv, time, warnings
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, base64

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mani_skill.envs
import gymnasium as gym

from local_vlm import vlm_chat
from material_table import MATERIAL_TABLE, lookup, get_gt_material

# ── constants ────────────────────────────────────────────────────────────────
GRAVITY               = 9.81
SAFETY_FACTOR         = 2.0
SIM_MU                = 0.3          # ManiSkill fixed mu for all YCB
PANDA_CONTACT_AREA_M2 = 0.02 * 0.01  # 2 cm × 1 cm finger pad

# ── VLM prompt ───────────────────────────────────────────────────────────────
VLM_PROMPT = """\
You are a robotics assistant. Analyse the image and fill in the form below.

Object in image: {name}

MATERIAL CATEGORIES (pick exactly one):
  metal        → steel/iron: cans, wrench, clamps, scissors, fork, knife, spoon, padlock, screwdriver shaft, hammer head
  hard_plastic → ABS/PC/PET: lego bricks, dice, rubiks cube, toy airplane, golf ball, large marker, cups (most), power drill body
  soft_plastic → HDPE/PP: squeeze bottles, mustard bottle, bleach cleanser, pitcher base
  cardboard    → hollow flat-faced boxes: cracker box, sugar box, pudding box, gelatin box
  rubber       → solid rubber balls: soccer ball, softball, baseball, tennis ball, racquetball; silicone cups
  foam         → very light & compressible: foam brick, kitchen sponge
  wood         → wood block, colored wood blocks, nine hole peg board, hammer handle
  ceramic      → hard pottery: mug, ceramic bowl
  glass        → marbles, glass jars
  fruit        → soft produce: banana, apple, orange, lemon, peach, pear, plum, strawberry
  composite    → mixed metal+plastic: power drill, hammer, screwdriver (whole), spatula
  other        → anything else

DENSITY TABLE (g per cc):
  metal=7.8  hard_plastic=1.2  soft_plastic=0.9  cardboard=0.15  rubber=1.1
  foam=0.08  wood=0.7  ceramic=2.0  glass=2.5  fruit=0.9  composite=2.0

MASS ESTIMATION:
  1. Estimate the bounding-box size in cm (length × width × height).
  2. mass_g = density × length_cm × width_cm × height_cm
  Cross-check: banana≈200g, soup can≈375g, mug≈310g, scissors≈80g,
               hammer≈364g, tennis ball≈161g, lego brick≈40g,
               foam brick≈200g, marble set≈143g, rubiks cube≈196g

FRICTION TABLE:
  metal=0.30  hard_plastic=0.30  soft_plastic=0.25  cardboard=0.45
  rubber=0.75  foam=0.40  wood=0.40  ceramic=0.40  glass=0.20
  fruit=0.50  composite=0.35  other=0.30

GRIP FORCE:
  grip_force_N = {safety} × mass_kg × {g} / (2 × friction)

Fill EXACTLY this template — one value per line, no extra text:
material: <category>
brittleness: <brittle|soft|ductile>
length_cm: <number>
width_cm: <number>
height_cm: <number>
mass_g: <integer>
friction: <number>
grip_force_N: <number>
"""

# ── physics helpers ──────────────────────────────────────────────────────────

def get_gt_physics(env_unwrapped):
    """Read ground-truth mass and sim friction from SAPIEN."""
    obj   = env_unwrapped.obj
    mass  = float(obj.get_mass())
    mu_s  = SIM_MU
    for c in obj._objs[0].get_components():
        if 'Physx' in type(c).__name__:
            shapes = c.get_collision_shapes()
            if shapes:
                mu_s = float(shapes[0].get_physical_material().static_friction)
    return dict(
        mass_kg=mass,
        mass_g=round(mass * 1000, 1),
        sim_mu=mu_s,
        F_gt_sim=round(SAFETY_FACTOR * mass * GRAVITY / (2 * mu_s), 3),
    )


def force_from(mass_kg: float, mu: float) -> float:
    return SAFETY_FACTOR * mass_kg * GRAVITY / (2 * mu)


def f_max(mat_props: dict) -> float:
    return mat_props['max_pressure_kPa'] * 1000 * PANDA_CONTACT_AREA_M2


# ── image encoding ───────────────────────────────────────────────────────────

def encode_rgb(rgb_np):
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


# ── VLM query ─────────────────────────────────────────────────────────────────

def query_vlm(rgb_np, name):
    b64     = encode_rgb(rgb_np)
    prompt  = VLM_PROMPT.format(name=name, safety=SAFETY_FACTOR, g=round(GRAVITY, 2))
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text",      "text": prompt}
    ]}]
    return vlm_chat(messages, max_new_tokens=150)


def parse_vlm(raw: str) -> dict:
    result = {}
    for line in raw.splitlines():
        m = re.match(r'(\w+(?:_\w+)*):\s*(.+)', line.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key in ('material', 'brittleness'):
            result[key] = val.lower()
        else:
            try:
                result[key] = float(re.search(r'[\d.]+', val).group())
            except (AttributeError, ValueError):
                result[key] = None
    return result


def model_id_to_name(model_id: str) -> str:
    parts = model_id.split('_', 1)
    return (parts[1] if len(parts) > 1 else model_id).replace('_', ' ')


# ── summary helpers ───────────────────────────────────────────────────────────

def compute_metrics(results, force_key, gt_force_key='F_gt_sim'):
    """MAE/MAPE for force, slip rate, damage rate, safe rate."""
    valid = [r for r in results if r.get(force_key) is not None]
    if not valid:
        return {}
    forces  = np.array([r[force_key]      for r in valid])
    gt_sim  = np.array([r[gt_force_key]   for r in valid])
    gt_real = np.array([r['F_gt_real']    for r in valid])
    fmax    = np.array([r['F_max']        for r in valid])

    mae_vs_sim  = float(np.mean(np.abs(forces - gt_sim)))
    mape_vs_sim = float(np.mean(np.abs(forces - gt_sim) / (gt_sim + 1e-9) * 100))
    slip        = float(np.mean(forces < gt_real) * 100)
    damage      = float(np.mean(forces > fmax)    * 100)
    safe        = float(np.mean((forces >= gt_real) & (forces <= fmax)) * 100)
    return dict(
        n=len(valid),
        mae_vs_sim=round(mae_vs_sim, 3),
        mape_vs_sim=round(mape_vs_sim, 1),
        slip_pct=round(slip, 1),
        damage_pct=round(damage, 1),
        safe_pct=round(safe, 1),
    )


def mass_metrics(results):
    valid = [r for r in results if r.get('vlm_mass_g') is not None]
    if not valid:
        return {}
    err  = [abs(r['vlm_mass_g'] - r['gt_mass_g']) for r in valid]
    pct  = [abs(r['vlm_mass_g'] - r['gt_mass_g']) / (r['gt_mass_g'] + 1e-9) * 100 for r in valid]
    return dict(mae=round(np.mean(err), 1), mape=round(np.mean(pct), 1))


def mass_metrics_dims(results):
    valid = [r for r in results
             if r.get('vlm_length_cm') and r.get('vlm_width_cm') and r.get('vlm_height_cm')]
    if not valid:
        return {}
    err  = [abs(r['E1c_mass_g'] - r['gt_mass_g']) for r in valid]
    pct  = [abs(r['E1c_mass_g'] - r['gt_mass_g']) / (r['gt_mass_g'] + 1e-9) * 100 for r in valid]
    return dict(mae=round(np.mean(err), 1), mape=round(np.mean(pct), 1))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"{run_id}_benchmark_e1")
    img_dir = os.path.join(out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    env = gym.make(
        'PickSingleYCB-v1',
        obs_mode='rgbd',
        control_mode='pd_joint_pos',
        render_mode='rgb_array',
        sensor_configs=dict(shader_pack='default', width=384, height=384),
        sim_backend='cpu',
    )

    all_model_ids = list(env.unwrapped.all_model_ids)
    print(f"Experiment 1 — {len(all_model_ids)} YCB objects")
    print(f"Output: {out_dir}/\n")

    results = []

    for i, model_id in enumerate(all_model_ids):
        name = model_id_to_name(model_id)
        print(f"[{i+1:2d}/{len(all_model_ids)}] {model_id}")

        # load object
        env.unwrapped.all_model_ids = np.array([model_id])
        obs, _ = env.reset(options=dict(reconfigure=True))
        rgb = obs['sensor_data']['base_camera']['rgb'].cpu().squeeze().numpy()

        # ground truth
        gt      = get_gt_physics(env.unwrapped)
        gt_mat  = get_gt_material(model_id)
        gt_props = lookup(gt_mat)
        F_gt_real = round(force_from(gt['mass_kg'], gt_props['mu_s']), 3)
        F_max_val = round(f_max(gt_props), 3)

        # save image
        plt.imsave(os.path.join(img_dir, f"{model_id}.png"), rgb)

        # VLM query
        t0  = time.time()
        raw = query_vlm(rgb, name)
        vlm = parse_vlm(raw)
        t1  = time.time()

        # material lookup from VLM prediction
        vlm_mat   = vlm.get('material', 'other')
        vlm_props = lookup(vlm_mat)

        # ── E1a: VLM mass + table friction ───────────────────────────────────
        e1a_mass_g = vlm.get('mass_g')
        e1a_force  = None
        if e1a_mass_g:
            e1a_force = round(force_from(e1a_mass_g / 1000, vlm_props['mu_s']), 3)

        # ── E1b: GT mass + table friction (ceiling) ───────────────────────────
        e1b_force = round(force_from(gt['mass_kg'], gt_props['mu_s']), 3)

        # ── E1c: VLM dims × table density → mass ─────────────────────────────
        l, w, h = vlm.get('length_cm'), vlm.get('width_cm'), vlm.get('height_cm')
        e1c_mass_g = None
        e1c_force  = None
        if l and w and h:
            vol_cc     = l * w * h
            dens_g_cc  = vlm_props['density_kg_m3'] / 1000
            e1c_mass_g = round(vol_cc * dens_g_cc, 1)
            e1c_force  = round(force_from(e1c_mass_g / 1000, vlm_props['mu_s']), 3)

        # ── material accuracy ─────────────────────────────────────────────────
        mat_correct = (vlm_mat == gt_mat)

        row = {
            'model_id':       model_id,
            'name':           name,
            # ground truth
            'gt_mass_g':      gt['mass_g'],
            'gt_mat':         gt_mat,
            'F_gt_sim':       gt['F_gt_sim'],
            'F_gt_real':      F_gt_real,
            'F_max':          F_max_val,
            # VLM outputs
            'vlm_material':   vlm_mat,
            'vlm_brittleness':vlm.get('brittleness'),
            'vlm_length_cm':  l,
            'vlm_width_cm':   w,
            'vlm_height_cm':  h,
            'vlm_mass_g':     e1a_mass_g,
            'vlm_friction':   vlm.get('friction'),
            'vlm_raw_force':  vlm.get('grip_force_N'),
            'vlm_raw':        raw.strip(),
            # derived
            'table_mu_vlm':   vlm_props['mu_s'],
            'table_mu_gt':    gt_props['mu_s'],
            'mat_correct':    mat_correct,
            # forces per variant
            'E1a_force':      e1a_force,
            'E1b_force':      e1b_force,
            'E1c_mass_g':     e1c_mass_g,
            'E1c_force':      e1c_force,
            'elapsed_s':      round(t1 - t0, 1),
        }
        results.append(row)

        mat_tag = "✓" if mat_correct else f"✗({gt_mat})"
        print(f"  GT:  {gt['mass_g']:.0f}g  mat={gt_mat}  F_sim={gt['F_gt_sim']:.2f}N  F_real={F_gt_real:.2f}N  F_max={F_max_val:.1f}N")
        print(f"  VLM: mat={vlm_mat} {mat_tag}  mass={e1a_mass_g or '?'}g")
        print(f"  E1a={e1a_force or '?'}N  E1b={e1b_force:.2f}N  E1c={e1c_force or '?'}N  [{t1-t0:.1f}s]")

    env.close()

    # ── save raw results ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "results_e1.json"), 'w') as f:
        json.dump(results, f, indent=2)

    csv_keys = [k for k in results[0] if k != 'vlm_raw']
    with open(os.path.join(out_dir, "results_e1.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in csv_keys})

    # ── compute metrics ───────────────────────────────────────────────────────
    mat_acc    = round(np.mean([r['mat_correct'] for r in results]) * 100, 1)
    mm_e1a     = mass_metrics(results)
    mm_e1c     = mass_metrics_dims(results)
    m_e1a      = compute_metrics(results, 'E1a_force')
    m_e1b      = compute_metrics(results, 'E1b_force')
    m_e1c      = compute_metrics(results, 'E1c_force')

    # ── per-material breakdown ────────────────────────────────────────────────
    from collections import defaultdict
    by_mat = defaultdict(list)
    for r in results:
        by_mat[r['gt_mat']].append(r)

    mat_breakdown = []
    for mat, rows in sorted(by_mat.items()):
        mm   = mass_metrics(rows)
        mf   = compute_metrics(rows, 'E1a_force')
        acc  = round(np.mean([r['mat_correct'] for r in rows]) * 100, 1)
        mat_breakdown.append(
            f"  {mat:<15} n={len(rows):2d}  mat_acc={acc:5.1f}%  "
            f"mass_mape={mm.get('mape','?'):6}%  "
            f"E1a_mape={mf.get('mape_vs_sim','?'):7}%  "
            f"slip={mf.get('slip_pct','?'):5}%  dmg={mf.get('damage_pct','?'):5}%"
        )

    # ── write summary ─────────────────────────────────────────────────────────
    sep = "─" * 100
    lines = [
        "EXPERIMENT 1 — Material Classification + Lookup Table",
        f"Run: {run_id}",
        sep,
        f"{'Object':<30} {'GT_mat':<14} {'VLM_mat':<14} {'ok':<3} "
        f"{'GT_g':>6} {'VLM_g':>6} {'E1a_N':>7} {'E1b_N':>7} {'E1c_N':>7} "
        f"{'Fsim_N':>7} {'Freal_N':>7} {'Fmax_N':>7}",
        sep,
    ]
    for r in results:
        ok = "✓" if r['mat_correct'] else "✗"
        lines.append(
            f"{r['name']:<30} {r['gt_mat']:<14} {r['vlm_material']:<14} {ok:<3}"
            f"{r['gt_mass_g']:>6.0f} {str(r['vlm_mass_g'] or '?'):>6} "
            f"{str(r['E1a_force'] or '?'):>7} {r['E1b_force']:>7.2f} "
            f"{str(r['E1c_force'] or '?'):>7} "
            f"{r['F_gt_sim']:>7.2f} {r['F_gt_real']:>7.2f} {r['F_max']:>7.1f}"
        )

    lines += [
        sep,
        f"Objects: {len(results)}",
        "",
        "── Material Classification ──────────────────────────────",
        f"  Accuracy          : {mat_acc:.1f}%",
        "",
        "── Mass Estimation ──────────────────────────────────────",
        f"  E1a (VLM direct)  : MAE={mm_e1a.get('mae','?')}g   MAPE={mm_e1a.get('mape','?')}%",
        f"  E1c (dims×density): MAE={mm_e1c.get('mae','?')}g   MAPE={mm_e1c.get('mape','?')}%",
        "",
        "── Force vs Simulator GT (mu=0.3) ───────────────────────",
        f"  E1a (VLM mass + table μ)  : MAE={m_e1a.get('mae_vs_sim','?')}N  MAPE={m_e1a.get('mape_vs_sim','?')}%",
        f"  E1b (GT  mass + table μ)  : MAE={m_e1b.get('mae_vs_sim','?')}N  MAPE={m_e1b.get('mape_vs_sim','?')}%",
        f"  E1c (dim mass + table μ)  : MAE={m_e1c.get('mae_vs_sim','?')}N  MAPE={m_e1c.get('mape_vs_sim','?')}%",
        "",
        "── Safety (vs real-world F_min/F_max) ───────────────────",
        f"  E1a: slip={m_e1a.get('slip_pct','?')}%  damage={m_e1a.get('damage_pct','?')}%  safe={m_e1a.get('safe_pct','?')}%",
        f"  E1b: slip={m_e1b.get('slip_pct','?')}%  damage={m_e1b.get('damage_pct','?')}%  safe={m_e1b.get('safe_pct','?')}%",
        f"  E1c: slip={m_e1c.get('slip_pct','?')}%  damage={m_e1c.get('damage_pct','?')}%  safe={m_e1c.get('safe_pct','?')}%",
        "",
        "── Per-Material Breakdown (E1a) ─────────────────────────",
    ] + mat_breakdown + [sep]

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(out_dir, "summary_e1.txt"), 'w') as f:
        f.write(summary)

    # ── scatter plot: predicted vs GT force (all three variants) ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variants = [
        ('E1a_force', 'E1a: VLM mass + table μ', 'steelblue'),
        ('E1b_force', 'E1b: GT mass + table μ',  'seagreen'),
        ('E1c_force', 'E1c: dim×density + table μ', 'darkorange'),
    ]
    for ax, (key, title, color) in zip(axes, variants):
        valid = [(r['F_gt_sim'], r[key]) for r in results if r.get(key)]
        if valid:
            gt_f, pred_f = zip(*valid)
            ax.scatter(gt_f, pred_f, alpha=0.7, color=color, s=30)
            lim = max(max(gt_f), max(pred_f)) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', lw=1, label='ideal')
            ax.set_xlabel('GT force (N, sim)')
            ax.set_ylabel('Predicted force (N)')
            ax.set_title(title)
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_e1.png"), dpi=120)
    plt.close()

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
