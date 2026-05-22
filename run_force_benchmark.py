"""
Force estimation benchmark: ground truth simulator physics vs Qwen2-VL visual estimates.

For each YCB object in PickSingleYCB:
  1. Load the object in isolation
  2. Read ground truth mass + friction → compute F_required
  3. Show RGB image to Qwen → ask for material, mass, friction, grip force estimate
  4. Compare and log results
"""

import os, sys, re, json, csv, time, warnings
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, base64

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import mani_skill.envs
import gymnasium as gym

from local_vlm import vlm_chat

GRAVITY        = 9.81
SAFETY_FACTOR  = 2.0
PANDA_MAX_N    = 70.0

VLM_PROMPT = """\
You are a robotics assistant estimating physical properties from a single image.

The image shows a {name} on a table. Estimate:
1. material  — choose one: metal / plastic / rubber / wood / fabric / food / ceramic / other
2. mass_g    — approximate mass in grams (integer)
3. friction  — coefficient of friction: 0.1=very slippery glass, 0.2=smooth plastic, \
0.3=average plastic/metal, 0.5=rubber, 0.7=very grippy rubber
4. grip_force_N — required gripper finger force (Newtons) to safely lift with a parallel-jaw \
gripper, using formula: force = {safety} × mass_kg × {g} / (2 × friction)

Reply in EXACTLY this format (numbers only, no units):
material: <value>
mass_g: <number>
friction: <number>
grip_force_N: <number>
"""


def get_physics(env_unwrapped):
    obj = env_unwrapped.obj
    mass = float(obj.get_mass())
    mu_s = 0.3
    for c in obj._objs[0].get_components():
        if 'Physx' in type(c).__name__:
            shapes = c.get_collision_shapes()
            if shapes:
                mu_s = float(shapes[0].get_physical_material().static_friction)
    F_min  = mass * GRAVITY / (2 * mu_s)
    F_req  = SAFETY_FACTOR * F_min
    return dict(mass_kg=mass, mass_g=round(mass*1000,1), friction=mu_s,
                F_min_N=round(F_min,3), F_required_N=round(F_req,3))


def encode_rgb(rgb_np):
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


def query_vlm(rgb_np, name):
    b64 = encode_rgb(rgb_np)
    prompt = VLM_PROMPT.format(name=name, safety=SAFETY_FACTOR, g=round(GRAVITY,2))
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text",      "text": prompt}
    ]}]
    raw = vlm_chat(messages, max_new_tokens=80)
    return raw


def parse_vlm(raw):
    result = {}
    for line in raw.splitlines():
        m = re.match(r'(\w+(?:_\w+)*):\s*(.+)', line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            if key == 'material':
                result['material'] = val
            else:
                try:
                    result[key] = float(re.search(r'[\d.]+', val).group())
                except:
                    result[key] = None
    return result


def model_id_to_name(model_id):
    # '011_banana' → 'banana'
    parts = model_id.split('_', 1)
    name = parts[1] if len(parts) > 1 else model_id
    return name.replace('_', ' ')


def main():
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"{run_id}_force_benchmark")
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
    print(f"Benchmark: {len(all_model_ids)} YCB objects")
    print(f"Output   : {out_dir}/\n")

    results = []

    for i, model_id in enumerate(all_model_ids):
        name = model_id_to_name(model_id)
        print(f"[{i+1:2d}/{len(all_model_ids)}] {model_id} ({name})")

        # ── load object ──────────────────────────────────────────────────────
        t0 = time.time()
        env.unwrapped.all_model_ids = np.array([model_id])
        obs, _ = env.reset(options=dict(reconfigure=True))
        rgb = obs['sensor_data']['base_camera']['rgb'].cpu().squeeze().numpy()

        # ── ground truth physics ─────────────────────────────────────────────
        gt = get_physics(env.unwrapped)

        # ── save image ───────────────────────────────────────────────────────
        img_path = os.path.join(img_dir, f"{model_id}.png")
        plt.imsave(img_path, rgb)

        # ── VLM estimate ─────────────────────────────────────────────────────
        raw  = query_vlm(rgb, name)
        vlm  = parse_vlm(raw)
        t1   = time.time()

        # ── compute errors ───────────────────────────────────────────────────
        vlm_force = vlm.get('grip_force_N')
        mass_err  = round(vlm.get('mass_g', 0) - gt['mass_g'], 1) if vlm.get('mass_g') else None
        force_err = round(vlm_force - gt['F_required_N'], 3)       if vlm_force          else None
        force_pct = round(force_err / gt['F_required_N'] * 100, 1) if force_err is not None else None

        row = {
            'model_id':       model_id,
            'name':           name,
            # ground truth
            'gt_mass_g':      gt['mass_g'],
            'gt_friction':    gt['friction'],
            'gt_F_min_N':     gt['F_min_N'],
            'gt_F_required_N': gt['F_required_N'],
            # VLM estimates
            'vlm_material':   vlm.get('material'),
            'vlm_mass_g':     vlm.get('mass_g'),
            'vlm_friction':   vlm.get('friction'),
            'vlm_force_N':    vlm_force,
            'vlm_raw':        raw.strip(),
            # errors
            'mass_err_g':     mass_err,
            'force_err_N':    force_err,
            'force_err_pct':  force_pct,
            'elapsed_s':      round(t1-t0, 1),
        }
        results.append(row)

        print(f"  GT:  {gt['mass_g']:.0f}g  μ={gt['friction']:.2f}  F={gt['F_required_N']:.2f}N")
        print(f"  VLM: {vlm.get('mass_g','?')}g  μ={vlm.get('friction','?')}  F={vlm_force or '?'}N"
              f"  (err: {force_pct:+.0f}%)" if force_pct is not None else "  VLM: parse failed")
        print(f"  [{t1-t0:.1f}s]")

    env.close()

    # ── save results ─────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, "results.csv")
    keys = [k for k in results[0] if k != 'vlm_raw']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})

    # ── summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r['force_err_pct'] is not None]
    mae_force = np.mean([abs(r['force_err_N'])  for r in valid])
    mae_mass  = np.mean([abs(r['mass_err_g'])   for r in valid if r['mass_err_g'] is not None])
    mae_pct   = np.mean([abs(r['force_err_pct']) for r in valid])

    summary_lines = [
        f"{'Object':<30} {'GT_mass':>8} {'VLM_mass':>9} {'GT_F':>7} {'VLM_F':>7} {'err%':>6} {'material'}",
        "-" * 90,
    ]
    for r in results:
        summary_lines.append(
            f"{r['name']:<30} {r['gt_mass_g']:>7.0f}g {str(r['vlm_mass_g'] or '?'):>8}g "
            f"{r['gt_F_required_N']:>6.2f}N {str(r['vlm_force_N'] or '?'):>6}N "
            f"{(str(r['force_err_pct'])+' %') if r['force_err_pct'] is not None else '   N/A':>7}  "
            f"{r['vlm_material'] or '?'}"
        )
    summary_lines += [
        "-" * 90,
        f"Valid responses   : {len(valid)}/{len(results)}",
        f"MAE force (N)     : {mae_force:.3f}",
        f"MAE force (%)     : {mae_pct:.1f}%",
        f"MAE mass  (g)     : {mae_mass:.1f}",
    ]

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, 'w') as f:
        f.write(summary)

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
