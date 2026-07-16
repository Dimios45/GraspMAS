"""
GraspMAS Evaluation — PickSingleYCB (74 objects)
=================================================
Runs the full Planner→Coder→Observer pipeline on every YCB object
with a single fixed random seed. Records per-object metrics and
produces a summary table.

Output: runs/YYYYMMDD_HHMMSS_graspmas_single/
  results.json   — per-object detailed results
  results.csv    — same as CSV
  summary.txt    — human-readable table
  imgs/<id>/     — saved grasp visualizations per object
"""

import os, sys, json, csv, time, asyncio, warnings
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mani_skill.envs
import gymnasium as gym

from agents.graspmas import GraspMAS

SEED       = 42
MAX_ROUND  = 4
API_FILE   = "api.key"


def model_id_to_name(model_id: str) -> str:
    parts = model_id.split('_', 1)
    return (parts[1] if len(parts) > 1 else model_id).replace('_', ' ')


def parse_observer_verdict(observation_summary: str) -> str:
    """Extract VALID/INVALID from observer summary string."""
    if observation_summary is None:
        return "NO_OBS"
    s = observation_summary.upper()
    if "VALID" in s and "INVALID" not in s:
        return "VALID"
    if "INVALID" in s:
        return "INVALID"
    return "UNKNOWN"


def main():
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"{run_id}_graspmas_single")
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(
        'PickSingleYCB-v1',
        obs_mode='rgbd',
        control_mode='pd_joint_pos',
        render_mode='rgb_array',
        sensor_configs=dict(shader_pack='default', width=384, height=384),
        sim_backend='cpu',
    )

    all_model_ids = list(env.unwrapped.all_model_ids)
    print(f"GraspMAS Evaluation — PickSingleYCB")
    print(f"Objects : {len(all_model_ids)}")
    print(f"Seed    : {SEED}")
    print(f"Rounds  : {MAX_ROUND}")
    print(f"Output  : {out_dir}/\n")

    graspmas = GraspMAS(api_file=API_FILE, max_round=MAX_ROUND)
    results  = []

    for i, model_id in enumerate(all_model_ids):
        name    = model_id_to_name(model_id)
        img_dir = os.path.join(out_dir, "imgs", model_id)
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n[{i+1:2d}/{len(all_model_ids)}] {model_id} — \"{name}\"")

        # load object
        env.unwrapped.all_model_ids = np.array([model_id])
        obs, _ = env.reset(seed=SEED, options=dict(reconfigure=True))
        rgb = obs['sensor_data']['base_camera']['rgb'].cpu().squeeze().numpy()

        # save input image
        plt.imsave(os.path.join(img_dir, "input.png"), rgb)

        query = f"Grasp the {name}."
        print(f"  Query: \"{query}\"")

        # reset agent state between objects
        graspmas.plan        = None
        graspmas.observation = None
        graspmas.code        = None

        t0 = time.time()
        try:
            _, grasp_pose = asyncio.run(graspmas.query(query, rgb, save_folder=img_dir))
            error = None
        except Exception as e:
            grasp_pose = None
            error      = str(e)
            print(f"  ERROR: {e}")
        elapsed = round(time.time() - t0, 1)

        # observer verdict from stored summary
        verdict = parse_observer_verdict(graspmas.observation)

        detected = grasp_pose is not None
        print(f"  Detected: {detected}  Verdict: {verdict}  Time: {elapsed}s")
        if grasp_pose:
            print(f"  Grasp: q={grasp_pose[0]:.3f} cx={grasp_pose[1]:.1f} cy={grasp_pose[2]:.1f} "
                  f"w={grasp_pose[3]:.1f} h={grasp_pose[4]:.1f} θ={grasp_pose[5]:.1f}°")

        row = {
            'model_id':   model_id,
            'name':       name,
            'query':      query,
            'detected':   detected,
            'verdict':    verdict,
            'grasp_quality': round(grasp_pose[0], 3) if grasp_pose else None,
            'grasp_cx':   round(grasp_pose[1], 2) if grasp_pose else None,
            'grasp_cy':   round(grasp_pose[2], 2) if grasp_pose else None,
            'grasp_w':    round(grasp_pose[3], 2) if grasp_pose else None,
            'grasp_h':    round(grasp_pose[4], 2) if grasp_pose else None,
            'grasp_angle':round(grasp_pose[5], 2) if grasp_pose else None,
            'observer_summary': graspmas.observation,
            'error':      error,
            'elapsed_s':  elapsed,
        }
        results.append(row)

    env.close()

    # ── save results ──────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    csv_keys = [k for k in results[0] if k != 'observer_summary']
    with open(os.path.join(out_dir, "results.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in csv_keys})

    # ── compute metrics ───────────────────────────────────────────────────────
    n            = len(results)
    n_detected   = sum(r['detected']         for r in results)
    n_valid      = sum(r['verdict'] == 'VALID' for r in results)
    avg_time     = round(np.mean([r['elapsed_s'] for r in results]), 1)
    total_time   = round(sum(r['elapsed_s']   for r in results) / 60, 1)

    # ── summary table ─────────────────────────────────────────────────────────
    sep = "─" * 80
    lines = [
        "GraspMAS Evaluation — PickSingleYCB",
        f"Run: {run_id}   Seed: {SEED}   max_round: {MAX_ROUND}",
        sep,
        f"{'Object':<30} {'Detected':<10} {'Verdict':<10} {'Time(s)':>7}  Grasp(cx,cy,θ)",
        sep,
    ]
    for r in results:
        g = f"({r['grasp_cx']:.0f},{r['grasp_cy']:.0f},{r['grasp_angle']:.0f}°)" \
            if r.get('grasp_cx') is not None else "None"
        lines.append(
            f"{r['name']:<30} {'✓' if r['detected'] else '✗':<10} "
            f"{r['verdict']:<10} {r['elapsed_s']:>7.1f}  {g}"
        )
    lines += [
        sep,
        f"Total objects     : {n}",
        f"Detection rate    : {n_detected}/{n} = {n_detected/n*100:.1f}%",
        f"VALID rate        : {n_valid}/{n} = {n_valid/n*100:.1f}%",
        f"Avg time/object   : {avg_time}s",
        f"Total time        : {total_time} min",
        sep,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(os.path.join(out_dir, "summary.txt"), 'w') as f:
        f.write(summary)

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
