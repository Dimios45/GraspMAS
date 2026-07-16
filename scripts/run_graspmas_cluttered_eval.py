"""
GraspMAS Evaluation — PickClutterYCB (100 random seeds)
========================================================
Runs the full Planner→Coder→Observer pipeline on cluttered scenes.
Each seed produces a different arrangement of multiple YCB objects.
A random target object is selected and the query is auto-generated.

Output: runs/YYYYMMDD_HHMMSS_graspmas_cluttered/
  results.json   — per-seed detailed results
  results.csv    — same as CSV
  summary.txt    — human-readable table
  imgs/<seed>/   — saved grasp visualizations
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

N_SEEDS    = 100
MAX_ROUND  = 4
API_FILE   = "api.key"

SEEDS = list(range(N_SEEDS))  # seeds 0–99 for reproducibility


def get_target_name(env_unwrapped) -> str:
    """
    Extract clean target object name from PickClutterYCB env.
    target_object.name is always "target_object" (generic) — match by pose
    against selectable_target_objects to find the real actor name.
    """
    try:
        tgt_pos = env_unwrapped.target_object.pose.sp.p
        for actor in env_unwrapped.selectable_target_objects[0]:
            if np.allclose(actor.pose.sp.p, tgt_pos, atol=1e-3):
                # actor.name looks like "set_0_011_banana"
                parts = actor.name.split('_', 3)
                return parts[-1].replace('_', ' ') if len(parts) >= 4 else actor.name
    except Exception:
        pass
    return "object"


def parse_observer_verdict(observation_summary: str) -> str:
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
    out_dir = os.path.join("runs", f"{run_id}_graspmas_cluttered")
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(
        'PickClutterYCB-v1',
        obs_mode='rgbd',
        control_mode='pd_joint_pos',
        render_mode='rgb_array',
        sensor_configs=dict(shader_pack='default', width=384, height=384),
        sim_backend='cpu',
    )

    print(f"GraspMAS Evaluation — PickClutterYCB")
    print(f"Seeds   : {N_SEEDS} (0–{N_SEEDS-1})")
    print(f"Rounds  : {MAX_ROUND}")
    print(f"Output  : {out_dir}/\n")

    graspmas = GraspMAS(api_file=API_FILE, max_round=MAX_ROUND)
    results  = []

    for i, seed in enumerate(SEEDS):
        img_dir = os.path.join(out_dir, "imgs", f"seed_{seed:03d}")
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n[{i+1:3d}/{N_SEEDS}] seed={seed}")

        obs, _ = env.reset(seed=seed)
        rgb  = obs['sensor_data']['base_camera']['rgb'].cpu().squeeze().numpy()
        name = get_target_name(env.unwrapped)
        plt.imsave(os.path.join(img_dir, "input.png"), rgb)

        query = f"Grasp the {name}."
        print(f"  Target: \"{name}\"  Query: \"{query}\"")

        # reset agent state
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

        verdict  = parse_observer_verdict(graspmas.observation)
        detected = grasp_pose is not None
        print(f"  Detected: {detected}  Verdict: {verdict}  Time: {elapsed}s")

        row = {
            'seed':       seed,
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

    # ── metrics ───────────────────────────────────────────────────────────────
    n          = len(results)
    n_detected = sum(r['detected']           for r in results)
    n_valid    = sum(r['verdict'] == 'VALID' for r in results)
    avg_time   = round(np.mean([r['elapsed_s'] for r in results]), 1)
    total_time = round(sum(r['elapsed_s']     for r in results) / 60, 1)

    # per-object-category breakdown
    from collections import defaultdict
    by_obj = defaultdict(list)
    for r in results:
        by_obj[r['name']].append(r)

    sep = "─" * 75
    lines = [
        "GraspMAS Evaluation — PickClutterYCB",
        f"Run: {run_id}   Seeds: 0–{N_SEEDS-1}   max_round: {MAX_ROUND}",
        sep,
        f"{'Seed':>5} {'Target':<25} {'Det':>4} {'Verdict':<10} {'Time':>6}",
        sep,
    ]
    for r in results:
        lines.append(
            f"{r['seed']:>5} {r['name']:<25} "
            f"{'✓' if r['detected'] else '✗':>4} "
            f"{r['verdict']:<10} {r['elapsed_s']:>6.1f}s"
        )

    lines += [
        sep,
        f"Total seeds       : {n}",
        f"Detection rate    : {n_detected}/{n} = {n_detected/n*100:.1f}%",
        f"VALID rate        : {n_valid}/{n} = {n_valid/n*100:.1f}%",
        f"Avg time/seed     : {avg_time}s",
        f"Total time        : {total_time} min",
        "",
        "── Per-Object Breakdown ─────────────────────────────────",
    ]
    for obj, rows in sorted(by_obj.items()):
        nd = sum(r['detected'] for r in rows)
        nv = sum(r['verdict'] == 'VALID' for r in rows)
        lines.append(
            f"  {obj:<28} n={len(rows):2d}  det={nd/len(rows)*100:.0f}%  valid={nv/len(rows)*100:.0f}%"
        )
    lines.append(sep)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), 'w') as f:
        f.write(summary)

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
