#!/usr/bin/env python3
"""
run_all_objects_demo.py
Run GraspMAS 4-direction demo for all 49 VALID YCB objects in one process.
VLMs and perception models are loaded once; each object gets its own sub-folder
under  object_demo/all_<timestamp>/

Usage:
  CUDA_VISIBLE_DEVICES=0 VLM_DEVICE=cuda:0 python run_all_objects_demo.py [--seed 42]
"""

import argparse, os, sys, asyncio, warnings, json, time
warnings.filterwarnings("ignore")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_ap = argparse.ArgumentParser()
_ap.add_argument("--seed", type=int, default=42)
_ap.add_argument("--resume", default=None,
                 help="Path to existing run dir to resume (skips objects already done)")
_args, _ = _ap.parse_known_args()
SEED = _args.seed

from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import sapien

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env

sys.path.insert(0, os.path.dirname(__file__))

from agents.graspmas import GraspMAS
from grasp_force import compute_required_force, print_force_report
from mani_skill_pick_YCB.pick_single_ycb import PickSingleYCBEnv

# ── All 49 VALID objects from prior eval ────────────────────────────────────
VALID_OBJECTS = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "009_gelatin_box", "010_potted_meat_can", "011_banana",
    "013_apple", "014_lemon", "015_peach", "017_orange",
    "021_bleach_cleanser", "024_bowl", "025_mug", "026_sponge",
    "033_spatula", "035_power_drill", "037_scissors", "040_large_marker",
    "042_adjustable_wrench", "043_phillips_screwdriver", "044_flat_screwdriver",
    "048_hammer", "050_medium_clamp", "051_large_clamp", "052_extra_large_clamp",
    "053_mini_soccer_ball", "054_softball", "055_baseball", "056_tennis_ball",
    "058_golf_ball", "061_foam_brick", "062_dice",
    "065-f_cups", "065-h_cups", "065-i_cups", "065-j_cups",
    "072-a_toy_airplane", "072-b_toy_airplane",
    "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
    "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo",
    "073-g_lego_duplo", "077_rubiks_cube",
]

# ── Pinned env — mutate _STATE["obj_id"] before each gym.make() ──────────────
_STATE  = {"obj_id": VALID_OBJECTS[0]}
_ENV_ID = "PickSingleYCB_batch-v1"

@register_env(_ENV_ID, max_episode_steps=50, asset_download_ids=["ycb"])
class _PinnedEnv(PickSingleYCBEnv):
    def _load_scene(self, options: dict):
        saved = self.all_model_ids
        self.all_model_ids = np.array([_STATE["obj_id"]])
        super()._load_scene(options)
        self.all_model_ids = saved


# ── Physics helpers ──────────────────────────────────────────────────────────
GRAVITY = 9.81

def _friction(env_u):
    for c in env_u._objs[0]._objs[0].get_components():
        if "Physx" in type(c).__name__:
            shapes = c.get_collision_shapes()
            if shapes:
                mat = shapes[0].get_physical_material()
                return mat.static_friction, mat.dynamic_friction
    return 0.5, 0.5

def get_physics(env_u):
    mass = float(env_u._objs[0].get_mass())
    mu_s, mu_d = _friction(env_u)
    raw  = env_u._objs[0].name
    nice = raw.rsplit("-", 1)[0].replace("_", " ")
    return dict(name=nice, full_name=raw, mass_kg=mass,
                static_friction=mu_s, dynamic_friction=mu_d,
                weight_N=mass * GRAVITY)

def is_grasped_lifted(env_u):
    grasped = bool(env_u.agent.is_grasping(env_u.obj).item())
    height  = float(env_u.obj.pose.p[0, 2].item())
    return grasped, height


# ── Output root (or resume existing) ─────────────────────────────────────────
if _args.resume:
    out_dir = _args.resume
    # load already-completed results
    resume_path = os.path.join(out_dir, "results.json")
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            all_results_init = json.load(f)
    else:
        all_results_init = []
    done_objs = {r["obj_id"] for r in all_results_init
                 if r.get("direction") == "bottom"}  # fully done = bottom finished
    done_dirs = {(r["obj_id"], r["direction"]) for r in all_results_init}
    print(f"\nResuming {out_dir}  ({len(done_objs)} objects fully complete)")
else:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("object_demo", f"all_{run_id}")
    all_results_init = []
    done_objs = set()
    done_dirs = set()

os.makedirs(out_dir, exist_ok=True)
print(f"\nBatch demo — {len(VALID_OBJECTS)} objects × 4 prompts")
print(f"Seed    : {SEED}")
print(f"Outputs : {out_dir}/\n")

def _save_results(out_dir, results):
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

# ── GraspMAS — loaded once for the entire batch ──────────────────────────────
graspmas = GraspMAS(api_file="api.key", max_round=5)

# ── Per-object 4-direction demo ──────────────────────────────────────────────
all_results = list(all_results_init)
t_batch_start = time.time()

for obj_idx, obj_id in enumerate(VALID_OBJECTS):
    if obj_id in done_objs:
        print(f"  [{obj_idx+1}/{len(VALID_OBJECTS)}]  {obj_id}  — already done, skipping")
        continue
    _STATE["obj_id"] = obj_id

    parts    = obj_id.split("_", 1)
    obj_name = parts[1].replace("_", " ") if len(parts) > 1 else obj_id
    obj_tag  = parts[1] if len(parts) > 1 else obj_id

    obj_vid_dir = os.path.join(out_dir, obj_tag, "videos")
    os.makedirs(obj_vid_dir, exist_ok=True)

    PROMPTS = [
        ("top",    f"Pick the {obj_name} up from the top"),
        ("right",  f"Pick the {obj_name} up from the right hand side"),
        ("left",   f"Pick the {obj_name} up from the left hand side"),
        ("bottom", f"Pick the {obj_name} up from the bottom part"),
    ]

    print(f"\n{'═'*60}")
    print(f"  [{obj_idx+1}/{len(VALID_OBJECTS)}]  {obj_id}")
    print(f"{'═'*60}")

    for tag, query in PROMPTS:
        img_dir     = os.path.join(out_dir, obj_tag, f"imgs_{tag}")
        dir_vid_dir = os.path.join(out_dir, obj_tag, "videos", tag)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(dir_vid_dir, exist_ok=True)

        # Skip if this direction already has a video or was recorded in prior results
        vid_exists = any(f.endswith(".mp4") for f in os.listdir(dir_vid_dir))
        if vid_exists or (obj_id, tag) in done_dirs:
            print(f"  [{tag}] already done — skipping")
            continue

        graspmas.plan = graspmas.observation = graspmas.code = None

        # Build env
        base_env = gym.make(
            _ENV_ID,
            obs_mode="rgbd",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="dense",
            sensor_configs=dict(shader_pack="default", width=384, height=384),
            human_render_camera_configs=dict(shader_pack="default"),
            viewer_camera_configs=dict(shader_pack="default"),
            sim_backend="cpu",
            enable_shadow=True,
        )
        env = RecordEpisode(
            base_env,
            output_dir=dir_vid_dir,
            save_trajectory=False,
            trajectory_name=f"{obj_tag}_{tag}",
            save_video=True,
            source_type="motionplanning",
            source_desc=f"GraspMAS – {query}",
            video_fps=20,
            save_on_reset=False,
        )

        obs, _ = env.reset(seed=SEED)
        rgb   = obs["sensor_data"]["base_camera"]["rgb"].cpu().squeeze().numpy()
        depth = obs["sensor_data"]["base_camera"]["depth"].cpu().squeeze().numpy()
        third = env.render().squeeze()
        third = third.cpu().numpy() if hasattr(third, "cpu") else np.array(third)

        # Observation strip
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].set_title("RGB");          axes[0].imshow(rgb)
        axes[1].set_title("Depth");        axes[1].imshow(depth, cmap="inferno")
        axes[2].set_title("Third-person"); axes[2].imshow(third)
        plt.suptitle(query, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, "observation.png"), dpi=150)
        plt.close()

        physics   = get_physics(env.unwrapped)
        force_req = compute_required_force(physics, safety_factor=2.0)

        # GraspMAS
        t0 = time.time()
        try:
            _, grasp_2d = asyncio.run(graspmas.query(query, rgb, save_folder=img_dir))
        except Exception as e:
            print(f"  [{tag}] GraspMAS error: {e}")
            try:
                env.flush_video(save=False)
                env.close()
            except Exception:
                pass
            rec = dict(obj_id=obj_id, obj_name=obj_name, direction=tag,
                       query=query, grasp_2d=None, mass_kg=physics["mass_kg"],
                       elapsed_graspmas_s=round(time.time() - t0, 1),
                       grasped=False, height=0.0, status="error", error=str(e))
            all_results.append(rec)
            _save_results(out_dir, all_results)
            continue
        elapsed_graspmas = time.time() - t0

        rec = dict(obj_id=obj_id, obj_name=obj_name, direction=tag,
                   query=query, grasp_2d=grasp_2d,
                   mass_kg=physics["mass_kg"],
                   elapsed_graspmas_s=round(elapsed_graspmas, 1))

        if grasp_2d is None:
            print(f"  [{tag}] No grasp detected.")
            env.flush_video(save=False)
            env.close()
            rec.update(grasped=False, height=0.0, status="no_grasp")
            all_results.append(rec)
            _save_results(out_dir, all_results)
            continue

        # 2D → 6DoF: GT position + directional offset + VLM jaw angle
        _, _cx, _cy, w, h, angle = grasp_2d
        obj_u   = env.unwrapped
        obj_pos = obj_u._objs[0].pose.p[0].cpu().numpy()

        mesh   = obj_u._objs[0].get_first_collision_mesh()
        bounds = mesh.bounding_box.bounds
        hy = (bounds[1, 1] - bounds[0, 1]) / 2 * 0.55
        hz = (bounds[1, 2] - bounds[0, 2]) / 2 * 0.55
        # Camera looks in -X direction → camera-right = world +Y, camera-left = world -Y.
        # "bottom" targets the lower bbox half but still approaches top-down (table blocks below).
        dir_offset = {
            "top":    np.array([0,   0,  hz]),
            "bottom": np.array([0,   0, -hz * 0.5]),
            "right":  np.array([0,  hy,  0]),
            "left":   np.array([0, -hy,  0]),
        }.get(tag, np.zeros(3))

        angle_rad   = (angle - 90) * np.pi / 180
        closing_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        approach_dir = {
            "top":    np.array([0.0,  0.0, -1.0]),
            "right":  np.array([0.0, -1.0, -1.0]) / np.sqrt(2),
            "left":   np.array([0.0,  1.0, -1.0]) / np.sqrt(2),
            "bottom": np.array([0.0,  0.0, -1.0]),
        }.get(tag, np.array([0., 0., -1.]))
        # Gram-Schmidt: ensure closing_dir ⊥ approach_dir
        closing_dir = closing_dir - np.dot(closing_dir, approach_dir) * approach_dir
        closing_dir = closing_dir / np.linalg.norm(closing_dir)
        grasp_6d    = env.unwrapped.agent.build_grasp_pose(
                          approach_dir, closing_dir, obj_pos + dir_offset)

        planner = PandaArmMotionPlanningSolver(
            env, debug=False, vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )

        planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0, 0, -0.05]))
        planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0.005, 0, 0.015]))
        planner.close_gripper()

        goal_pose = grasp_6d * sapien.Pose([0, 0, -0.4])
        env.unwrapped.goal_pos = torch.from_numpy(goal_pose.p)
        env.unwrapped.goal_site.set_pose(Pose.create_from_pq(env.unwrapped.goal_pos))
        planner.move_to_pose_with_screw(goal_pose)

        grasped, height = is_grasped_lifted(env.unwrapped)
        planner.close()
        env.flush_video()
        env.close()

        rec.update(grasped=grasped, height=round(height, 4), status="ok",
                   grasp_angle=angle)
        all_results.append(rec)
        _save_results(out_dir, all_results)
        print(f"  [{tag}] grasped={grasped}  height={height:.3f} m  angle={angle:.1f}°")

    # Per-object mini-summary
    obj_rows = [r for r in all_results if r["obj_id"] == obj_id]
    n_ok = sum(1 for r in obj_rows if r.get("grasped"))
    print(f"  → {obj_name}: {n_ok}/4 directions grasped")

# ── Save full results ────────────────────────────────────────────────────────
results_path = os.path.join(out_dir, "results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

# ── Final summary table ──────────────────────────────────────────────────────
elapsed_total = (time.time() - t_batch_start) / 60
print(f"\n{'═'*70}")
print(f"  BATCH DEMO COMPLETE — {len(VALID_OBJECTS)} objects × 4 prompts")
print(f"{'═'*70}")
print(f"  {'Object':<30}  top   right  left   bottom")
print(f"  {'-'*30}  {'─'*28}")
for obj_id in VALID_OBJECTS:
    rows = {r["direction"]: r.get("grasped", False)
            for r in all_results if r["obj_id"] == obj_id}
    parts = obj_id.split("_", 1)
    name  = (parts[1].replace("_", " ") if len(parts) > 1 else obj_id)[:29]
    cols  = " ".join("YES" if rows.get(d) else "NO " for d in ("top","right","left","bottom"))
    print(f"  {name:<30}  {cols}")

total_grasped = sum(1 for r in all_results if r.get("grasped"))
total_attempts = len(all_results)
print(f"\n  Overall: {total_grasped}/{total_attempts} grasps succeeded "
      f"({100*total_grasped/total_attempts:.1f}%)")
print(f"  Total time: {elapsed_total:.1f} min")
print(f"  Results JSON → {results_path}")
print(f"  Videos → {out_dir}/<object>/videos/")
print(f"{'═'*70}\n")
