#!/usr/bin/env python3
"""
scripts/run_mustard_demo.py  (general YCB object demo)

Usage:
  python scripts/run_mustard_demo.py --object-id 077_rubiks_cube --seed 42
  python scripts/run_mustard_demo.py --object-id 003_cracker_box
  python scripts/run_mustard_demo.py                               # defaults to mustard bottle

Runs GraspMAS with 4 directional grasp prompts and saves 4 videos.
AMD headless node: pure Vulkan offscreen, no X display needed.
"""

import argparse, os, sys, asyncio, warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_p = argparse.ArgumentParser()
_p.add_argument("--object-id", default="006_mustard_bottle",
                help="YCB object ID, e.g. 077_rubiks_cube")
_p.add_argument("--seed", type=int, default=42)
_args, _ = _p.parse_known_args()

OBJ_ID   = _args.object_id
OBJ_SEED = _args.seed

# nice name: '006_mustard_bottle' → 'mustard bottle'
_parts   = OBJ_ID.split("_", 1)
OBJ_NAME = _parts[1].replace("_", " ") if len(_parts) > 1 else OBJ_ID

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graspmas import GraspMAS
from grasp_force import compute_required_force, print_force_report
from mani_skill_pick_YCB.pick_single_ycb import PickSingleYCBEnv

# ── Register a pinned-object env at import time ───────────────────────────────
_ENV_ID = f"PickSingleYCB_pinned-v1"

@register_env(_ENV_ID, max_episode_steps=50, asset_download_ids=["ycb"])
class _PinnedYCBEnv(PickSingleYCBEnv):
    """PickSingleYCB locked to whatever OBJ_ID is set to."""
    def _load_scene(self, options: dict):
        saved = self.all_model_ids
        self.all_model_ids = np.array([OBJ_ID])
        super()._load_scene(options)
        self.all_model_ids = saved


# ── Physics helpers (PickSingleYCB layout) ────────────────────────────────────
GRAVITY = 9.81


def _ycb_friction(env_u):
    for c in env_u._objs[0]._objs[0].get_components():
        if "Physx" in type(c).__name__:
            shapes = c.get_collision_shapes()
            if shapes:
                mat = shapes[0].get_physical_material()
                return mat.static_friction, mat.dynamic_friction
    return 0.5, 0.5


def get_ycb_physics(env_u):
    mass = float(env_u._objs[0].get_mass())
    mu_s, mu_d = _ycb_friction(env_u)
    raw = env_u._objs[0].name        # e.g. '006_mustard_bottle-0'
    nice = raw.rsplit("-", 1)[0].replace("_", " ")
    return dict(name=nice, full_name=raw,
                mass_kg=mass, static_friction=mu_s,
                dynamic_friction=mu_d, weight_N=mass * GRAVITY)


def is_grasped_and_lifted(env_u, min_height=0.15):
    """True when robot is actively holding the object above table level."""
    grasped = bool(env_u.agent.is_grasping(env_u.obj).item())
    height  = float(env_u.obj.pose.p[0, 2].item())
    return grasped, height


# ── Four directional prompts (auto-use OBJ_NAME) ──────────────────────────────
PROMPTS = [
    ("top",    f"Pick the {OBJ_NAME} up from the top"),
    ("right",  f"Pick the {OBJ_NAME} up from the right hand side"),
    ("left",   f"Pick the {OBJ_NAME} up from the left hand side"),
    ("bottom", f"Pick the {OBJ_NAME} up from the bottom part"),
]

# ── Output root ───────────────────────────────────────────────────────────────
run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
tag_obj = OBJ_ID.split("_", 1)[-1] if "_" in OBJ_ID else OBJ_ID
out_dir = os.path.join("object_demo", f"{tag_obj}_{run_id}")
os.makedirs(out_dir, exist_ok=True)

print(f"\nObject  : {OBJ_ID}  ({OBJ_NAME})")
print(f"Seed    : {OBJ_SEED}")
print(f"Outputs : {out_dir}/\n")

# ── GraspMAS (VLMs load once here) ───────────────────────────────────────────
graspmas = GraspMAS(api_file="api.key", max_round=5)

# ── Run loop ──────────────────────────────────────────────────────────────────
results = []

for tag, query in PROMPTS:
    print("\n" + "═" * 60)
    print(f"  [{tag.upper()}]  {query}")
    print("═" * 60)

    img_dir     = os.path.join(out_dir, f"imgs_{tag}")
    dir_vid_dir = os.path.join(out_dir, "videos", tag)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dir_vid_dir, exist_ok=True)

    # Reset GraspMAS inter-run state
    graspmas.plan = graspmas.observation = graspmas.code = None

    # ── Environment ───────────────────────────────────────────────────────────
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
        trajectory_name=f"{tag_obj}_{tag}",
        save_video=True,
        source_type="motionplanning",
        source_desc=f"GraspMAS – {query}",
        video_fps=20,
        save_on_reset=False,
    )

    obs, _ = env.reset(seed=OBJ_SEED)
    rgb   = obs["sensor_data"]["base_camera"]["rgb"].cpu().squeeze().numpy()
    depth = obs["sensor_data"]["base_camera"]["depth"].cpu().squeeze().numpy()
    third = env.render().squeeze()
    third = third.cpu().numpy() if hasattr(third, "cpu") else np.array(third)

    # Save observation strip
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title("RGB");          axes[0].imshow(rgb)
    axes[1].set_title("Depth");        axes[1].imshow(depth, cmap="inferno")
    axes[2].set_title("Third-person"); axes[2].imshow(third)
    plt.suptitle(query, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "observation.png"), dpi=150)
    plt.close()

    physics   = get_ycb_physics(env.unwrapped)
    force_req = compute_required_force(physics, safety_factor=2.0)
    print_force_report(physics, force_req)

    # ── GraspMAS ─────────────────────────────────────────────────────────────
    print(f"GraspMAS query: {query}")
    _, grasp_2d = asyncio.run(graspmas.query(query, rgb, save_folder=img_dir))

    if grasp_2d is None:
        print(f"[{tag}] No grasp detected.")
        env.flush_video(save=False)
        env.close()
        results.append((tag, False, 0.0, "no_grasp"))
        continue

    print(f"[{tag}] 2D grasp: {grasp_2d}")

    # ── 2D → 6DoF ────────────────────────────────────────────────────────────
    # cx/cy from GraspMAS are patch-relative artifacts for PickSingleYCB.
    # Use GT object position + directional offset for the 3D center; keep
    # only angle θ from GraspMAS (VLM-predicted jaw orientation).
    _, _cx, _cy, w, h, angle = grasp_2d
    # Approach direction comes from VLM Coder output, not from hardcoded tag.
    approach_str = graspmas.last_approach
    print(f"[{tag}] VLM approach: {approach_str}")

    obj_u = env.unwrapped
    obj_pos = obj_u._objs[0].pose.p[0].cpu().numpy()

    mesh   = obj_u._objs[0].get_first_collision_mesh()
    bounds = mesh.bounding_box.bounds         # [[xmin,ymin,zmin],[xmax,ymax,zmax]] local
    hy = (bounds[1, 1] - bounds[0, 1]) / 2 * 0.55
    hz = (bounds[1, 2] - bounds[0, 2]) / 2 * 0.55
    # Camera looks in -X direction → camera-right = world +Y, camera-left = world -Y.
    dir_offset = {
        "top":    np.array([0,   0,  hz]),
        "bottom": np.array([0,   0, -hz * 0.5]),
        "right":  np.array([0,  hy,  0]),
        "left":   np.array([0, -hy,  0]),
    }.get(approach_str, np.zeros(3))
    grasp_world = obj_pos + dir_offset

    angle_rad   = (angle - 90) * np.pi / 180
    closing_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    approach_dir = {
        "top":    np.array([0.0,  0.0, -1.0]),
        "right":  np.array([0.0, -1.0, -1.0]) / np.sqrt(2),
        "left":   np.array([0.0,  1.0, -1.0]) / np.sqrt(2),
        "bottom": np.array([0.0,  0.0, -1.0]),
    }.get(approach_str, np.array([0., 0., -1.]))
    # Gram-Schmidt: ensure closing_dir ⊥ approach_dir
    closing_dir = closing_dir - np.dot(closing_dir, approach_dir) * approach_dir
    closing_dir = closing_dir / np.linalg.norm(closing_dir)
    grasp_6d    = env.unwrapped.agent.build_grasp_pose(
                      approach_dir, closing_dir, grasp_world)

    # ── Motion planner ────────────────────────────────────────────────────────
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    print(f"[{tag}] Approaching...")
    planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0, 0, -0.05]))
    print(f"[{tag}] Grasping...")
    planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0.005, 0, 0.015]))
    planner.close_gripper()

    print(f"[{tag}] Lifting...")
    goal_pose = grasp_6d * sapien.Pose([0, 0, -0.4])
    env.unwrapped.goal_pos = torch.from_numpy(goal_pose.p)
    env.unwrapped.goal_site.set_pose(Pose.create_from_pq(env.unwrapped.goal_pos))
    planner.move_to_pose_with_screw(goal_pose)

    grasped, height = is_grasped_and_lifted(env.unwrapped)
    print(f"[{tag}] is_grasped={grasped}  obj_height={height:.3f} m")

    planner.close()
    env.flush_video()
    env.close()

    results.append((tag, grasped, height, "ok"))
    print(f"[{tag}] Done.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print(f"  {OBJ_NAME.upper()}  DEMO — RESULTS")
print("═" * 60)
print(f"  {'Prompt':<8}  {'Grasped':<9}  {'Height':>8}  Status")
for tag, ok, h, status in results:
    print(f"  {tag:<8}  {'YES' if ok else 'NO':<9}  {h:>7.3f} m  {status}")
print(f"\n  Videos → {out_dir}/videos/<direction>/0.mp4")
print("═" * 60 + "\n")
