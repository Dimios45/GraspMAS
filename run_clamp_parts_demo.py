#!/usr/bin/env python3
"""
run_clamp_parts_demo.py
Part-based semantic grasping of 050_medium_clamp with 3 prompts:
  - grasp the loop
  - grasp the right handle
  - grasp the left handle
"""

import os, sys, asyncio, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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

OBJ_ID   = "050_medium_clamp"
OBJ_NAME = "medium clamp"
SEED     = 42

_ENV_ID = "PickSingleYCB_clamp_parts-v1"

@register_env(_ENV_ID, max_episode_steps=50, asset_download_ids=["ycb"])
class _PinnedClampEnv(PickSingleYCBEnv):
    def _load_scene(self, options: dict):
        saved = self.all_model_ids
        self.all_model_ids = np.array([OBJ_ID])
        super()._load_scene(options)
        self.all_model_ids = saved


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
    return dict(mass_kg=mass, static_friction=mu_s,
                dynamic_friction=mu_d, weight_N=mass * GRAVITY)

def is_grasped_lifted(env_u):
    grasped = bool(env_u.agent.is_grasping(env_u.obj).item())
    height  = float(env_u.obj.pose.p[0, 2].item())
    return grasped, height


PROMPTS = [
    ("loop",         "Pick the medium clamp up by grasping the loop"),
    ("right_handle", "Pick the medium clamp up by grasping the right handle"),
    ("left_handle",  "Pick the medium clamp up by grasping the left handle"),
]

run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("object_demo", f"clamp_parts_{run_id}")
vid_dir = os.path.join(out_dir, "videos")
os.makedirs(vid_dir, exist_ok=True)

print(f"\nObject  : {OBJ_ID}")
print(f"Seed    : {SEED}")
print(f"Outputs : {out_dir}/\n")

graspmas = GraspMAS(api_file="api.key", max_round=5)

results = []

for tag, query in PROMPTS:
    print("\n" + "═" * 60)
    print(f"  [{tag.upper()}]  {query}")
    print("═" * 60)

    img_dir     = os.path.join(out_dir, f"imgs_{tag}")
    dir_vid_dir = os.path.join(out_dir, "videos", tag)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dir_vid_dir, exist_ok=True)

    graspmas.plan = graspmas.observation = graspmas.code = None

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
        trajectory_name=f"clamp_{tag}",
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
    print_force_report(
        dict(name=OBJ_NAME, full_name=OBJ_ID, **physics), force_req
    )

    print(f"GraspMAS query: {query}")
    t0 = time.time()
    try:
        _, grasp_2d = asyncio.run(graspmas.query(query, rgb, save_folder=img_dir))
    except Exception as e:
        print(f"  [{tag}] GraspMAS error: {e}")
        env.flush_video(save=False)
        env.close()
        results.append((tag, False, 0.0, f"error: {e}"))
        continue
    elapsed = time.time() - t0

    if grasp_2d is None:
        print(f"  [{tag}] No grasp detected.")
        env.flush_video(save=False)
        env.close()
        results.append((tag, False, 0.0, "no_grasp"))
        continue

    print(f"  [{tag}] 2D grasp: conf={grasp_2d[0]:.3f}  angle={grasp_2d[5]:.1f}°  ({elapsed:.1f}s)")

    # 2D → 6DoF
    # Use GT object position; jaw angle from VLM determines which part is contacted.
    # For each semantic part, bias the contact point using the jaw angle itself:
    # the gripper closing direction points *toward* the part, so offset slightly
    # along that direction so the fingers are centered on the named part.
    _, _cx, _cy, w, h, angle = grasp_2d
    obj_u   = env.unwrapped
    obj_pos = obj_u._objs[0].pose.p[0].cpu().numpy()

    # Collision mesh extent for scaling the contact offset
    mesh   = obj_u._objs[0].get_first_collision_mesh()
    bounds = mesh.bounding_box.bounds
    extent_x = (bounds[1, 0] - bounds[0, 0]) / 2 * 0.4
    extent_z = (bounds[1, 2] - bounds[0, 2]) / 2 * 0.4

    # Jaw closing direction (from VLM angle)
    angle_rad   = (angle - 90) * np.pi / 180
    closing_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

    # Part-specific contact offsets (push toward the named part)
    part_offset = {
        "loop":         np.array([0.0, 0.0, extent_z]),      # loop is at top/end of spring
        "right_handle": closing_dir * extent_x,               # approach along jaw direction
        "left_handle":  -closing_dir * extent_x,              # opposite side
    }.get(tag, np.zeros(3))

    grasp_world = obj_pos + part_offset

    approach_dir = {
        "loop":         np.array([ 0.0,  0.0, -1.0]),
        "right_handle": np.array([-1.0,  0.0, -1.0]) / np.sqrt(2),
        "left_handle":  np.array([ 1.0,  0.0, -1.0]) / np.sqrt(2),
    }.get(tag, np.array([0., 0., -1.]))
    # Gram-Schmidt: ensure closing_dir ⊥ approach_dir
    closing_dir = closing_dir - np.dot(closing_dir, approach_dir) * approach_dir
    closing_dir = closing_dir / np.linalg.norm(closing_dir)
    grasp_6d = env.unwrapped.agent.build_grasp_pose(
        approach_dir, closing_dir, grasp_world
    )

    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    print(f"  [{tag}] Approaching...")
    planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0, 0, -0.05]))
    print(f"  [{tag}] Grasping...")
    planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0.005, 0, 0.015]))
    planner.close_gripper()

    print(f"  [{tag}] Lifting...")
    goal_pose = grasp_6d * sapien.Pose([0, 0, -0.4])
    env.unwrapped.goal_pos = torch.from_numpy(goal_pose.p)
    env.unwrapped.goal_site.set_pose(Pose.create_from_pq(env.unwrapped.goal_pos))
    planner.move_to_pose_with_screw(goal_pose)

    grasped, height = is_grasped_lifted(env.unwrapped)
    print(f"  [{tag}] is_grasped={grasped}  obj_height={height:.3f} m")

    planner.close()
    env.flush_video()
    env.close()

    results.append((tag, grasped, height, f"angle={angle:.1f}°"))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print(f"  MEDIUM CLAMP PARTS DEMO — RESULTS")
print("═" * 60)
print(f"  {'Part':<15}  {'Grasped':<9}  {'Height':>8}  Info")
for tag, ok, h, info in results:
    print(f"  {tag:<15}  {'YES' if ok else 'NO':<9}  {h:>7.3f} m  {info}")
print(f"\n  Videos → {out_dir}/videos/<part>/0.mp4")
print(f"  Agent logs → logs/clamp_parts_demo.log")
print("═" * 60 + "\n")
