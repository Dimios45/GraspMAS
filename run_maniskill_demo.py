import os
import sys
import asyncio
import argparse
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import mani_skill.envs
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import sapien
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.structs import Actor, Pose

sys.path.insert(0, os.path.dirname(__file__))
from agents.graspmas import GraspMAS
from grasp_force import get_object_physics, compute_required_force, get_contact_forces, print_force_report

# ── Output directories (timestamped per run) ─────────────────────────────────
run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir  = os.path.join("runs", run_id)
img_dir  = os.path.join(run_dir, "imgs")
vid_dir  = os.path.join(run_dir, "video")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)
print(f"Run output → {run_dir}/")

# ── Environment ──────────────────────────────────────────────────────────────
env = gym.make(
    "PickClutterYCB-v1",
    obs_mode='rgbd',
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    reward_mode=None,
    sensor_configs=dict(shader_pack='default', width=384, height=384),
    human_render_camera_configs=dict(shader_pack='default'),
    viewer_camera_configs=dict(shader_pack='default'),
    sim_backend='cpu',
    enable_shadow=True,
)

env = RecordEpisode(
    env,
    output_dir=vid_dir,
    save_trajectory=False,
    trajectory_name='abc',
    save_video=True,
    source_type="motionplanning",
    source_desc="motionplanning solution PickClutterYCB-V1",
    video_fps=20,
    save_on_reset=False,
)

seed = int(np.random.randint(0, 10000))
print(f"Seed: {seed}")
obs, _ = env.reset(seed=seed)
rgb   = obs['sensor_data']['base_camera']['rgb'].cpu().squeeze().numpy()
depth = obs['sensor_data']['base_camera']['depth'].cpu().squeeze().numpy()
third = env.render().squeeze()
if hasattr(third, 'cpu'):
    third = third.cpu().numpy()
else:
    third = np.array(third)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].set_title('RGB'); axes[0].imshow(rgb)
axes[1].set_title('Depth'); axes[1].imshow(depth)
axes[2].set_title('Third view'); axes[2].imshow(third)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'observation.png'), dpi=150)
plt.close()
print(f"Observation saved → {img_dir}/observation.png")

# ── Ground-truth physics ──────────────────────────────────────────────────────
physics   = get_object_physics(env.unwrapped)
force_req = compute_required_force(physics, safety_factor=2.0)
print_force_report(physics, force_req)
query = f"Grasp the {physics['name']}."

# ── GraspMAS ─────────────────────────────────────────────────────────────────
graspmas = GraspMAS(api_file='api.key', max_round=5)

print(f"Query: {query}")
_, grasp_2d = asyncio.run(graspmas.query(query, rgb, save_folder=img_dir))

if grasp_2d is None:
    print("No grasp detected — exiting.")
    env.close()
    sys.exit(1)

print("2D grasp (quality, x, y, w, h, angle):", grasp_2d)

# ── Motion planner ────────────────────────────────────────────────────────────
planner = PandaArmMotionPlanningSolver(
    env,
    debug=False,
    vis=False,
    base_pose=env.unwrapped.agent.robot.pose,
    visualize_target_grasp_pose=False,
    print_env_info=False,
)

# ── 2D → 6DoF ─────────────────────────────────────────────────────────────────
_, x, y, w, h, angle = map(round, grasp_2d)
camera_params = obs["sensor_param"]['base_camera']
K  = camera_params['intrinsic_cv'][0].cpu().numpy()
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
# depth is int16 in mm; convert to meters
z = depth[y, x] / 1000.0
if z < 0.1:
    print(f"WARNING: depth at ({x},{y}) is {z:.4f}m — point may be background")
print(f"Grasp center: image ({x},{y}), depth z={z:.4f}m")

grasp_center_cam = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z, 1])

ext = camera_params['extrinsic_cv'][0].cpu().numpy()
R, t = ext[:3, :3], ext[:3, 3]
T_cam2world = np.eye(4)
T_cam2world[:3, :3] = R.T
T_cam2world[:3, 3]  = -R.T @ t
grasp_center_world = (T_cam2world @ grasp_center_cam)[:3]

angle_rad  = (angle - 90) * np.pi / 180
closing_cam = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
approaching = np.array([0, 0, -1])
grasp_6d = env.unwrapped.agent.build_grasp_pose(approaching, closing_cam, grasp_center_world)
print("6DoF grasp pose:", grasp_6d)

# ── Robot arm execution ───────────────────────────────────────────────────────
print("\nMoving to reach pose...")
planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0, 0, -0.05]))

print("Grasping...")
planner.move_to_pose_with_screw(grasp_6d * sapien.Pose([0.005, 0, 0.015]))
planner.close_gripper()

contact = get_contact_forces(env.unwrapped)
print_force_report(physics, force_req, contact)

print("Moving to goal...")
goal_pose = grasp_6d * sapien.Pose([0, 0, -0.4])
env.unwrapped.goal_pos = torch.from_numpy(goal_pose.p)
env.unwrapped.goal_site.set_pose(Pose.create_from_pq(env.unwrapped.goal_pos))
planner.move_to_pose_with_screw(goal_pose)

planner.close()
env.flush_video()
env.close()
print(f"\nDone! All outputs saved → {run_dir}/")
