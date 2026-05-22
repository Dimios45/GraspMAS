import os
import sys
import asyncio
import warnings
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

os.makedirs('simulated_demo/pick_YCB_Clutter', exist_ok=True)
env = RecordEpisode(
    env,
    output_dir='simulated_demo/pick_YCB_Clutter',
    save_trajectory=False,
    trajectory_name='abc',
    save_video=True,
    source_type="motionplanning",
    source_desc="motionplanning solution PickClutterYCB-V1",
    video_fps=20,
    save_on_reset=False,
)

seed = 17
obs, _ = env.reset(seed=seed)
rgb   = obs['sensor_data']['base_camera']['rgb'].cpu().squeeze().numpy()
depth = obs['sensor_data']['base_camera']['depth'].cpu().squeeze().numpy()
third = env.render().squeeze()
if hasattr(third, 'cpu'):
    third = third.cpu().numpy()
else:
    third = np.array(third)

os.makedirs('imgs', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].set_title('RGB'); axes[0].imshow(rgb)
axes[1].set_title('Depth'); axes[1].imshow(depth)
axes[2].set_title('Third view'); axes[2].imshow(third)
plt.tight_layout()
plt.savefig('imgs/maniskill_obs.png', dpi=150)
plt.close()
print("Observation saved → imgs/maniskill_obs.png")

# ── GraspMAS ─────────────────────────────────────────────────────────────────
graspmas = GraspMAS(api_file='api.key', max_round=5)

query = "Grasp the lemon."
print(f"\nQuery: {query}")
_, grasp_2d = asyncio.run(graspmas.query(query, rgb))

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

print("Moving to goal...")
goal_pose = grasp_6d * sapien.Pose([0, 0, -0.4])
env.unwrapped.goal_pos = torch.from_numpy(goal_pose.p)
env.unwrapped.goal_site.set_pose(Pose.create_from_pq(env.unwrapped.goal_pos))
planner.move_to_pose_with_screw(goal_pose)

planner.close()
env.flush_video()
env.close()
print("\nDone! Video saved → simulated_demo/pick_YCB_Clutter/")
