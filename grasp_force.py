"""
Ground-truth physics extraction and gripper force computation.

Physics model (parallel-jaw gripper, quasi-static):
  - Two finger pads each apply normal force F_n
  - Friction force per pad: f = mu * F_n
  - Anti-gravity condition: 2 * mu * F_n >= m * g
  - Minimum force: F_n_min = m * g / (2 * mu)
  - With safety factor k: F_n = k * m * g / (2 * mu)

Franka Panda gripper specs:
  - Max continuous force per finger: ~70 N
  - Peak force: ~140 N
"""

import numpy as np
import torch

GRAVITY = 9.81
PANDA_MAX_FINGER_FORCE_N = 70.0   # conservative continuous rating


def _get_named_target(env_unwrapped):
    """Return the actual named Actor that is the current target object."""
    tgt_pos = env_unwrapped.target_object.pose.sp.p
    for o in env_unwrapped.selectable_target_objects[0]:
        if np.allclose(o.pose.sp.p, tgt_pos, atol=1e-3):
            return o
    return None


def _get_friction(actor):
    """Read static friction from the first PhysX collision shape."""
    for c in actor._objs[0].get_components():
        if "Physx" in type(c).__name__:
            shapes = c.get_collision_shapes()
            if shapes:
                mat = shapes[0].get_physical_material()
                return mat.static_friction, mat.dynamic_friction
    return 0.3, 0.3   # ManiSkill default fallback


def get_object_physics(env_unwrapped):
    """
    Extract ground-truth physical properties of the current target object.

    Returns dict:
        name            str    — object class name (e.g. 'lemon')
        full_name       str    — full actor name (e.g. 'set_0_014_lemon')
        mass_kg         float
        static_friction float
        dynamic_friction float
        weight_N        float  — m * g
    """
    actor = _get_named_target(env_unwrapped)
    if actor is None:
        raise RuntimeError("Could not locate named target object in selectable list")

    mass = float(actor.get_mass())
    mu_s, mu_d = _get_friction(actor)
    full_name = actor.name                          # set_0_014_lemon
    name = full_name.split("_", 3)[-1].replace("_", " ")  # master chef can

    return {
        "name":             name,
        "full_name":        full_name,
        "mass_kg":          mass,
        "static_friction":  mu_s,
        "dynamic_friction": mu_d,
        "weight_N":         mass * GRAVITY,
    }


def compute_required_force(physics: dict, safety_factor: float = 2.0) -> dict:
    """
    Compute the minimum gripper finger force to hold the object.

    Returns dict:
        F_min_N        float  — minimum finger force (no safety margin)
        F_required_N   float  — with safety factor applied
        safety_factor  float
        feasible       bool   — True if F_required <= Panda max
        utilization    float  — F_required / PANDA_MAX (0..1+)
    """
    m   = physics["mass_kg"]
    mu  = physics["static_friction"]
    g   = GRAVITY

    F_min      = m * g / (2.0 * mu)
    F_required = safety_factor * F_min

    return {
        "F_min_N":       round(F_min, 4),
        "F_required_N":  round(F_required, 4),
        "safety_factor": safety_factor,
        "feasible":      F_required <= PANDA_MAX_FINGER_FORCE_N,
        "utilization":   round(F_required / PANDA_MAX_FINGER_FORCE_N, 3),
    }


def get_contact_forces(env_unwrapped) -> dict:
    """
    Read net contact forces on the target object after grasping.
    Call this after close_gripper() to validate the grasp.

    Returns dict:
        force_vec_N    np.ndarray (3,)  — net contact force vector [x,y,z]
        force_mag_N    float            — magnitude
        lifted         bool             — z-component > object weight (rough check)
    """
    obj = env_unwrapped.target_object
    forces = obj.get_net_contact_forces()           # shape (num_envs, 3)
    if isinstance(forces, torch.Tensor):
        forces = forces.cpu().numpy()
    fvec = forces[0]                                # first (only) env
    fmag = float(np.linalg.norm(fvec))
    weight = float(obj.get_mass()) * GRAVITY

    return {
        "force_vec_N": fvec,
        "force_mag_N": round(fmag, 4),
        "lifted":      fvec[2] > weight * 0.5,     # z-contact > 50% of weight
    }


def print_force_report(physics: dict, force_req: dict, contact: dict = None):
    print("\n===== Grasp Force Analysis =====")
    print(f"  Object        : {physics['name']} ({physics['full_name']})")
    print(f"  Mass          : {physics['mass_kg']*1000:.1f} g  ({physics['weight_N']:.3f} N)")
    print(f"  Friction (μ_s): {physics['static_friction']:.2f}")
    print(f"  F_min         : {force_req['F_min_N']:.3f} N  (no safety margin)")
    print(f"  F_required    : {force_req['F_required_N']:.3f} N  (×{force_req['safety_factor']} safety)")
    print(f"  Panda max     : {PANDA_MAX_FINGER_FORCE_N:.1f} N")
    print(f"  Utilization   : {force_req['utilization']*100:.1f}%  → {'OK' if force_req['feasible'] else 'EXCEEDS LIMIT'}")
    if contact:
        print(f"  Contact force : {contact['force_mag_N']:.3f} N  (post-grasp)")
        print(f"  Lifted        : {contact['lifted']}")
    print("================================\n")
