"""A/B evaluation harness — 49-YCB grasp-detection stage.

Phase 1 (--render): render each of the 49 YCB single-object ManiSkill scenes
once (seeded) into a shared image cache, so both arms see byte-identical
inputs. Phase 2: run the 4-direction prompts of run_all_objects_demo.py
through the chosen arm on the cached renders and record detection success,
approach-direction correctness, and telemetry. No simulation rollout — this
is the grasp-detection stage of the protocol (the sim lift check lives in
run_all_objects_demo.py).

Usage (from repo root):
  python -m gmas.eval.run_ycb_ab --render                 # once, needs ManiSkill
  python -m gmas.eval.run_ycb_ab --arm legacy
  python -m gmas.eval.run_ycb_ab --arm gmas --flags all,timeout_s=300
"""

import os, sys, json, time, asyncio, argparse, warnings
from datetime import datetime
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

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

CACHE_DIR = os.path.join(REPO_ROOT, "object_demo", "ycb_render_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true",
                   help="phase 1: render the 49 scenes into the shared cache")
    p.add_argument("--arm", default="legacy", choices=["legacy", "gmas"])
    p.add_argument("--flags", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-round", type=int, default=5)
    p.add_argument("--tag", default="")
    p.add_argument("--api-file", default="api.key")
    return p.parse_args()


def render_cache(seed):
    """Render each pinned single-object scene once (same setup as
    run_all_objects_demo.py, minus video recording)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mani_skill.envs                      # noqa: F401
    import gymnasium as gym
    from mani_skill.utils.registration import register_env
    from mani_skill_pick_YCB.pick_single_ycb import PickSingleYCBEnv

    state = {"obj_id": VALID_OBJECTS[0]}

    @register_env("PickSingleYCB_abcache-v1", max_episode_steps=50,
                  asset_download_ids=["ycb"])
    class _PinnedEnv(PickSingleYCBEnv):
        def _load_scene(self, options: dict):
            saved = self.all_model_ids
            self.all_model_ids = np.array([state["obj_id"]])
            super()._load_scene(options)
            self.all_model_ids = saved

    os.makedirs(CACHE_DIR, exist_ok=True)
    for i, obj_id in enumerate(VALID_OBJECTS):
        out = os.path.join(CACHE_DIR, f"{obj_id}.png")
        if os.path.exists(out):
            print(f"[{i+1:2d}/49] {obj_id} — cached")
            continue
        state["obj_id"] = obj_id
        env = gym.make("PickSingleYCB_abcache-v1", obs_mode="rgbd",
                       control_mode="pd_joint_pos", render_mode="rgb_array",
                       sensor_configs=dict(shader_pack="default", width=384, height=384),
                       sim_backend="cpu", enable_shadow=True)
        obs, _ = env.reset(seed=seed)
        rgb = obs["sensor_data"]["base_camera"]["rgb"].cpu().squeeze().numpy()
        plt.imsave(out, rgb.astype(np.uint8))
        env.close()
        print(f"[{i+1:2d}/49] {obj_id} — rendered")
    print(f"cache complete: {CACHE_DIR}")


def build_pipeline(args):
    from gmas.eval.instrumented import instrument
    if args.arm == "legacy":
        from agents.graspmas import GraspMAS
        return instrument(GraspMAS(api_file=args.api_file, max_round=args.max_round))
    from gmas.agents.orchestrator import GmasPipeline
    from gmas.config import Flags
    return instrument(GmasPipeline(api_file=args.api_file, max_round=args.max_round,
                                   flags=Flags.from_csv(args.flags)))


def main():
    args = parse_args()
    if args.render:
        render_cache(args.seed)
        return

    import cv2
    from gmas.telemetry import Telemetry

    missing = [o for o in VALID_OBJECTS
               if not os.path.exists(os.path.join(CACHE_DIR, f"{o}.png"))]
    if missing:
        sys.exit(f"{len(missing)} renders missing (e.g. {missing[0]}) — "
                 f"run with --render first")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = os.path.join(REPO_ROOT, "runs", f"{run_id}_ycb_{args.arm}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "protocol": "ycb49_detection"}, f, indent=2)

    pipe = build_pipeline(args)
    results = []
    tel_file = open(os.path.join(out_dir, "telemetry.jsonl"), "w")

    n_total = len(VALID_OBJECTS) * 4
    k = 0
    for obj_id in VALID_OBJECTS:
        obj_name = obj_id.split("_", 1)[1].replace("_", " ")
        rgb = cv2.cvtColor(cv2.imread(os.path.join(CACHE_DIR, f"{obj_id}.png")),
                           cv2.COLOR_BGR2RGB)
        for tag, query in [
            ("top",    f"Pick the {obj_name} up from the top"),
            ("right",  f"Pick the {obj_name} up from the right hand side"),
            ("left",   f"Pick the {obj_name} up from the left hand side"),
            ("bottom", f"Pick the {obj_name} up from the bottom part"),
        ]:
            k += 1
            img_dir = os.path.join(out_dir, "imgs", f"{obj_id}_{tag}")
            os.makedirs(img_dir, exist_ok=True)
            pipe.plan = pipe.observation = pipe.code = None

            print(f"\n[{k:3d}/{n_total}] {obj_id} [{tag}]")
            tel = Telemetry.start(meta={"obj_id": obj_id, "direction": tag})
            t0 = time.time()
            try:
                _, grasp = asyncio.run(pipe.query(query, rgb, save_folder=img_dir))
                error = None
            except Exception as e:
                grasp, error = None, repr(e)
                print(f"  ERROR: {e}")
            elapsed = round(time.time() - t0, 1)
            tel = Telemetry.stop()
            ts = tel.summary()
            tel_file.write(json.dumps({"obj_id": obj_id, "direction": tag,
                                       **tel.dump()}, default=str) + "\n")
            tel_file.flush()

            approach = getattr(pipe, "last_approach", "center")
            detected = grasp is not None
            results.append({
                "obj_id": obj_id, "direction": tag, "query": query,
                "detected": detected,
                "approach": approach,
                "approach_correct": detected and approach == tag,
                "grasp": [round(float(v), 3) for v in grasp] if grasp else None,
                "error": error, "elapsed_s": elapsed,
                "elapsed_net_s": round(elapsed - ts["vlm_load_s"], 1),
                **{key: ts[key] for key in ("llm_calls", "input_tokens",
                                            "output_tokens", "rounds", "vlm_load_s")},
            })
            print(f"  detected={detected} approach={approach} "
                  f"rounds={ts['rounds']} t={results[-1]['elapsed_net_s']}s")
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
    tel_file.close()

    n = len(results)
    det = sum(r["detected"] for r in results)
    ok_dir = sum(r["approach_correct"] for r in results)
    times = np.array([r["elapsed_net_s"] for r in results])
    err = sum(1 for r in results if r["error"])
    lines = [
        f"49-YCB detection A/B — arm={args.arm} flags={args.flags or '-'} seed={args.seed}",
        "─" * 70,
        f"Detection rate       : {det}/{n} = {det/n*100:.1f}%",
        f"Approach correct     : {ok_dir}/{n} = {ok_dir/n*100:.1f}%",
        f"Query-level errors   : {err}/{n}",
        f"Wall time /query     : mean {times.mean():.1f}s  median {np.median(times):.1f}s  "
        f"p90 {np.percentile(times, 90):.1f}s",
        f"LLM calls /query     : {np.mean([r['llm_calls'] for r in results]):.2f}",
        f"Rounds /query        : {np.mean([r['rounds'] for r in results]):.2f}",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
