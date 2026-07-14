"""A/B evaluation harness — OCID-VLG.

Runs the same stratified query set through a chosen pipeline arm and records
accuracy (IoU-based success, same metric as the paper) plus efficiency
telemetry (LLM calls, tokens, wall time, rounds, failure taxonomy).

Arms:
  legacy — the unmodified GraspMAS Planner→Coder→Observer loop (baseline)
  gmas   — the OpenClaw-style runtime (added in P1; feature flags via --flags)

Usage (from repo root):
  python -m gmas.eval.run_ocidvlg_ab --root /mnt/data/mritunjoyh/datasets/ocid-vlg \
      --arm legacy --n 100 --seed 42 [--simple-only] [--categories a,b,c]

Output: runs/<ts>_ab_<arm>[_<tag>]/
  results.json, telemetry.jsonl, summary.txt, imgs/<idx>/
"""

import os, sys, json, time, asyncio, argparse, warnings
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from gmas.telemetry import Telemetry
from gmas.eval.instrumented import instrument, classify_failure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="OCID-VLG root directory")
    p.add_argument("--arm", default="legacy", choices=["legacy", "gmas"])
    p.add_argument("--flags", default="", help="gmas feature flags, comma-separated (P2/P3)")
    p.add_argument("--split", default="test")
    p.add_argument("--version", default="unique")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-round", type=int, default=4)
    p.add_argument("--categories", default=None)
    p.add_argument("--simple-only", action="store_true")
    p.add_argument("--tag", default="", help="suffix for the output dir name")
    p.add_argument("--api-file", default="api.key")
    return p.parse_args()


def build_pipeline(args):
    if args.arm == "legacy":
        from agents.graspmas import GraspMAS
        return instrument(GraspMAS(api_file=args.api_file, max_round=args.max_round))
    from gmas.agents.orchestrator import GmasPipeline
    from gmas.config import Flags
    return instrument(GmasPipeline(api_file=args.api_file, max_round=args.max_round,
                                   flags=Flags.from_csv(args.flags)))


def reset_pipeline(pipe):
    pipe.plan = None
    pipe.observation = None
    pipe.code = None


def pct(x, n):
    return f"{x}/{n} = {x / n * 100:.1f}%" if n else "n/a"


def main():
    args = parse_args()
    # reuse the metric + sampling helpers from the original eval script so
    # both arms are scored identically to the paper protocol
    import run_ocidvlg_eval as base

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = os.path.join(REPO_ROOT, "runs", f"{run_id}_ab_{args.arm}{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(args.root, "refer", args.version, f"{args.split}_expressions.json")
    all_data = json.load(open(json_path))["data"]
    if args.categories:
        cats = set(c.strip() for c in args.categories.split(","))
        all_data = [s for s in all_data if s["target"].rsplit("_", 1)[0] in cats]
    if args.simple_only:
        all_data = [s for s in all_data
                    if not any(w in s["question"].lower() for w in base.SPATIAL_WORDS)]
    samples = base.stratified_sample(all_data, args.n, seed=args.seed)

    cfg = vars(args).copy()
    print(f"A/B eval — arm={args.arm}  n={len(samples)}  out={out_dir}")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    pipe = build_pipeline(args)
    results = []
    tel_file = open(os.path.join(out_dir, "telemetry.jsonl"), "w")

    for rank, (orig_idx, sample) in enumerate(samples):
        rel_path = sample["image_filename"].replace(",", "/rgb/")
        rgb = cv2.cvtColor(cv2.imread(os.path.join(args.root, rel_path)), cv2.COLOR_BGR2RGB)
        query, target, gt_corners = sample["question"], sample["target"], sample["grasps"]
        cat = target.rsplit("_", 1)[0]

        img_dir = os.path.join(out_dir, "imgs", f"{rank:03d}")
        os.makedirs(img_dir, exist_ok=True)
        plt.imsave(os.path.join(img_dir, "input.png"), rgb)

        print(f"\n[{rank + 1:3d}/{len(samples)}] idx={orig_idx} cat={cat} query=\"{query}\"")
        reset_pipeline(pipe)

        tel = Telemetry.start(meta={"rank": rank, "orig_idx": orig_idx, "query": query})
        t0 = time.time()
        try:
            _, grasp_pose = asyncio.run(pipe.query(query, rgb, save_folder=img_dir))
            error = None
        except Exception as e:
            grasp_pose, error = None, repr(e)
            print(f"  ERROR: {e}")
        elapsed = round(time.time() - t0, 1)
        tel = Telemetry.stop()
        ts = tel.summary()
        tel_file.write(json.dumps({"rank": rank, **tel.dump()}, default=str) + "\n")
        tel_file.flush()

        success, iou, adiff = base.is_success(grasp_pose, gt_corners)
        failure = classify_failure(error, grasp_pose, success, iou, ts)
        elapsed_net = round(elapsed - ts["vlm_load_s"], 1)

        print(f"  success={success} iou={iou:.3f} Δang={adiff:.1f} "
              f"failure={failure} rounds={ts['rounds']} llm_calls={ts['llm_calls']} "
              f"tokens={ts['input_tokens']}+{ts['output_tokens']} t={elapsed_net}s")

        results.append({
            "rank": rank, "orig_idx": orig_idx, "category": cat, "target": target,
            "query": query, "n_gt_grasps": len(gt_corners),
            "detected": grasp_pose is not None, "success": success,
            "best_iou": iou, "angle_diff": adiff, "failure": failure,
            "grasp": [round(float(v), 3) for v in grasp_pose] if grasp_pose else None,
            "error": error, "elapsed_s": elapsed, "elapsed_net_s": elapsed_net,
            **{k: ts[k] for k in ("llm_calls", "llm_wall_s", "input_tokens",
                                  "output_tokens", "rounds", "vlm_load_s")},
            "llm_per_tag": ts["llm_per_tag"],
        })
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    tel_file.close()

    # ── summary ───────────────────────────────────────────────────────────
    n = len(results)
    succ = sum(r["success"] for r in results)
    det = sum(r["detected"] for r in results)
    times = np.array([r["elapsed_net_s"] for r in results])
    fail_counts = Counter(r["failure"] for r in results)
    rounds_hist = Counter(r["rounds"] for r in results)

    per_tag = defaultdict(lambda: {"calls": 0, "wall_s": 0.0})
    for r in results:
        for tag, d in r["llm_per_tag"].items():
            per_tag[tag]["calls"] += d["calls"]
            per_tag[tag]["wall_s"] += d["wall_s"]

    lines = [
        f"A/B eval — arm={args.arm} flags={args.flags or '-'} n={n} seed={args.seed} "
        f"version={args.version} simple_only={args.simple_only}",
        "─" * 78,
        f"Success rate       : {pct(succ, n)}",
        f"Detection rate     : {pct(det, n)}",
        f"Mean IoU           : {np.mean([r['best_iou'] for r in results]):.3f}",
        f"Wall time /query   : mean {times.mean():.1f}s  median {np.median(times):.1f}s  "
        f"p90 {np.percentile(times, 90):.1f}s",
        f"LLM calls /query   : {np.mean([r['llm_calls'] for r in results]):.2f}",
        f"LLM tokens /query  : in {np.mean([r['input_tokens'] for r in results]):.0f}  "
        f"out {np.mean([r['output_tokens'] for r in results]):.0f}",
        f"Rounds /query      : {np.mean([r['rounds'] for r in results]):.2f}  "
        f"hist {dict(sorted(rounds_hist.items()))}",
        "",
        "LLM time by agent:",
    ]
    for tag, d in sorted(per_tag.items()):
        lines.append(f"  {tag:<10} calls={d['calls']:<5} wall={d['wall_s']:.0f}s")
    lines += ["", "Failure taxonomy:"]
    for k, v in fail_counts.most_common():
        lines.append(f"  {k:<22} {pct(v, n)}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
