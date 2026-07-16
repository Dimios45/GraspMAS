"""
GraspMAS Evaluation — OCID-VLG Dataset (100 samples)
=====================================================
Evaluates GraspMAS on real-world cluttered indoor scenes from OCID-VLG.
Reads directly from test_expressions.json (no OCIDVLGDataset class).
Uses stratified sampling across 30 object categories for good representation.

Success metric (same as paper):
  IoU(predicted_grasp, best_gt_grasp) > 0.25  AND  |angle_diff| < 30°

Comparison baseline:
  GraspMAS (GPT-4o): 0.62 success rate on OCID-VLG (Table II of paper)

Usage:
  python scripts/run_ocidvlg_eval.py --root /path/to/ocid-vlg --n 100

Output: runs/YYYYMMDD_HHMMSS_ocidvlg_eval/
  results.json, results.csv, summary.txt, imgs/<idx>/
"""

import os, sys, json, csv, time, asyncio, warnings, argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graspmas import GraspMAS

API_FILE  = "api.key"
MAX_ROUND = 4


# ── grasp format conversion ───────────────────────────────────────────────────

def corners_to_params(corners):
    """
    Convert 4-corner grasp (Nx2 or list of 4 [x,y]) → (cx, cy, w, h, angle_deg).
    Uses cv2.minAreaRect convention; angle in [-90, 0) mapped to degrees.
    """
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    # cv2.minAreaRect returns angle in [-90, 0); ensure w >= h (gripper width > opening)
    if w < h:
        w, h = h, w
        angle = angle + 90
    return cx, cy, w, h, angle


def grasp_to_polygon(cx, cy, w, h, angle_deg):
    box = cv2.boxPoints(((cx, cy), (w, h), -(angle_deg + 180)))
    return box.astype(np.float32)


def polygon_iou(poly1, poly2):
    from shapely.geometry import Polygon as ShapelyPoly
    try:
        p1 = ShapelyPoly(poly1).convex_hull
        p2 = ShapelyPoly(poly2).convex_hull
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def angle_diff(a1, a2):
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)


def is_success(pred_grasp, gt_corners_list, iou_thresh=0.25, angle_thresh=30.0):
    """
    pred_grasp      : [quality, cx, cy, w, h, angle]
    gt_corners_list : list of N grasps, each [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    if pred_grasp is None or len(pred_grasp) < 6:
        return False, 0.0, 180.0

    _, pcx, pcy, pw, ph, pangle = pred_grasp
    pred_poly = grasp_to_polygon(pcx, pcy, pw, ph, pangle)

    best_iou   = 0.0
    best_adiff = 180.0

    for corners in gt_corners_list:
        gcx, gcy, gw, gh, gangle = corners_to_params(corners)
        gt_poly = grasp_to_polygon(gcx, gcy, gw, gh, gangle)
        iou     = polygon_iou(pred_poly, gt_poly)
        adiff   = angle_diff(pangle, gangle)
        if iou > best_iou:
            best_iou   = iou
            best_adiff = adiff

    success = (best_iou >= iou_thresh) and (best_adiff <= angle_thresh)
    return success, round(best_iou, 4), round(best_adiff, 2)


# ── stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(data, n_total, seed=42):
    """
    Sample n_total items from data with approximate stratification by object category.
    Category key: target.rsplit('_', 1)[0]   e.g. 'cereal_box_1' → 'cereal_box'
    Returns sorted list of (original_index, sample_dict).
    """
    rng = np.random.default_rng(seed)

    by_cat = defaultdict(list)
    for idx, s in enumerate(data):
        cat = s['target'].rsplit('_', 1)[0]
        by_cat[cat].append(idx)

    cats    = sorted(by_cat.keys())
    n_cats  = len(cats)
    base    = n_total // n_cats
    extra   = n_total  % n_cats

    chosen = []
    for i, cat in enumerate(cats):
        k    = base + (1 if i < extra else 0)
        pool = by_cat[cat]
        k    = min(k, len(pool))
        idxs = rng.choice(pool, size=k, replace=False).tolist()
        chosen.extend(idxs)

    # fill up to n_total if rounding left gaps
    if len(chosen) < n_total:
        all_idxs = list(range(len(data)))
        remaining = [i for i in all_idxs if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen += remaining[: n_total - len(chosen)]

    chosen = sorted(chosen)
    return [(i, data[i]) for i in chosen]


# ── main ─────────────────────────────────────────────────────────────────────

SPATIAL_WORDS = ['behind', 'front', 'right', 'left', 'nearest', 'distant',
                 'next to', 'top', 'bottom', 'most', 'rear', 'fore']

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",        required=True,  help="Path to OCID-VLG root directory")
    p.add_argument("--split",       default="test", help="Dataset split: train/val/test")
    p.add_argument("--version",     default="unique",
                   help="Dataset version: multiple/unique/novel-instances/novel-classes")
    p.add_argument("--n",           type=int, default=100, help="Number of samples to evaluate")
    p.add_argument("--seed",        type=int, default=42,  help="Random seed for sampling")
    p.add_argument("--categories",  default=None,
                   help="Comma-separated list of categories to include, e.g. orange,lime,peach")
    p.add_argument("--simple-only", action="store_true",
                   help="Only include queries without spatial relational words")
    return p.parse_args()


def main():
    args   = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"{run_id}_ocidvlg_eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"GraspMAS Evaluation — OCID-VLG")
    print(f"Root    : {args.root}")
    print(f"Split   : {args.split}  Version: {args.version}")
    print(f"Samples : {args.n}  Seed: {args.seed}")
    print(f"Output  : {out_dir}/\n")

    # ── load JSON directly ────────────────────────────────────────────────────
    json_path = os.path.join(args.root, "refer", args.version,
                             f"{args.split}_expressions.json")
    raw = json.load(open(json_path))
    all_data = raw['data']
    print(f"Dataset size ({args.split}/{args.version}): {len(all_data)}")

    # optional filters
    if args.categories:
        cats_filter = set(c.strip() for c in args.categories.split(','))
        all_data = [s for s in all_data
                    if s['target'].rsplit('_', 1)[0] in cats_filter]
        print(f"After category filter ({args.categories}): {len(all_data)}")

    if args.simple_only:
        all_data = [s for s in all_data
                    if not any(w in s['question'].lower() for w in SPATIAL_WORDS)]
        print(f"After simple-only filter: {len(all_data)}")

    samples = stratified_sample(all_data, args.n, seed=args.seed)
    print(f"Selected {len(samples)} samples across "
          f"{len(set(s['target'].rsplit('_',1)[0] for _, s in samples))} categories\n")

    graspmas = GraspMAS(api_file=API_FILE, max_round=MAX_ROUND)
    results  = []

    for rank, (orig_idx, sample) in enumerate(samples):
        # ── load image ───────────────────────────────────────────────────────
        rel_path = sample['image_filename'].replace(',', '/rgb/')
        img_path = os.path.join(args.root, rel_path)
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        query       = sample['question']
        target      = sample['target']
        gt_corners  = sample['grasps']          # list of [[x,y]×4]
        cat         = target.rsplit('_', 1)[0]

        img_dir = os.path.join(out_dir, "imgs", f"{rank:03d}")
        os.makedirs(img_dir, exist_ok=True)
        plt.imsave(os.path.join(img_dir, "input.png"), rgb)

        print(f"\n[{rank+1:3d}/{args.n}] idx={orig_idx}  cat={cat}  target={target}")
        print(f"  query: \"{query}\"  |  GT grasps: {len(gt_corners)}")

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

        detected             = grasp_pose is not None
        success, iou, adiff  = is_success(grasp_pose, gt_corners)

        print(f"  Detected: {detected}  Success: {success}  "
              f"IoU: {iou:.3f}  ΔAngle: {adiff:.1f}°  [{elapsed}s]")

        results.append({
            'rank':         rank,
            'orig_idx':     orig_idx,
            'category':     cat,
            'target':       target,
            'query':        query,
            'n_gt_grasps':  len(gt_corners),
            'detected':     detected,
            'success':      success,
            'best_iou':     iou,
            'angle_diff':   adiff,
            'grasp_quality': round(grasp_pose[0], 3) if grasp_pose else None,
            'grasp_cx':      round(grasp_pose[1], 2) if grasp_pose else None,
            'grasp_cy':      round(grasp_pose[2], 2) if grasp_pose else None,
            'grasp_w':       round(grasp_pose[3], 2) if grasp_pose else None,
            'grasp_h':       round(grasp_pose[4], 2) if grasp_pose else None,
            'grasp_angle':   round(grasp_pose[5], 2) if grasp_pose else None,
            'observer_summary': graspmas.observation,
            'error':         error,
            'elapsed_s':     elapsed,
        })

    # ── save ──────────────────────────────────────────────────────────────────
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
    n_detected = sum(r['detected'] for r in results)
    n_success  = sum(r['success']  for r in results)
    mean_iou   = np.mean([r['best_iou']  for r in results])
    avg_time   = np.mean([r['elapsed_s'] for r in results])

    # per-category breakdown
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r['category']].append(r)

    sep = "─" * 82
    lines = [
        "GraspMAS Evaluation — OCID-VLG",
        f"Run: {run_id}   Split: {args.split}   Version: {args.version}   N: {n}",
        sep,
        f"{'#':>4} {'Category':<20} {'Query':<30} {'Det':>4} {'Suc':>4} {'IoU':>6} {'ΔAng':>6} {'t':>5}",
        sep,
    ]
    for r in results:
        q = r['query'][:28] + '..' if len(r['query']) > 30 else r['query']
        lines.append(
            f"{r['rank']:>4} {r['category']:<20} {q:<30} "
            f"{'✓' if r['detected'] else '✗':>4} "
            f"{'✓' if r['success'] else '✗':>4} "
            f"{r['best_iou']:>6.3f} {r['angle_diff']:>6.1f} {r['elapsed_s']:>5.1f}"
        )
    lines += [
        sep,
        f"Samples evaluated  : {n}",
        f"Detection rate     : {n_detected}/{n} = {n_detected/n*100:.1f}%",
        f"Success rate       : {n_success}/{n} = {n_success/n*100:.1f}%  "
        f"(paper GPT-4o baseline: 62.0%)",
        f"Mean IoU           : {mean_iou:.3f}",
        f"Avg time/sample    : {avg_time:.1f}s",
        f"Total time         : {sum(r['elapsed_s'] for r in results)/60:.1f} min",
        "",
        "── Per-Category Breakdown ───────────────────────────────────────────────────",
    ]
    for cat, rows in sorted(by_cat.items()):
        nd = sum(r['detected'] for r in rows)
        ns = sum(r['success']  for r in rows)
        lines.append(
            f"  {cat:<25} n={len(rows):2d}  "
            f"det={nd/len(rows)*100:.0f}%  "
            f"succ={ns/len(rows)*100:.0f}%"
        )
    lines.append(sep)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), 'w') as f:
        f.write(summary)

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
