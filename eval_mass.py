"""
eval_mass.py — 5-fold cross-validation mass evaluation for YCB objects.

Usage
-----
python -u eval_mass.py [options]

Options
-------
--n-samples N         Self-consistency samples per object  (default: 5)
--temperature T       LLM sampling temperature             (default: 0.9)
--k-anchors K         Retrieval anchor count               (default: 5)
--no-retrieval        Random anchors instead of NN          (ablation A)
--direct              Skip chain-of-thought decomp          (ablation B)
--no-anchor-blend     Skip geometric post-blend             (ablation C)
--sim-density-hint    Tell model about sim's simplified densities
--with-vision         Attach object RGB crop when available
--image-dir PATH      Root dir with <obj_id>/input.png images
--out-dir PATH        Output directory (default: runs/<ts>_mass_eval)
--limit N             First N objects only (quick smoke test)

ManiSkill mass-readout API
--------------------------
mass = density × trimesh.load("models/<obj_id>/collision.ply").convex_hull.volume × scale³
density and scale from info_pick_v0.json.
All 78 YCB objects: nonzero masses, range 0.005–1.37 kg.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from property_inference import (
    MassPredictor,
    build_gt_table,
    label_for,
    load_image_b64,
    _VLM_AVAILABLE,
)

_DEFAULT_IMG_DIR = Path("runs/20260522_215724_graspmas_single/imgs")


# ── 5-fold split ──────────────────────────────────────────────────────────────
def make_folds(obj_ids: list[str], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    rng   = np.random.default_rng(seed)
    ids   = rng.permutation(obj_ids).tolist()
    size  = len(ids) // n_folds
    folds = [ids[i * size: (i + 1) * size] for i in range(n_folds)]
    for j, extra in enumerate(ids[n_folds * size:]):
        folds[j].append(extra)
    return folds


# ── Scatter plot ──────────────────────────────────────────────────────────────
def save_scatter(results: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok    = [r for r in results if r["pred_kg"] is not None]
    tvs   = [r["true_kg"]  for r in ok]
    pvs   = [r["pred_kg"]  for r in ok]
    labs  = [label_for(r["obj_id"]) for r in ok]
    confs = [r["confidence"] for r in ok]

    lo = min(min(tvs), min(pvs)) * 0.85
    hi = max(max(tvs), max(pvs)) * 1.15

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="perfect")

    sc = ax.scatter(tvs, pvs, c=confs, cmap="RdYlGn", vmin=0, vmax=1,
                    alpha=0.8, edgecolors="grey", s=70)
    plt.colorbar(sc, ax=ax, label="confidence")

    for tv, pv, lb in zip(tvs, pvs, labs):
        if abs(pv - tv) / max(tv, 1e-6) > 0.6:
            ax.annotate(lb, (tv, pv), fontsize=6, alpha=0.75,
                        xytext=(4, 4), textcoords="offset points")

    mae  = np.mean([abs(p - t) for t, p in zip(tvs, pvs)])
    mape = np.mean([abs(p - t) / max(t, 1e-6) for t, p in zip(tvs, pvs)]) * 100
    ax.set_xlabel("True mass (kg)", fontsize=12)
    ax.set_ylabel("Predicted mass (kg)", fontsize=12)
    ax.set_title("YCB Mass: True vs Predicted", fontsize=13)
    ax.legend(fontsize=10)
    ax.text(0.05, 0.95,
            f"MAE={mae:.4f} kg  MAPE={mape:.1f}%\nn={len(ok)}/{len(results)} detected",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


# ── Main eval loop ────────────────────────────────────────────────────────────
def run_eval(args: argparse.Namespace) -> list[dict]:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"runs/{ts}_mass_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.image_dir) if args.image_dir else _DEFAULT_IMG_DIR

    backend = "local vlm_chat (graspmas env)" if _VLM_AVAILABLE else "offline stub"
    print("=" * 62)
    print("GraspMAS — YCB Mass Evaluation")
    print("=" * 62)
    print(f"Output dir      : {out_dir}")
    print(f"Backend         : {backend}")
    print(f"Retrieval       : {'NN embedding' if not args.no_retrieval else 'random (ablation)'}")
    print(f"Prompt          : {'chain-of-thought decomp' if not args.direct else 'direct (ablation)'}")
    print(f"Anchor blend    : {'ON' if not args.no_anchor_blend else 'OFF (ablation)'}")
    print(f"Sim density hint: {'ON' if args.sim_density_hint else 'off'}")
    print(f"Vision          : {'ON' if args.with_vision else 'off'}")
    print(f"Samples / temp  : {args.n_samples} / {args.temperature}")
    print(f"k anchors       : {args.k_anchors}")
    print()

    print("Loading ground-truth mass table (trimesh + info_pick_v0.json)...")
    gt_table = build_gt_table()
    print(f"  {len(gt_table)} objects, {min(gt_table.values()):.4f}–{max(gt_table.values()):.4f} kg")
    print()

    all_ids = sorted(gt_table.keys())
    if args.limit:
        all_ids = all_ids[: args.limit]
        print(f"[--limit {args.limit}] evaluating first {args.limit} objects only")

    predictor = MassPredictor(
        gt_table         = gt_table,
        n_samples        = args.n_samples,
        temperature      = args.temperature,
        k_anchors        = args.k_anchors,
        use_retrieval    = not args.no_retrieval,
        use_decomp       = not args.direct,
        use_anchor_blend = not args.no_anchor_blend,
        sim_density_hint = args.sim_density_hint,
    )

    folds   = make_folds(all_ids, n_folds=5)
    results: list[dict] = []
    trace_path = out_dir / "trace.jsonl"

    print(f"Running 5-fold cross-validation on {len(all_ids)} objects...\n")

    for fold_idx, fold_ids in enumerate(folds):
        exclude_set = set(fold_ids)
        for obj_id in fold_ids:
            t0      = time.time()
            true_kg = gt_table[obj_id]

            image_b64: Optional[str] = None
            if args.with_vision:
                image_b64 = load_image_b64(img_dir / obj_id / "input.png")

            pred_kg, conf, raw_samples, raw_texts = predictor.predict(
                obj_id,
                image_b64   = image_b64,
                exclude_ids = exclude_set - {obj_id},
            )

            elapsed  = time.time() - t0
            abs_err  = abs(pred_kg - true_kg) if pred_kg is not None else None
            pct_err  = (abs_err / max(true_kg, 1e-6) * 100) if abs_err is not None else None

            row = dict(
                fold       = fold_idx,
                obj_id     = obj_id,
                label      = label_for(obj_id),
                true_kg    = true_kg,
                pred_kg    = pred_kg,
                abs_err    = abs_err,
                pct_err    = pct_err,
                confidence = conf,
                n_valid    = sum(1 for s in raw_samples if s is not None),
                elapsed_s  = round(elapsed, 2),
            )
            results.append(row)

            with open(trace_path, "a") as tf:
                tf.write(json.dumps(dict(
                    obj_id     = obj_id,
                    label      = label_for(obj_id),
                    true_kg    = true_kg,
                    pred_kg    = pred_kg,
                    raw_samples= raw_samples,
                    raw_texts  = raw_texts,
                    confidence = conf,
                    elapsed_s  = elapsed,
                )) + "\n")

            if pred_kg is None:
                status = "FAILED (no valid prediction)"
            else:
                status = f"{pred_kg:.4f} kg  err={pct_err:.1f}%  conf={conf:.2f}"
            print(f"  [{fold_idx+1}/5] {obj_id:<36} true={true_kg:.4f}  {status}  t={elapsed:.1f}s")

    print()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    detected = [r for r in results if r["pred_kg"] is not None]
    failed   = [r for r in results if r["pred_kg"] is None]
    n_total  = len(results)
    n_det    = len(detected)

    mae_kg   = float(np.mean([r["abs_err"] for r in detected])) if detected else float("nan")
    mape_pct = float(np.mean([r["pct_err"] for r in detected])) if detected else float("nan")
    med_pct  = float(np.median([r["pct_err"] for r in detected])) if detected else float("nan")

    print("=" * 62)
    print(f"RESULTS  ({n_det}/{n_total} detected, {100*n_det/n_total:.1f}%)")
    print("=" * 62)
    print(f"  MAE_kg        = {mae_kg:.4f} kg")
    print(f"  MAPE_pct      = {mape_pct:.1f} %")
    print(f"  Median err    = {med_pct:.1f} %")
    if detected:
        in25  = sum(1 for r in detected if r["pct_err"] <  25)
        in50  = sum(1 for r in detected if r["pct_err"] <  50)
        in100 = sum(1 for r in detected if r["pct_err"] < 100)
        print(f"  <25%  err     = {in25}/{n_det} ({100*in25/n_det:.1f}%)")
        print(f"  <50%  err     = {in50}/{n_det} ({100*in50/n_det:.1f}%)")
        print(f"  <100% err     = {in100}/{n_det} ({100*in100/n_det:.1f}%)")
        print(f"  Avg conf      = {np.mean([r['confidence'] for r in detected]):.2f}")
    print()

    if failed:
        print(f"Failed objects ({len(failed)}):")
        for r in failed:
            print(f"  {r['label']:<30} true={r['true_kg']:.4f} kg")
        print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path  = out_dir / "mass_results.csv"
    fields    = ["fold","obj_id","label","true_kg","pred_kg",
                 "abs_err","pct_err","confidence","n_valid","elapsed_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"CSV          → {csv_path}")

    plot_path = out_dir / "true_vs_pred.png"
    save_scatter(results, plot_path)
    print(f"Scatter plot → {plot_path}")
    print(f"Trace log    → {trace_path}")
    print()

    if detected:
        worst = sorted(detected, key=lambda r: r["pct_err"], reverse=True)[:10]
        best  = sorted(detected, key=lambda r: r["pct_err"])[:10]
        print(f"{'Best 10':<30} {'True':>8} {'Pred':>8} {'Err%':>7}")
        for r in best:
            print(f"  {r['label']:<28} {r['true_kg']:>8.4f} {r['pred_kg']:>8.4f} {r['pct_err']:>7.1f}%")
        print()
        print(f"{'Worst 10':<30} {'True':>8} {'Pred':>8} {'Err%':>7}")
        for r in worst:
            print(f"  {r['label']:<28} {r['true_kg']:>8.4f} {r['pred_kg']:>8.4f} {r['pct_err']:>7.1f}%")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate YCB mass prediction via local Qwen2-VL-7B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-samples",        type=int,   default=5)
    p.add_argument("--temperature",      type=float, default=0.9)
    p.add_argument("--k-anchors",        type=int,   default=5)
    p.add_argument("--no-retrieval",     action="store_true")
    p.add_argument("--direct",           action="store_true")
    p.add_argument("--no-anchor-blend",  action="store_true")
    p.add_argument("--sim-density-hint", action="store_true")
    p.add_argument("--with-vision",      action="store_true")
    p.add_argument("--image-dir",        default=None)
    p.add_argument("--out-dir",          default=None)
    p.add_argument("--limit",            type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_eval(_parse_args())
