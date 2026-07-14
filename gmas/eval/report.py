"""Comparison report across A/B eval runs.

Usage:
  python -m gmas.eval.report runs/<ts>_ab_legacy_p0_baseline [runs/<ts>_ab_gmas_* ...]

Reads results.json + telemetry.jsonl from each run dir, re-derives the
failure taxonomy from raw telemetry (labels are recomputed here so fixes to
the classifier apply retroactively), and prints one column per run.
"""

import json
import os
import sys
from collections import Counter

import numpy as np


def _reclassify(row, tel):
    """Failure label from raw telemetry — mirrors instrumented.classify_failure
    but applied post-hoc (handles the error_logs == "None" detection-miss case)."""
    if row["success"]:
        return "success"
    events = tel["events"]
    code_errors = [e for e in events
                   if e["kind"] == "code_runtime_error" and str(e.get("error")) != "None"]
    if row.get("error"):
        e = row["error"].lower()
        if "syntaxerror" in e or "invalid syntax" in e or "outside function" in e \
                or "indentation" in e or "never closed" in e:
            return "code_syntax_error"
        if any(ev["kind"] in ("planner_error", "observer_error") for ev in events):
            return "agent_parse_error"
        if "list index out of range" in e or "jsondecode" in e or "expecting value" in e:
            return "agent_parse_error"
        return "other_error"
    if not row["detected"]:
        return "code_runtime_error" if code_errors else "no_detection"
    if row["best_iou"] == 0.0:
        return "wrong_location"
    return "bad_grasp"


def load_run(run_dir):
    results = json.load(open(os.path.join(run_dir, "results.json")))
    tels = {}
    tel_path = os.path.join(run_dir, "telemetry.jsonl")
    if os.path.exists(tel_path):
        for line in open(tel_path):
            t = json.loads(line)
            tels[t["rank"]] = t
    for r in results:
        t = tels.get(r["rank"], {"events": []})
        r["failure"] = _reclassify(r, t)
        r["_events"] = t["events"]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    return cfg, results


def col(run_dir):
    cfg, rs = load_run(run_dir)
    n = len(rs)
    times = np.array([r["elapsed_net_s"] for r in rs])
    fails = Counter(r["failure"] for r in rs)
    code_err_rounds = sum(
        sum(1 for e in r["_events"] if e["kind"] == "code_runtime_error"
            and str(e.get("error")) != "None")
        for r in rs)
    name = f"{cfg['arm']}" + (f"[{cfg['flags']}]" if cfg.get("flags") else "")
    return name, {
        "n": n,
        "success %": 100 * sum(r["success"] for r in rs) / n,
        "detected %": 100 * sum(r["detected"] for r in rs) / n,
        "mean IoU": float(np.mean([r["best_iou"] for r in rs])),
        "median t (s)": float(np.median(times)),
        "p90 t (s)": float(np.percentile(times, 90)),
        "LLM calls/q": float(np.mean([r["llm_calls"] for r in rs])),
        "tokens in/q": float(np.mean([r["input_tokens"] for r in rs])),
        "tokens out/q": float(np.mean([r["output_tokens"] for r in rs])),
        "rounds/q": float(np.mean([r["rounds"] for r in rs])),
        "code-error rounds": code_err_rounds,
        **{f"fail:{k}": v for k, v in sorted(fails.items())},
    }


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    cols = [col(r) for r in runs]
    keys = []
    for _, c in cols:
        for k in c:
            if k not in keys:
                keys.append(k)
    w = max(len(k) for k in keys) + 2
    header = " " * w + "".join(f"{name:>24}" for name, _ in cols)
    print(header)
    print("─" * len(header))
    for k in keys:
        cells = []
        for _, c in cols:
            v = c.get(k, "")
            cells.append(f"{v:>24.2f}" if isinstance(v, float) else f"{v:>24}")
        print(f"{k:<{w}}" + "".join(cells))


if __name__ == "__main__":
    main()
