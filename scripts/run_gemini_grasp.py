"""
Gemini Direct Grasp Prediction
================================
Baseline comparison for GraspMAS: sends the scene RGB image directly to
Gemini 2.5 Flash and asks it to predict a parallel-jaw grasp in pixel coords.

No GraspNet, no GroundingDINO, no SAM, no LLM planning loop.
Single API call per image — pure vision-LLM baseline.

Can be run on any image saved by scripts/run_graspmas_eval.py or scripts/run_ocidvlg_eval.py.

Usage:
  # Single image:
  python scripts/run_gemini_grasp.py --image runs/.../imgs/011_banana/input.png --object banana

  # Batch over a full graspmas eval run (reads all input.png files):
  python scripts/run_gemini_grasp.py --run_dir runs/20260522_215724_graspmas_single/

Output: gemini_results.json in the run directory (batch) or stdout (single).
"""

import os, sys, json, re, time, argparse, base64, warnings
from io import BytesIO

warnings.filterwarnings('ignore')

import numpy as np
import requests
from PIL import Image as PILImage

GEMINI_KEY_FILE = "gemini.key"
GEMINI_MODEL    = "gemini-2.5-flash"


def rgb_to_b64(rgb: np.ndarray) -> str:
    buf = BytesIO()
    PILImage.fromarray(rgb).save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def gemini_grasp(rgb: np.ndarray, object_name: str, key: str) -> dict:
    """
    Single Gemini API call: image + object name → grasp prediction.

    Returns dict with keys:
      detected  (bool)
      cx, cy    grasp centre in pixels
      w, h      jaw width and contact depth in pixels
      angle     jaw axis angle in degrees (0=horizontal, 90=vertical)
      elapsed_s time taken
      error     error string or None
    """
    H, W = rgb.shape[:2]
    prompt = (
        f"You are a robot grasp planner. The image shows a {W}×{H} pixel scene "
        f"from a robot arm camera above a table.\n"
        f"Task: predict ONE 2D parallel-jaw grasp rectangle to pick up the '{object_name}'.\n\n"
        f"Return ONLY valid JSON — no explanation, no markdown:\n"
        f'{{\n'
        f'  "cx": <integer — grasp centre x in pixels from left>,\n'
        f'  "cy": <integer — grasp centre y in pixels from top>,\n'
        f'  "w":  <integer — gripper jaw opening width in pixels>,\n'
        f'  "h":  <integer — finger contact depth, typically 30-50% of w>,\n'
        f'  "angle": <float — jaw axis angle in degrees: 0=horizontal, 90=vertical>\n'
        f'}}\n\n'
        f"If '{object_name}' is not visible: "
        f'{{"cx": null, "cy": null, "w": null, "h": null, "angle": null}}'
    )

    payload = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg", "data": rgb_to_b64(rgb)}},
        {"text": prompt},
    ]}]}

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={key}")

    t0 = time.time()
    try:
        r       = requests.post(url, json=payload, timeout=60)
        elapsed = round(time.time() - t0, 1)

        if r.status_code != 200:
            msg = r.json().get("error", {}).get("message", "unknown")[:200]
            return {"detected": False, "error": msg, "elapsed_s": elapsed}

        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        m   = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            return {"detected": False, "error": f"no JSON in response: {raw[:80]}",
                    "elapsed_s": elapsed}

        g = json.loads(m.group())
        if g.get("cx") is None:
            return {"detected": False, "error": None, "elapsed_s": elapsed}

        return {
            "detected":  True,
            "cx":        g["cx"],   "cy":    g["cy"],
            "w":         g["w"],    "h":     g["h"],
            "angle":     g["angle"],
            "elapsed_s": elapsed,   "error": None,
        }
    except Exception as e:
        return {"detected": False, "error": str(e),
                "elapsed_s": round(time.time()-t0, 1)}


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",   help="Path to a single scene RGB image")
    g.add_argument("--run_dir", help="Path to a graspmas eval run dir (batch mode)")
    p.add_argument("--object",  help="Object name (required for --image mode)")
    p.add_argument("--delay",   type=float, default=6.0,
                   help="Seconds between requests in batch mode (default 6 = 10 RPM safe)")
    p.add_argument("--limit",   type=int, default=None,
                   help="Max number of objects in batch mode (default: all)")
    args = p.parse_args()

    if not os.path.exists(GEMINI_KEY_FILE):
        print(f"ERROR: {GEMINI_KEY_FILE} not found"); sys.exit(1)
    key = open(GEMINI_KEY_FILE).read().strip()

    # ── single image mode ────────────────────────────────────────────────────
    if args.image:
        if not args.object:
            print("ERROR: --object required with --image"); sys.exit(1)
        rgb    = np.array(PILImage.open(args.image).convert("RGB"))
        result = gemini_grasp(rgb, args.object, key)
        result["object"] = args.object
        result["image"]  = args.image
        print(json.dumps(result, indent=2))
        return

    # ── batch mode: walk run_dir/imgs/<model_id>/input.png ───────────────────
    imgs_dir = os.path.join(args.run_dir, "imgs")
    if not os.path.isdir(imgs_dir):
        print(f"ERROR: {imgs_dir} not found"); sys.exit(1)

    # read existing graspmas results to get object names
    graspmas_results = {}
    results_path = os.path.join(args.run_dir, "results.json")
    if os.path.exists(results_path):
        for r in json.load(open(results_path)):
            graspmas_results[r.get("model_id", r.get("rank", ""))] = r

    model_ids  = sorted(os.listdir(imgs_dir))
    if args.limit:
        model_ids = model_ids[:args.limit]
    all_results = []
    n_detected  = 0

    import math
    eta = len(model_ids) * (args.delay + 10)
    print(f"Gemini Direct — batch over {len(model_ids)} images")
    print(f"Model : {GEMINI_MODEL}  |  delay={args.delay}s  |  ETA ~{math.ceil(eta/60)}min\n")

    for i, model_id in enumerate(model_ids):
        img_path = os.path.join(imgs_dir, model_id, "input.png")
        if not os.path.exists(img_path):
            # try scene.png or query_image.png
            for alt in ("scene.png", "query_image.png"):
                alt_path = os.path.join(imgs_dir, model_id, alt)
                if os.path.exists(alt_path):
                    img_path = alt_path; break
            else:
                continue

        # get object name
        gm = graspmas_results.get(model_id, {})
        name = gm.get("name") or model_id.split('_',1)[-1].replace('_',' ')

        rgb    = np.array(PILImage.open(img_path).convert("RGB"))
        result = gemini_grasp(rgb, name, key)
        result.update({"model_id": model_id, "name": name, "image": img_path,
                        "graspmas_detected": gm.get("detected"),
                        "graspmas_verdict":  gm.get("verdict")})
        all_results.append(result)
        if result["detected"]:
            n_detected += 1

        det = "✓" if result["detected"] else "✗"
        g_str = (f"cx={result['cx']} cy={result['cy']} θ={result['angle']}°"
                 if result["detected"] else result.get("error","")[:40])
        print(f"[{i+1:2d}/{len(model_ids)}] {name:<25} {det}  {g_str}  [{result['elapsed_s']}s]")

        if i < len(model_ids) - 1:
            time.sleep(args.delay)

    print(f"\nDetected: {n_detected}/{len(all_results)} = {n_detected/len(all_results)*100:.1f}%")

    out_path = os.path.join(args.run_dir, "gemini_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
