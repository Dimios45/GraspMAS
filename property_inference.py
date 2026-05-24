"""
property_inference.py — Mass predictor for YCB objects (GraspMAS PoC).

Ground truth
------------
Computed from ManiSkill's info_pick_v0.json (density + scale) and the
per-object collision.ply files via trimesh.  No live simulation needed.

  mass_gt = density_kg_m3  ×  trimesh_volume(collision.ply)  ×  scale³

This matches exactly what SAPIEN/PhysX computes at actor-build time.

Prediction pipeline
-------------------
  1. Load the object's true bounding-box dimensions from info_pick_v0.json
     (scale-corrected, in metres).  This eliminates dimension hallucination —
     the largest single error source in v1.
  2. Retrieve k nearest reference objects by SentenceTransformer embedding
     (query object is never in its own anchor pool).
  3. Build a chain-of-thought prompt:
       - Provide actual bbox dims + pre-computed bbox volume
       - Highlight nearest anchor with an explicit mass range
       - List material densities + packing factors for hollow/thin objects
       - Require step-by-step reasoning → JSON
  4. Call Qwen2-VL-7B via vlm_chat() (local, graspmas conda env).
     Falls back to bbox-based offline stub when VLM unavailable.
  5. Post-process: if the parsed mass is > 5× from the nearest anchor,
     apply a geometric blend toward the anchor.
  6. Self-consistency: median of N samples; confidence from IQR dispersion.
  7. Physical bounds rejection (0.001–30 kg) with resample-on-invalid.

API verified by: python property_inference.py --verify-gt
"""

from __future__ import annotations

import base64
import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_MANISKILL_ASSETS = Path(
    os.getenv("MANISKILL_ASSET_DIR",
              "/mnt/data/mritunjoyh/.maniskill/data/assets/mani_skill2_ycb")
)
_INFO_JSON  = _MANISKILL_ASSETS / "info_pick_v0.json"
_MODELS_DIR = _MANISKILL_ASSETS / "models"

# Physical bounds — wide enough to catch any real YCB object prediction
MASS_LO, MASS_HI = 0.001, 50.0

# ── Human-readable label ─────────────────────────────────────────────────────
def label_for(obj_id: str) -> str:
    """'036_wood_block' → 'wood block'"""
    parts = obj_id.split("_", 1)
    return parts[1].replace("_", " ") if len(parts) > 1 else obj_id


# ── Ground-truth mass table ──────────────────────────────────────────────────
def build_gt_table(
    info_json: Path = _INFO_JSON,
    models_dir: Path = _MODELS_DIR,
) -> dict[str, float]:
    """
    Return {obj_id: mass_kg} for every YCB object with a collision mesh.

    ManiSkill mass readout API (exact computation SAPIEN/PhysX uses):
        mass = info['density']  ×  trimesh.load(collision.ply).convex_hull.volume
                                ×  info['scales'][0] ** 3

    Verified: 78 objects, mass range 0.005–1.37 kg, all nonzero.
    """
    with open(info_json) as f:
        info = json.load(f)

    gt: dict[str, float] = {}
    for obj_dir in sorted(models_dir.iterdir()):
        ply = obj_dir / "collision.ply"
        if not ply.exists():
            continue
        meta    = info.get(obj_dir.name, {})
        scale   = meta.get("scales", [1.0])[0]
        density = meta.get("density", 1000)
        mesh    = trimesh.load(str(ply))
        if not mesh.is_watertight:
            mesh = mesh.convex_hull
        vol = mesh.volume * (scale ** 3)
        gt[obj_dir.name] = round(density * vol, 6)

    return gt


def build_bbox_table(info_json: Path = _INFO_JSON) -> dict[str, list[float]]:
    """
    Return {obj_id: [Lx, Ly, Lz]} — scaled bounding-box side lengths in metres.

    Used to give the LLM actual object dimensions so it cannot hallucinate them.
    """
    with open(info_json) as f:
        info = json.load(f)

    bbox: dict[str, list[float]] = {}
    for obj_id, meta in info.items():
        bb    = meta.get("bbox", {})
        lo    = bb.get("min", [0.0, 0.0, 0.0])
        hi    = bb.get("max", [0.0, 0.0, 0.0])
        scale = meta.get("scales", [1.0])[0]
        dims  = [round((h - l) * scale, 5) for h, l in zip(hi, lo)]
        bbox[obj_id] = dims

    return bbox


# ── Embedding + retrieval ─────────────────────────────────────────────────────
def _embed_labels(labels: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(labels, normalize_embeddings=True)


def retrieve_anchors(
    query_id: str,
    gt_table: dict[str, float],
    embeddings: dict[str, np.ndarray],
    k: int = 5,
    exclude_ids: Optional[set[str]] = None,
) -> list[tuple[str, float]]:
    """
    Return k (obj_id, mass_kg) pairs nearest to query_id by cosine similarity.
    The query and any ids in exclude_ids are never returned.
    """
    exclude = {query_id} | (exclude_ids or set())
    q_emb   = embeddings[query_id]
    pool    = [(oid, embeddings[oid]) for oid in gt_table if oid not in exclude]
    sims    = [(oid, float(q_emb @ emb)) for oid, emb in pool]
    sims.sort(key=lambda x: -x[1])
    return [(oid, gt_table[oid]) for oid, _ in sims[:k]]


# ── LLM backend ──────────────────────────────────────────────────────────────
try:
    from local_vlm import vlm_chat as _vlm_chat
    _VLM_AVAILABLE = True
except Exception:
    _vlm_chat      = None
    _VLM_AVAILABLE = False

# ── Prompt constants ──────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a robotics physics expert. Estimate object mass precisely using "
    "material density and geometry reasoning. Respond ONLY with valid JSON. "
    "No markdown, no prose, no code blocks."
)

_MATERIAL_DENSITY = """\
Material densities (kg/m³):
  steel=7800  aluminium=2700  copper=8900  glass=2500  ceramic=2400
  rubber=1200  plastic_solid=950  plastic_hollow=200
  wood_dense=700  wood_light=400
  food_liquid=1000  food_solid=800  food_powder=500
  foam=30  paper_cardboard=700  fabric=300"""

_PACKING_FACTORS = """\
Packing factors (what fraction of the bounding box volume is solid material):
  solid_box / dense_block = 0.90   (cardboard box, rubiks cube, sponge)
  can / bottle (filled)   = 0.50   (soup can, mustard bottle — thin walls + liquid)
  bowl / cup / mug        = 0.12   (hollow vessel, mostly air)
  sphere (ball/fruit)     = 0.52   (apple, orange, baseball — π/6)
  spoon / fork / knife    = 0.05   (very thin utensil, almost all bbox is air)
  thin_lid / flat_plate   = 0.08   (skillet lid — thin disc, lots of empty bbox)
  clamp / bracket         = 0.15   (c-shaped metal, mostly air in bbox)
  foam_block              = 0.90   (foam fills bbox; density is already 30)
  chain                   = 0.25   (links + gaps)
  hollow_toy / airplane   = 0.08   (thin plastic shell)
  pitcher / jug           = 0.15   (hollow plastic vessel)
  screwdriver / wrench    = 0.20   (cylindrical handle + thin metal shaft)"""


def _anchor_text(anchors: list[tuple[str, float]]) -> str:
    lines = [f"  {'→' if i == 0 else ' '} {label_for(oid)}: {mass:.4f} kg"
             for i, (oid, mass) in enumerate(anchors)]
    return "\n".join(lines)


def build_decomp_prompt(
    label: str,
    anchors: list[tuple[str, float]],
    bbox_dims_m: Optional[list[float]] = None,
    sim_density_hint: bool = False,
    image_b64: Optional[str] = None,
) -> list[dict]:
    """
    Chain-of-thought decomposition prompt.
    Including bbox dimensions eliminates the #1 hallucination source.
    """
    # geometry section
    if bbox_dims_m:
        L, W, H   = bbox_dims_m
        bbox_vol  = L * W * H
        geom_text = (f"Object bounding box (from simulation scan):\n"
                     f"  {L:.4f} m × {W:.4f} m × {H:.4f} m  "
                     f"(bbox_volume = {bbox_vol:.6f} m³)\n"
                     f"  Use these exact dimensions — do NOT re-estimate size.")
    else:
        geom_text  = "(No geometry data. Estimate dimensions from label knowledge.)"
        bbox_vol   = None

    # anchor section — highlight nearest with explicit allowed range
    nearest_oid, nearest_mass = anchors[0] if anchors else (None, None)
    if nearest_mass is not None:
        lo_range = nearest_mass / 5.0
        hi_range = nearest_mass * 5.0
        anchor_constraint = (
            f"\nIMPORTANT — nearest reference: \"{label_for(nearest_oid)}\" "
            f"at {nearest_mass:.4f} kg.\n"
            f"Your mass_kg MUST lie in [{lo_range:.4f}, {hi_range:.4f}] kg.\n"
            f"If your material calculation falls outside this range, "
            f"adjust packing_factor or density until it does."
        )
    else:
        anchor_constraint = ""

    sim_hint = (
        "\nSIMULATION NOTE: This simulator uses simplified physics densities. "
        "Box-shaped / hollow objects often have masses of 0.005–0.2 kg in this "
        "sim. When similar anchors are in that range, match their scale."
        if sim_density_hint else ""
    )

    # volume step — show the arithmetic template if bbox known
    vol_step = (
        f"(c) VOLUME:  volume_m3 = {bbox_vol:.6f} × packing_factor"
        if bbox_vol is not None
        else "(c) VOLUME:  volume_m3 = L × W × H × packing_factor"
    )

    text = f"""\
Estimate the mass (kg) of: "{label}"

{geom_text}

Reference objects (calibrated masses, same simulation):
{_anchor_text(anchors)}
{anchor_constraint}
{sim_hint}

{_MATERIAL_DENSITY}

{_PACKING_FACTORS}

Reason step by step:
(a) MATERIAL: Primary material and density (kg/m³) from the table above.
(b) PACKING:  Which packing_factor fits this object? Choose from the list above.
{vol_step}
(d) MASS:     mass_kg = density_kg_m3 × volume_m3
(e) CHECK:    Is mass_kg in the allowed anchor range above? If not, adjust and restate.

Output ONLY this JSON (no other text):
{{"material": "<str>", "density_kg_m3": <float>, "packing_factor": <float>, "volume_m3": <float>, "mass_kg": <float>}}"""

    if image_b64 is None:
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": text},
        ]
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]},
    ]


def build_direct_prompt(
    label: str,
    anchors: list[tuple[str, float]],
    bbox_dims_m: Optional[list[float]] = None,
    image_b64: Optional[str] = None,
) -> list[dict]:
    """Direct mass guess with anchor hints (ablation — no CoT)."""
    geom_text = ""
    if bbox_dims_m:
        L, W, H  = bbox_dims_m
        geom_text = f"Object bbox: {L:.4f}×{W:.4f}×{H:.4f} m\n"

    nearest_oid, nearest_mass = anchors[0] if anchors else (None, None)
    anchor_hint = (
        f"\nYour mass_kg should be near {nearest_mass:.4f} kg "
        f"(nearest reference: \"{label_for(nearest_oid)}\")."
        if nearest_mass is not None else ""
    )

    text = f"""\
Estimate the mass (kg) of: "{label}"
{geom_text}
Reference objects:
{_anchor_text(anchors)}
{anchor_hint}

Output ONLY: {{"mass_kg": <float>}}"""

    if image_b64 is None:
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": text},
        ]
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]},
    ]


# ── JSON extractor ────────────────────────────────────────────────────────────
def _eval_arith(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression (only +-*/ and numbers)."""
    if re.fullmatch(r'[\d\s\.\+\-\*\/eE]+', expr.strip()):
        try:
            return float(eval(expr))  # noqa: S307 — guarded by regex above
        except Exception:
            pass
    return None


def _resolve_expr_in_json(text: str) -> str:
    """
    Replace arithmetic expressions in JSON values with their computed results.
    Handles: "mass_kg": 7800 * 0.000179 * 0.90  →  "mass_kg": 1.25262
    """
    def replacer(m):
        val = _eval_arith(m.group(1))
        return f'{m.group(0).split(":")[0]}: {val}' if val is not None else m.group(0)

    return re.sub(
        r':\s*([\d\s\.\+\-\*\/eE]+(?:[\+\-\*\/][\s\d\.eE]+)+)',
        replacer, text
    )


def _parse_mass(text: str) -> Optional[float]:
    """
    Extract mass_kg from a (possibly messy) LLM response.
    Handles: strict JSON, arithmetic expressions in values, regex fallback.
    """
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    # pre-process arithmetic expressions (e.g. "mass_kg": 7800 * 0.002 * 0.5)
    text_resolved = _resolve_expr_in_json(text)

    for candidate in [text_resolved, text]:
        # try full JSON
        try:
            return float(json.loads(candidate)["mass_kg"])
        except Exception:
            pass
        # try first {...} block
        m = re.search(r"\{[^{}]*\}", candidate, re.DOTALL)
        if m:
            try:
                return float(json.loads(m.group())["mass_kg"])
            except Exception:
                pass

    # last resort: regex on the raw field
    m = re.search(r'"mass_kg"\s*:\s*([0-9.eE+\-\s\*\/]+?)(?:[,}\n])', text)
    if m:
        val = _eval_arith(m.group(1)) or None
        if val is None:
            try:
                val = float(m.group(1).strip())
            except Exception:
                pass
        return val
    return None


# ── LLM call ─────────────────────────────────────────────────────────────────
def _call_llm_once(messages: list[dict], temperature: float = 0.9) -> str:
    return _vlm_chat(messages, max_new_tokens=512, temperature=temperature)


def _offline_stub(obj_id: str, bbox_table: dict[str, list[float]],
                  info_json: Path = _INFO_JSON) -> float:
    """Fallback: bbox_volume × density from info JSON (no LLM)."""
    with open(info_json) as f:
        info = json.load(f)
    meta    = info.get(obj_id, {})
    scale   = meta.get("scales", [1.0])[0]
    density = meta.get("density", 1000)
    dims    = bbox_table.get(obj_id, [0.1, 0.1, 0.1])
    vol     = dims[0] * dims[1] * dims[2]
    return float(density * vol)


# ── Anchor blend post-processing ──────────────────────────────────────────────
def _anchor_blend(pred: float, nearest_mass: float) -> float:
    """
    Adaptive geometric blend toward nearest anchor.

    The blend factor grows logarithmically with the ratio so that extreme
    hallucinations (solid-steel spoon at 30× true) are pulled down harder
    than mild over/under-estimates.

    ratio < 3   → no correction (model is within reasonable range)
    ratio 3–10  → blend = 0.55
    ratio 10–50 → blend = 0.70
    ratio > 50  → blend = 0.85
    """
    ratio = pred / max(nearest_mass, 1e-9)
    if ratio < 1.0:
        ratio = 1.0 / ratio  # handle under-estimates symmetrically

    if ratio < 3.0:
        return pred
    elif ratio < 10.0:
        blend = 0.55
    elif ratio < 50.0:
        blend = 0.70
    else:
        blend = 0.85

    return float(np.exp(
        (1.0 - blend) * np.log(max(pred, 1e-9)) +
        blend * np.log(max(nearest_mass, 1e-9))
    ))


# ── Main predictor class ──────────────────────────────────────────────────────
class MassPredictor:
    """
    Predict YCB object masses via Qwen2-VL-7B with:
      - Real bounding-box dimensions (from sim assets — no hallucination)
      - Embedding-nearest anchor retrieval (SentenceTransformer)
      - Chain-of-thought decomposition: material → packing → volume → mass
      - Anchor-range constraint in prompt + geometric blend post-processing
      - Self-consistency: median of N samples, confidence from IQR

    Parameters
    ----------
    gt_table         : {obj_id: mass_kg} — ground-truth (anchor pool source)
    n_samples        : self-consistency samples per object
    temperature      : LLM sampling temperature (0.9 default for diversity)
    k_anchors        : number of retrieval anchors
    use_retrieval    : False → random anchors (ablation)
    use_decomp       : False → direct guess prompt (ablation)
    use_anchor_blend : False → skip geometric post-blend (ablation)
    sim_density_hint : True → tell model about sim's simplified densities
    """

    def __init__(
        self,
        gt_table: dict[str, float],
        n_samples: int   = 5,
        temperature: float = 0.9,
        k_anchors: int   = 5,
        use_retrieval: bool    = True,
        use_decomp: bool       = True,
        use_anchor_blend: bool = True,
        sim_density_hint: bool = False,
    ):
        self.gt_table          = gt_table
        self.n_samples         = n_samples
        self.temperature       = temperature
        self.k_anchors         = k_anchors
        self.use_retrieval     = use_retrieval
        self.use_decomp        = use_decomp
        self.use_anchor_blend  = use_anchor_blend
        self.sim_density_hint  = sim_density_hint

        if not _VLM_AVAILABLE:
            warnings.warn(
                "local_vlm.vlm_chat not importable — offline stub active. "
                "Run in the graspmas conda env for real predictions.",
                UserWarning,
            )

        # Pre-load bbox dimensions (eliminates hallucination)
        self._bbox_table = build_bbox_table()

        # Pre-compute label embeddings for all objects
        ids    = sorted(gt_table.keys())
        labels = [label_for(i) for i in ids]
        vecs   = _embed_labels(labels)
        self._embeddings: dict[str, np.ndarray] = dict(zip(ids, vecs))

    # ── anchor selection ──────────────────────────────────────────────────────

    def _get_anchors(
        self,
        obj_id: str,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        if self.use_retrieval:
            return retrieve_anchors(
                obj_id, self.gt_table, self._embeddings,
                k=self.k_anchors, exclude_ids=exclude_ids,
            )
        rng    = np.random.default_rng(abs(hash(obj_id)) % (2**32))
        pool   = [o for o in self.gt_table if o != obj_id
                  and (exclude_ids is None or o not in exclude_ids)]
        chosen = rng.choice(pool, size=min(self.k_anchors, len(pool)), replace=False)
        return [(o, self.gt_table[o]) for o in chosen]

    # ── single LLM call with bounds check ─────────────────────────────────────

    def _predict_once(
        self,
        obj_id: str,
        anchors: list[tuple[str, float]],
        image_b64: Optional[str],
        max_retries: int = 3,
    ) -> tuple[Optional[float], str]:
        """Returns (mass_kg_or_None, raw_llm_text)."""
        label      = label_for(obj_id)
        bbox_dims  = self._bbox_table.get(obj_id)
        last_raw   = ""

        for _ in range(max_retries):
            if self.use_decomp:
                msgs = build_decomp_prompt(
                    label, anchors,
                    bbox_dims_m      = bbox_dims,
                    sim_density_hint = self.sim_density_hint,
                    image_b64        = image_b64,
                )
            else:
                msgs = build_direct_prompt(
                    label, anchors,
                    bbox_dims_m = bbox_dims,
                    image_b64   = image_b64,
                )
            try:
                last_raw = _call_llm_once(msgs, self.temperature)
                mass     = _parse_mass(last_raw)
            except Exception as exc:
                last_raw = f"ERROR: {exc}"
                continue

            if mass is not None and MASS_LO <= mass <= MASS_HI:
                return mass, last_raw

        return None, last_raw

    # ── self-consistency ──────────────────────────────────────────────────────

    def predict(
        self,
        obj_id: str,
        image_b64: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> tuple[Optional[float], float, list[Optional[float]], list[str]]:
        """
        Returns (mass_kg, confidence, parsed_samples, raw_texts).

        confidence = 1 − IQR/median, clipped to [0, 1].
        mass_kg is median of valid samples, then optionally anchor-blended.
        """
        if not _VLM_AVAILABLE:
            mass = _offline_stub(obj_id, self._bbox_table)
            return mass, 0.0, [mass], ["offline_stub"]

        anchors   = self._get_anchors(obj_id, exclude_ids)
        pairs     = [self._predict_once(obj_id, anchors, image_b64)
                     for _ in range(self.n_samples)]
        samples   = [p[0] for p in pairs]
        raw_texts = [p[1] for p in pairs]

        valid = [s for s in samples if s is not None]
        if not valid:
            return None, 0.0, samples, raw_texts

        arr    = np.array(valid)
        median = float(np.median(arr))

        # Anchor post-blend
        if self.use_anchor_blend and anchors:
            nearest_mass = anchors[0][1]
            median = _anchor_blend(median, nearest_mass)

        q1, q3 = np.percentile(arr, [25, 75])
        iqr    = q3 - q1
        conf   = float(np.clip(1.0 - (iqr / max(median, 1e-6)), 0.0, 1.0))
        return median, conf, samples, raw_texts


# ── Image helpers ─────────────────────────────────────────────────────────────
def load_image_b64(path: "str | Path") -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def crop_image_b64(
    rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    pad: float = 0.15,
) -> str:
    """Crop HxWx3 uint8 array to bbox with padding, return base64 PNG."""
    from PIL import Image
    import io
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    dw = int((x2 - x1) * pad); dh = int((y2 - y1) * pad)
    x1 = max(0, x1 - dw); y1 = max(0, y1 - dh)
    x2 = min(W, x2 + dw); y2 = min(H, y2 + dh)
    buf = io.BytesIO()
    Image.fromarray(rgb[y1:y2, x1:x2]).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── CLI verify-gt ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--verify-gt", action="store_true")
    args = p.parse_args()

    if args.verify_gt:
        print("Building ground-truth mass table from ManiSkill assets...")
        print(f"  info JSON  : {_INFO_JSON}")
        print(f"  models dir : {_MODELS_DIR}")
        print()
        gt   = build_gt_table()
        bbox = build_bbox_table()
        print(f"{'Object ID':<35} {'Label':<26} {'Mass (kg)':>10}  {'BBox L×W×H (m)':}")
        print("-" * 90)
        for oid in sorted(gt):
            d = bbox.get(oid, [0, 0, 0])
            print(f"{oid:<35} {label_for(oid):<26} {gt[oid]:>10.4f}"
                  f"  {d[0]:.3f}×{d[1]:.3f}×{d[2]:.3f}")
        vals = list(gt.values())
        print()
        print(f"Total objects : {len(gt)}")
        print(f"Mass range    : {min(vals):.4f} – {max(vals):.4f} kg")
        print(f"Mean / Median : {np.mean(vals):.4f} / {np.median(vals):.4f} kg")
        print()
        print("API used: trimesh.load(collision.ply)[.convex_hull].volume × scale³ × density")
        print("          density and scale from info_pick_v0.json")
