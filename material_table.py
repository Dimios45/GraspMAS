"""
Material physics lookup table for gripper force estimation.

Per-material properties:
  density_kg_m3    : bulk density (kg/m³) — used for mass = density × volume
  mu_s             : realistic static friction coefficient
  max_pressure_kPa : max safe contact pressure before surface damage (kPa)
  brittleness      : "brittle" (shatters) | "soft" (crushes) | "ductile" (deforms)
  notes            : which YCB objects fall in this category
"""

MATERIAL_TABLE = {
    # ── Metals ──────────────────────────────────────────────────────────────
    "metal": dict(
        density_kg_m3=7800, mu_s=0.30,
        max_pressure_kPa=500, brittleness="ductile",
        notes="steel cans, tools, clamps, scissors, cutlery, padlock, wrench"
    ),
    "aluminum": dict(
        density_kg_m3=2700, mu_s=0.30,
        max_pressure_kPa=350, brittleness="ductile",
        notes="lightweight metal parts, some tool bodies"
    ),

    # ── Plastics ─────────────────────────────────────────────────────────────
    "hard_plastic": dict(
        density_kg_m3=1150, mu_s=0.30,
        max_pressure_kPa=250, brittleness="ductile",
        notes="ABS/PC/PET — lego duplo, dice, rubiks cube, toy airplane, "
              "golf ball, large marker, cups (most), power drill body"
    ),
    "soft_plastic": dict(
        density_kg_m3=920, mu_s=0.25,
        max_pressure_kPa=80, brittleness="soft",
        notes="HDPE/PP — mustard bottle, bleach cleanser, pitcher base, "
              "flexible containers"
    ),

    # ── Cardboard / Paper ────────────────────────────────────────────────────
    "cardboard": dict(
        density_kg_m3=150, mu_s=0.45,
        max_pressure_kPa=50, brittleness="soft",
        notes="hollow boxes: cracker box, sugar box, pudding box, gelatin box. "
              "Apparent density is low because the box volume is mostly air."
    ),

    # ── Rubber ───────────────────────────────────────────────────────────────
    "rubber": dict(
        density_kg_m3=1150, mu_s=0.75,
        max_pressure_kPa=45, brittleness="soft",
        notes="solid rubber balls — mini soccer ball, softball, baseball, "
              "tennis ball, racquetball. High friction, deforms under pressure."
    ),

    # ── Foam ─────────────────────────────────────────────────────────────────
    "foam": dict(
        density_kg_m3=80, mu_s=0.40,
        max_pressure_kPa=5, brittleness="soft",
        notes="foam brick, sponge — extremely light, crushes at minimal force"
    ),

    # ── Wood ─────────────────────────────────────────────────────────────────
    "wood": dict(
        density_kg_m3=700, mu_s=0.40,
        max_pressure_kPa=180, brittleness="ductile",
        notes="wood block, colored wood blocks, nine hole peg test, hammer handle"
    ),

    # ── Fragile ──────────────────────────────────────────────────────────────
    "ceramic": dict(
        density_kg_m3=2000, mu_s=0.40,
        max_pressure_kPa=220, brittleness="brittle",
        notes="mug, ceramic bowl — strong in compression but shatters on impact. "
              "Handle is the weak point, not the body."
    ),
    "glass": dict(
        density_kg_m3=2500, mu_s=0.20,
        max_pressure_kPa=150, brittleness="brittle",
        notes="marbles — very slippery, chips at edges under point load"
    ),

    # ── Fruit / Food ─────────────────────────────────────────────────────────
    "fruit": dict(
        density_kg_m3=880, mu_s=0.50,
        max_pressure_kPa=20, brittleness="soft",
        notes="banana, apple, orange, lemon, peach, pear, plum, strawberry. "
              "YCB replicas are synthetic rubber but should be treated as real "
              "fruit for force safety (bruise threshold is very low)."
    ),
    "food": dict(
        density_kg_m3=1000, mu_s=0.45,
        max_pressure_kPa=20, brittleness="soft",
        notes="generic food — density close to water, soft tissue"
    ),

    # ── Composite ────────────────────────────────────────────────────────────
    "composite": dict(
        density_kg_m3=2000, mu_s=0.35,
        max_pressure_kPa=130, brittleness="ductile",
        notes="power drill, hammer, screwdrivers, spatula — mixed metal+plastic. "
              "High apparent density because metal parts dominate mass."
    ),

    # ── Fallback ─────────────────────────────────────────────────────────────
    "other": dict(
        density_kg_m3=1000, mu_s=0.30,
        max_pressure_kPa=100, brittleness="ductile",
        notes="unknown material — conservative defaults"
    ),
}

# Ground-truth material labels for every YCB object in PickSingleYCB-v1.
# Based on actual object composition, NOT ManiSkill's uniform mu=0.3 fiction.
YCB_GT_MATERIAL = {
    # Cans (metal)
    "002_master_chef_can":       "metal",
    "005_tomato_soup_can":       "metal",
    "007_tuna_fish_can":         "metal",
    "010_potted_meat_can":       "metal",

    # Boxes (cardboard)
    "003_cracker_box":           "cardboard",
    "004_sugar_box":             "cardboard",
    "008_pudding_box":           "cardboard",
    "009_gelatin_box":           "cardboard",

    # Plastic bottles / containers
    "006_mustard_bottle":        "soft_plastic",
    "019_pitcher_base":          "soft_plastic",
    "021_bleach_cleanser":       "soft_plastic",

    # Fruit (soft, crushable — YCB replicas are synthetic rubber)
    "011_banana":                "fruit",
    "012_strawberry":            "fruit",
    "013_apple":                 "fruit",
    "014_lemon":                 "fruit",
    "015_peach":                 "fruit",
    "016_pear":                  "fruit",
    "017_orange":                "fruit",
    "018_plum":                  "fruit",

    # Kitchenware
    "024_bowl":                  "hard_plastic",   # YCB bowl is white plastic
    "025_mug":                   "ceramic",
    "026_sponge":                "foam",

    # Cutlery (metal)
    "030_fork":                  "metal",
    "031_spoon":                 "metal",
    "032_knife":                 "metal",

    # Tools — composite (metal head + plastic/wood handle)
    "033_spatula":               "composite",
    "035_power_drill":           "composite",
    "037_scissors":              "metal",
    "042_adjustable_wrench":     "metal",
    "043_phillips_screwdriver":  "composite",
    "044_flat_screwdriver":      "composite",
    "048_hammer":                "composite",

    # Clamps (metal)
    "050_medium_clamp":          "metal",
    "051_large_clamp":           "metal",
    "052_extra_large_clamp":     "metal",

    # Other tools
    "038_padlock":               "metal",
    "040_large_marker":          "hard_plastic",

    # Balls (rubber)
    "053_mini_soccer_ball":      "rubber",
    "054_softball":              "rubber",
    "055_baseball":              "rubber",
    "056_tennis_ball":           "rubber",
    "057_racquetball":           "rubber",
    "058_golf_ball":             "hard_plastic",   # surlyn hard cover

    # Misc
    "036_wood_block":            "wood",
    "061_foam_brick":            "foam",
    "062_dice":                  "hard_plastic",
    "063-a_marbles":             "glass",
    "063-b_marbles":             "glass",

    # Cups — mostly hard plastic; 065-h is a silicone/rubber cup
    "065-a_cups":                "hard_plastic",
    "065-b_cups":                "hard_plastic",
    "065-c_cups":                "hard_plastic",
    "065-d_cups":                "hard_plastic",
    "065-e_cups":                "hard_plastic",
    "065-f_cups":                "hard_plastic",
    "065-g_cups":                "hard_plastic",
    "065-h_cups":                "rubber",          # silicone cup
    "065-i_cups":                "hard_plastic",
    "065-j_cups":                "hard_plastic",

    # Wood items
    "070-a_colored_wood_blocks": "wood",
    "070-b_colored_wood_blocks": "wood",
    "071_nine_hole_peg_test":    "wood",

    # Toys — hard ABS plastic
    "072-a_toy_airplane":        "hard_plastic",
    "072-b_toy_airplane":        "hard_plastic",
    "072-c_toy_airplane":        "hard_plastic",
    "072-d_toy_airplane":        "hard_plastic",
    "072-e_toy_airplane":        "hard_plastic",
    "073-a_lego_duplo":          "hard_plastic",
    "073-b_lego_duplo":          "hard_plastic",
    "073-c_lego_duplo":          "hard_plastic",
    "073-d_lego_duplo":          "hard_plastic",
    "073-e_lego_duplo":          "hard_plastic",
    "073-f_lego_duplo":          "hard_plastic",
    "073-g_lego_duplo":          "hard_plastic",
    "077_rubiks_cube":           "hard_plastic",
}


def lookup(material_str: str) -> dict:
    """Return material properties dict, falling back to 'other' for unknowns."""
    key = material_str.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MATERIAL_TABLE:
        return MATERIAL_TABLE[key]
    for k in MATERIAL_TABLE:
        if key.startswith(k):
            return MATERIAL_TABLE[k]
    return MATERIAL_TABLE["other"]


def get_gt_material(model_id: str) -> str:
    return YCB_GT_MATERIAL.get(model_id, "other")
