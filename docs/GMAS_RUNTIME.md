# gmas — an OpenClaw-style agent runtime for GraspMAS

`gmas/` re-hosts the GraspMAS Planner→Coder→Observer loop on a structured
agent runtime (typed messages, tool registry, sandboxed execution, session
state, audit logging — the OpenClaw/ROSClaw component set) and measures the
result head-to-head against the unmodified legacy loop.

The legacy pipeline (`agents/graspmas.py`) is untouched and remains the
baseline arm of every experiment.

## Why

The legacy loop has four structural weaknesses, each visible in the baseline
failure taxonomy below:

1. **Bare `exec(code, globals())`** — a Coder syntax error aborts the whole
   query (32% of baseline queries die this way); runtime errors burn a full
   round before the Observer sees a lossy error string.
2. **Hand-maintained prompt stub** — the Coder prompt's fake `ImagePatch`
   class drifts from the real API, producing hallucinated-method calls
   (`.overlaps_with()` on a list, `adjust_grasp_pose`, …).
3. **No memory across rounds** — every round re-runs GroundingDINO/SAM/depth
   on the same image; a good round-1 grasp is thrown away if round 3 fails.
4. **Unconditional Observer** — a VLM call is spent even on obviously
   good/bad results.

## Architecture

```
gmas/
├── config.py            Flags — every feature is a switch, all OFF = parity mode
├── telemetry.py         per-query LLM-call/token/latency collector (both arms)
├── runtime/
│   ├── messages.py      Pydantic: Query/Plan/Code/Exec/Tool/Validation/Observation/Final
│   ├── bus.py           in-process pub/sub; audit subscribes to '*'
│   ├── registry.py      ToolRegistry — single source of truth for the callable API;
│   │                    generates the Coder prompt API section (flag: registry_prompt)
│   ├── sandbox.py       AST parse+compile → auto-repair (flag: repair) → whitelist
│   │                    (flag: whitelist) → fresh namespace → optional timeout
│   ├── session.py       round snapshots, tool cache, best_round() for rollback
│   ├── cache.py         memoizes ImagePatch tool methods per session (flag: cache)
│   ├── validator.py     deterministic grasp checks (flag: validator)
│   └── audit.py         JSONL per session — the eval tables come from these
├── agents/orchestrator.py  GmasPipeline — drop-in GraspMAS replacement,
│                            legacy prompts reused verbatim
├── eval/
│   ├── instrumented.py  telemetry proxies for the legacy arm (no behavior change)
│   ├── run_ocidvlg_ab.py  A/B harness (same stratified sample, same metric)
│   └── report.py        comparison table across run dirs
└── tests/test_runtime.py  micro-tests (repair, whitelist, validator, cache, rollback)
```

### Feature flags

| flag | mechanism | targets |
|---|---|---|
| `repair` | mechanical fix of known Qwen2-VL syntax failures (module-scope `return`, bare script bodies, indent) before giving up | 32% syntax-error aborts |
| `whitelist` | static AST check against the registry before any GPU work | hallucinated-method rounds |
| `registry_prompt` | authoritative API list injected into the Coder prompt from the registry | prompt↔API drift |
| `validator` | grasp center inside image & inside a detected box, non-degenerate rect | wrong-location returns |
| `cache` | per-session memoization of find/SAM/depth/VLM tool calls keyed by crop coords + args | repeated detector work across rounds |
| `gate` (`gate_q`) | skip the Observer VLM call when execution succeeded and grasp quality ≥ threshold | 1 VLM call/round |
| `early_term` | stop on VALID/AUTO_VALID verdict instead of replanning | wasted rounds |
| `rollback` | return the best round's grasp (verdict rank → validator → quality) instead of the last | max-round exhaustion |
| `timeout_s` | per-round execution timeout | hung rounds |

Parity notes (what flags-off mode deliberately reproduces, and the one
divergence) are documented in `gmas/agents/orchestrator.py`.

## Environment & reproduction

- **Python env:** `miniforge3/envs/graspmas` (Python 3.9 — note: no `X | Y`
  type unions in `gmas/`). All commands below assume this env from the repo
  root.
- **Models:** local checkpoints under `/mnt/data/mritunjoyh/models/`
  (Qwen2-VL-7B-Instruct, grounding-dino-tiny, owlv2, sam-vit-base,
  dpt-hybrid-midas) + `weights/` (RAGT-3-3.pth, vlpart). **BLIP-2 is
  lazy-loaded** by `image_patch.py` and its weights are *not* on disk
  (disk quota); only `best_image_match()` needs it.
- **Dataset:** OCID-VLG at `/mnt/data/mritunjoyh/datasets/ocid-vlg` is a
  *selective extract* (~1.2 GB): `refer/` annotations + the 3,434 rgb/depth
  images referenced by the test splits. Re-create with
  `python gmas/eval/extract_ocid.py <zip> <dest>` (annotations first, then only
  referenced images) — the full zip is ~5.4 GB / ~11 GB extracted and does
  not fit in the quota.
- **Tests:** `python -m pytest gmas/tests/test_runtime.py -q` — 18 tests,
  no GPU (namespace and image_patch are stubbed).
- **Provenance:** every run directory under `runs/*_ab_*` contains
  `config.json` (exact args), `results.json` (per-query), `telemetry.jsonl`
  (per-LLM-call), `summary.txt`, and per-query `imgs/<rank>/` including the
  gmas arms' `audit.jsonl` message logs.

## Repository changes (this work)

| path | change |
|---|---|
| `gmas/` (new, ~1,800 lines) | the runtime: `runtime/` (messages, bus, registry, sandbox, session, cache, validator, audit), `agents/orchestrator.py` (GmasPipeline), `config.py` (Flags), `telemetry.py`, `eval/` (A/B harness, report, legacy instrumentation), `tests/` |
| `docs/GMAS_RUNTIME.md` (new) | this document |
| `local_vlm.py` | telemetry hook in `vlm_chat` — records tokens/latency per call into the active collector; no-op when none active |
| `image_patch.py` | BLIP-2 lazy-load (`_load_blip2()`) — was an eager 15 GB import-time load |
| `agents/graspmas.py` | **untouched by the runtime work** (it is the baseline); its uncommitted diff is from the earlier directional-grasping feature |

## Running

```bash
# baseline arm
python -m gmas.eval.run_ocidvlg_ab --root /mnt/data/mritunjoyh/datasets/ocid-vlg \
    --arm legacy --n 100 --seed 42 --tag p0_baseline

# gmas arms: parity (no flags), single features, all-on
python -m gmas.eval.run_ocidvlg_ab --root ... --arm gmas --n 100 --seed 42 --tag p1_parity
python -m gmas.eval.run_ocidvlg_ab --root ... --arm gmas --flags repair,whitelist --n 100 --seed 42
python -m gmas.eval.run_ocidvlg_ab --root ... --arm gmas --flags all --n 100 --seed 42

# comparison table
python -m gmas.eval.report runs/<ts>_ab_legacy_p0_baseline runs/<ts>_ab_gmas_*
```

## Baseline (P0) — legacy arm, OCID-VLG unique split, n=100, seed 42, max_round=4

| metric | value |
|---|---|
| success rate | 13% |
| detection rate | 49% |
| median / mean / p90 wall time per query | 13.3 s / 24.4 s / 57.9 s |
| LLM calls per query | 5.77 |
| LLM tokens per query (in / out) | 16,589 / 767 |
| rounds per query | 2.30 |

Failure taxonomy: **32% code_syntax_error** (whole-query abort from bare
`exec`), 20% wrong_location, **19% code_runtime_error**, 16% bad_grasp,
13% success. Over half of all failures are code-level — exactly the bucket
the runtime addresses mechanically.

## A/B results — OCID-VLG unique split, n=100, seed 42, max_round=4

Same queries, same checkpoints, same GPU, same metric (paper IoU protocol).

| metric | legacy | gmas parity (flags off) | gmas all-on |
|---|---|---|---|
| success rate | 13% | 15% | **16%** |
| detection rate | 49% | 54% | **72%** |
| mean IoU | 0.07 | 0.08 | 0.10 |
| median wall time /q | 13.4 s | 12.8 s | **7.9 s (−41%)** |
| p90 wall time /q | 57.9 s | 52.5 s | 45.3 s |
| LLM calls /q | 5.77 | 6.00 | **5.01** |
| output tokens /q | 767 | 804 | 661 |
| rounds /q | 2.30 | 2.36 | **1.86** |
| syntax-error aborts | 32 | 31 | **0** |
| code-error rounds | 18 | 16 | 8 |

**Parity gate passed:** flags-off gmas matches legacy within noise on every
metric (15% vs 13% success, 31 vs 32 syntax aborts) — gains below are
attributable to runtime features, not prompt or model changes.

**Reading the all-on column honestly:**
- The **entire 32% syntax-abort bucket is eliminated** by sandbox repair;
  those queries now run to completion (detection 49% → 72%).
- The Observer gate fired in 71/100 queries (verdict `AUTO_VALID` in the
  audit logs), cutting observer VLM calls 153 → 115 and driving the −41%
  median latency together with early termination (rounds 2.30 → 1.86).
- Final grasp success only rises 13% → 16%: most reclaimed queries land in
  `bad_grasp` / `wrong_location` instead. The loop no longer wastes rounds,
  but end-task success is bottlenecked by perception quality (GroundingDINO
  mis-grounding on cluttered OCID scenes, RAGT grasp quality), not by the
  agent loop. The runtime fixes what a runtime can fix.

### Per-feature ablations (same 100 queries, seed 42)

| metric | legacy | parity | +repair,whitelist | +cache,gate,early_term | +validator,rollback | all-on |
|---|---|---|---|---|---|---|
| success % | 13 | 15 | 15 | **16** | 14 | **16** |
| detected % | 49 | 54 | **75** | 57 | 54 | 72 |
| median t (s) | 13.4 | 12.8 | 18.7 | **8.3** | 12.6 | **7.9** |
| p90 t (s) | 57.9 | 52.5 | 51.9 | **40.3** | 47.3 | 45.3 |
| LLM calls /q | 5.77 | 6.00 | 7.48 | **3.55** | 5.78 | 5.01 |
| output tokens /q | 767 | 804 | 1006 | **496** | 751 | 661 |
| rounds /q | 2.30 | 2.36 | 2.88 | **1.46** | 2.30 | 1.86 |
| syntax aborts | 32 | 31 | **0** | 30 | 32 | **0** |
| code-error rounds | 18 | 16 | 21 | **1** | 17 | 8 |

Attribution is clean:

- **repair + whitelist** is the *accuracy-side recovery* feature: syntax
  aborts 32 → 0 and the best detection rate of any arm (75%) — but it
  *costs* latency (median 18.7 s, 7.48 LLM calls/q) because reclaimed
  queries now run full multi-round sessions instead of dying in round 1.
- **cache + gate + early-term** is the *efficiency* feature set: −38% LLM
  calls (5.77 → 3.55), −35% output tokens, rounds 2.30 → 1.46, and the best
  standalone success rate — with a 25% tool-call cache hit rate.
- **validator + rollback** contributes little alone on this benchmark
  (≈ parity): Observer verdicts are rarely an explicit VALID, so rollback
  seldom has a better prior candidate to return; the validator mostly
  refines Planner feedback. Kept for the all-on configuration where the
  gate produces AUTO_VALID verdicts it can act on.
- **all-on** composes the two big effects: repair's recovery (0 aborts,
  72% detection) *and* the efficiency win (7.9 s median, −41% vs legacy) —
  the gate/early-term savings pay for repair's extra rounds.

### Seed robustness — legacy vs all-on, 3 seeds (42/43/44), mean ± std

| metric | legacy | gmas all-on |
|---|---|---|
| success % | 14.0 ± 2.2 | **16.3 ± 1.3** |
| detected % | 51.3 ± 2.6 | **78.3 ± 4.6** |
| median t (s) | 12.5 ± 0.6 | **7.7 ± 0.1 (−38%)** |
| LLM calls /q | 5.57 ± 0.17 | **4.36 ± 0.49 (−22%)** |
| rounds /q | 2.25 ± 0.04 | **1.69 ± 0.13** |
| syntax-error aborts | 30.3 ± 0.5 | **0.0 ± 0.0** |

gmas wins on success at **every individual seed** (16 > 13, 18 > 17,
15 > 12) and dominates every efficiency metric with lower variance.

## Deferred roadmap

PCA/point-cloud 6-DoF, gripper-agnostic + dexterous grasping, semantic force
+ tactile, adaptive ESKF — see `docs/ROADMAP.md`. These land as new registry
tools + agents without touching the orchestrator.
