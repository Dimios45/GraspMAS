"""Per-session tool-result cache.

Rounds of the same query re-run the same detectors on the same image (the
Planner typically refines *strategy*, not the image), so find/SAM/depth/VLM
calls repeat with identical arguments. The cache memoizes ImagePatch tool
methods for the lifetime of one Session.

Key assumption: within a session there is exactly one source image, so a
patch is uniquely identified by its absolute crop coordinates — no content
hashing needed.
"""

import time
from typing import Any

# expensive tool methods worth memoizing (GPU model invocations)
CACHED_METHODS = (
    "find", "find_part", "exists", "verify_property", "simple_query",
    "llm_query", "compute_depth", "grasp_detection",
    "grasp_detection_directional",
)


def _canon(v: Any) -> Any:
    """Canonicalize an argument for the cache key (patches → their coords)."""
    if all(hasattr(v, a) for a in ("left", "lower", "right", "upper")):
        return ("patch", v.left, v.lower, v.right, v.upper)
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    return v


def make_cached_image_patch(session):
    """Subclass ImagePatch whose tool methods are memoized in `session`."""
    from image_patch import ImagePatch  # deferred: heavy import

    def _wrap(name, fn):
        def wrapper(self, *args, **kwargs):
            key = (name, self.left, self.lower, self.right, self.upper,
                   _canon(args), _canon(tuple(sorted(kwargs.items()))))
            try:
                hit = key in session.tool_cache
            except TypeError:            # unhashable arg — just call through
                return fn(self, *args, **kwargs)
            if hit:
                session.cache_hits += 1
                try:
                    from gmas.telemetry import Telemetry
                    tel = Telemetry.active()
                    if tel is not None:
                        tel.event("tool", tool=name, cache_hit=True, wall_s=0.0)
                except Exception:
                    pass
                return session.tool_cache[key]
            t0 = time.time()
            out = fn(self, *args, **kwargs)
            session.cache_misses += 1
            session.tool_cache[key] = out
            try:
                from gmas.telemetry import Telemetry
                tel = Telemetry.active()
                if tel is not None:
                    tel.event("tool", tool=name, cache_hit=False,
                              wall_s=round(time.time() - t0, 2))
            except Exception:
                pass
            return out
        wrapper.__name__ = name
        return wrapper

    ns = {m: _wrap(m, getattr(ImagePatch, m)) for m in CACHED_METHODS
          if hasattr(ImagePatch, m)}
    return type("CachedImagePatch", (ImagePatch,), ns)
